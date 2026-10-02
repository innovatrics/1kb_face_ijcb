# SPDX-License-Identifier: MIT
r"""ISO/IEC 29794-5 annex study, stage 1: the compress -> decode -> embed sweep.

A **cell** is one (codec, resolution, colour, manipulation, codec flags, 1024 B
budget) configuration. For every crop of the evaluation subset (``prep.py``) a cell
takes the cached crop at its resolution, applies the manipulation, encodes it to the
budget, decodes it, resizes it to 112 px and embeds it. Nothing is written in between;
only the embedding arrays and one statistics row per cell remain::

    WORK_ROOT/<dataset>/annex/emb/<cell_id>__<model>.npy   (N, 512) float16
    WORK_ROOT/<dataset>/annex/cells_<stage>_<pid>.parquet  one row per cell

``cell_id`` is ``<codec>_r<res>_<colour>_<manip>_<flags>_b1024`` with the flags as
sorted ``<name><value>`` pairs joined by ``-`` (``def`` without flags). The statistics
row holds the cell definition, ``bytes_median`` (median bytes of the fitted encodes),
``bytes_fixed_q`` (median bytes at a fixed quality: 50, JPEG 2000 rate 100),
``fit_rate`` (share of crops that fit the budget) and ``n``.

Cell axes: resolution 56, 64, 80, 96, 112, 168, 224; colour ``color`` or ``gray``
(a genuine one-channel encode); manipulation ``none``, ``mean`` (3 x 3 box blur of the
whole crop), ``rectangle_mean`` (blur outside a hard rectangle around the five
template landmarks, grown by 20 % of the resolution on each side) and
``ofiq_landmarks`` (blur outside the hard face-contour polygon of the 106-point
landmarks). The manipulation follows ``helpers/manipulate.py`` of
github.com/dasec/1kB-FaceImage and is applied after the resize.

Encoding: the largest setting that fits the budget is found by a binary search over
quality 2..94 (step 2), or over a geometric JPEG 2000 rate ladder from 400 down to
about 2 (3 % steps), warm-started from the previous crop's answer
(:func:`face1kb.baselines.search.hinted_binary_search_fit`). JPEG uses
``optimize=True``; its ``arithmetic`` flag re-codes the Huffman stream losslessly with
``jpegtran -arithmetic`` (libjpeg-turbo). WebP cells with an ``sns`` flag use the
``cwebp`` command-line encoder with its own ``-size`` search, since Pillow does not
expose ``sns``. JPEG XL effort 10 runs as effort 9 (libjxl exposes 10 only as an
expert option). Missing ``jpegtran`` or ``cwebp`` is an error.

Stages (``--stage``):

* ``baseline``: uncompressed crops at every resolution (``none_r<res>_...``), the
  references of the self-similarity objective; 5 matchers.
* ``exp1``: our recommended configuration per codec, the 10 Annex E/F
  configurations (JPEG and HEIF are identical in both annexes) and a ``cwebp``
  control; the four anchors plus ``cvlface_vit_b`` (the annexes' matcher).
* ``A``: the 336-cell grid (6 codecs x 7 resolutions x 2 colours x 4
  manipulations, default flags); ``arcface_antelopev2`` and ``cvlface_vit_b``; the
  paper runs it on the first 2000 crops (``--limit 2000``).
* ``B`` / ``confirm``: the cells listed by ``stageb.py`` (flag sweep around the
  Stage-A winners; the winners at the full population with all 5 matchers).
* ``jpegai``: the JPEG-AI arm, 168/224 px x colour/grey x 4 manipulations, on the
  first ``--limit`` crops (500 in the paper), in this process on the GPU. The target
  bit rate is warm-started from the previous crop
  (:func:`face1kb.baselines.jpeg_ai.hinted_fit`); a greyscale crop is fed to the
  encoder as grey RGB.

``--limit N`` runs the first ``N`` crops (the subset is sorted by subject, so a prefix
keeps whole subjects) and fills the remaining rows with NaN. A cell is skipped when
all its embedding arrays exist with at least ``N`` finite rows and its statistics row
exists. Compression runs in ``--workers`` CPU processes; the matchers run on
``--device``.

Cost (one RTX 2080 Ti, 24 CPU workers): ``exp1`` about 1 h per dataset, ``A`` a few
hours, ``B`` and ``confirm`` about 1 h each, ``jpegai`` about 8 GPU-hours (2-3 s per
encode, 500 crops x 16 cells).

Examples
--------
    python experiments/annex/sweep.py --stage baseline --dataset colorferet
    python experiments/annex/sweep.py --stage A --dataset colorferet --limit 2000
    python experiments/annex/sweep.py --stage jpegai --dataset colorferet --limit 500
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from face1kb import config
from face1kb.baselines.classical import PIL_FORMATS
from face1kb.baselines.classical import decode as decode_image
from face1kb.baselines.search import hinted_binary_search_fit
from face1kb.data import ARCFACE_DST_112
from face1kb.data.cli_utils import parse_list, setup_logging
from face1kb.eval.embeddings import ANCHOR_MODELS

log = logging.getLogger("annex.sweep")

BUDGET = 1024
MODEL_INPUT = 112
#: Fixed quality of the "bytes at a fixed quality" statistic (JPEG 2000: rate).
FIXED_Q = {"jpeg2000": 100, None: 50}
#: JPEG 2000 rate ladder: geometric, ~3 % of the file size per step, down to ratio
#: ~2 (at 56 px a 1024 B budget is about a third of a bit per pixel).
J2K_RATES = sorted({round(400 * 0.97**i, 2) for i in range(176)}, reverse=True)
#: Quality grid of the other codecs.
QUALITIES = list(range(2, 96, 2))

RESOLUTIONS = (56, 64, 80, 96, 112, 168, 224)
COLORS = ("color", "gray")
MANIPS = ("none", "rectangle_mean", "ofiq_landmarks", "mean")
CODECS = ("jpeg", "jpeg2000", "webp", "jpeg_xl", "avif", "heif")
JPEGAI_RESOLUTIONS = (168, 224)

#: Our recommended resolution per codec (colour, no manipulation, default flags).
OURS = {
    "jpeg": 96,
    "webp": 96,
    "avif": 112,
    "heif": 96,
    "jpeg_xl": 112,
    "jpeg2000": 224,
}

#: The unique Annex E/F configurations: (annex, codec, res, colour, manip, flags).
ANNEX = [
    ("E", "jpeg", 96, "gray", "rectangle_mean", {"smooth": 30, "arithmetic": True}),
    ("E", "jpeg2000", 56, "color", "rectangle_mean", {"num_resolutions": 3}),
    ("E", "jpeg_xl", 64, "color", "rectangle_mean", {"effort": 10}),
    ("E", "avif", 56, "color", "rectangle_mean", {"subsampling": "4:2:0", "speed": 1}),
    ("E", "heif", 96, "color", "rectangle_mean", {"chroma_downsampling": "average"}),
    ("E", "webp", 64, "color", "rectangle_mean", {"method": 6, "sns": 40}),
    ("F", "jpeg2000", 56, "gray", "rectangle_mean", {"num_resolutions": 5}),
    ("F", "jpeg_xl", 56, "gray", "ofiq_landmarks", {"effort": 10}),
    ("F", "avif", 56, "gray", "none", {"subsampling": "4:0:0", "speed": 1}),
    ("F", "webp", 80, "gray", "ofiq_landmarks", {"method": 6, "sns": 40}),
]

ANCHORS = ANCHOR_MODELS
#: The annexes' matcher (CVLface AdaFace ViT-B KP-RPE, WebFace12M).
THEIRS = "cvlface_vit_b"
#: Embedding batch sizes (CVLface ViT-B needs a small batch on an 11 GB GPU).
EMB_BATCH = {"cvlface_vit_b": 32}
EMB_BATCH_DEFAULT = 64

# 106-point contour: jaw 0-32, then the two brow arcs reversed, closing the polygon
# over the forehead.
CONTOUR = list(range(33)) + list(range(105, 96, -1)) + list(range(51, 42, -1))


# ------------------------------------------------------------------ locations
def annex_dir(dataset: str) -> Path:
    """Annex work folder of ``dataset``: ``WORK_ROOT/<dataset>/annex``."""
    return config.work_dir(dataset) / "annex"


def cache_dir(dataset: str, cache_root: str | None = None) -> Path:
    """Crop cache of ``prep.py`` (``annex/cache`` or ``<cache-root>/<dataset>``)."""
    return Path(cache_root) / dataset if cache_root else annex_dir(dataset) / "cache"


def cells_json(name: str, path: str | None = None) -> Path:
    """Cell list written by ``stageb.py`` (``OUTPUT_ROOT/annex/<name>``)."""
    return Path(path) if path else config.output_dir("annex", name)


# ------------------------------------------------------------------ manipulations
def mask_rect(res: int) -> np.ndarray:
    """Hard rectangle over the 5 template landmarks, grown by 20 % of ``res``."""
    lmk = ARCFACE_DST_112 * (res / 112.0)
    g = res / 5.0
    x0, y0 = np.maximum(lmk.min(0) - g, 0).astype(int)
    x1, y1 = np.minimum(lmk.max(0) + g, res - 1).astype(int)
    m = np.zeros((res, res), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def mask_contour(lmk106: np.ndarray, res: int) -> np.ndarray:
    """Hard face-contour polygon of the 106-point landmarks, scaled to ``res``."""
    poly = (lmk106[CONTOUR] * (res / 224.0)).round().astype(np.int32)
    m = np.zeros((res, res), np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m


def manipulate(img: np.ndarray, manip: str, lmk106: np.ndarray) -> np.ndarray:
    """Apply one annex manipulation to a crop at its final resolution."""
    if manip == "none":
        return img
    blur = cv2.blur(img, (3, 3))
    if manip == "mean":
        return blur
    res = img.shape[0]
    m = mask_rect(res) if manip == "rectangle_mean" else mask_contour(lmk106, res)
    return np.where(m[:, :, None] == 1, img, blur)


# ------------------------------------------------------------------ encoders
_PLUGINS = False


def _register_plugins() -> None:
    global _PLUGINS  # noqa: PLW0603
    if not _PLUGINS:
        import pillow_heif  # noqa: PLC0415
        import pillow_jxl  # noqa: F401, PLC0415  (registers JPEG XL on import)

        pillow_heif.register_heif_opener()
        _PLUGINS = True


def _save(img: Image.Image, fmt: str, **kw) -> tuple[int, bytes]:
    b = io.BytesIO()
    img.save(b, format=fmt, **kw)
    return b.getbuffer().nbytes, b.getvalue()


def jpegtran_arith(data: bytes) -> bytes:
    """Losslessly re-code a Huffman JPEG with arithmetic coding (``jpegtran``)."""
    p = subprocess.run(
        ["jpegtran", "-arithmetic", "-copy", "none"],
        input=data,
        capture_output=True,
        check=False,
    )
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(
            "jpegtran -arithmetic failed (libjpeg-turbo built with arithmetic "
            f"coding is required): {p.stderr.decode(errors='replace').strip()}"
        )
    return p.stdout


def cwebp(img: Image.Image, budget: int, flags: dict) -> tuple[int, bytes]:
    """Encode with ``cwebp -size <budget>`` (the only encoder exposing ``-sns``)."""
    with tempfile.TemporaryDirectory(prefix="annex_webp_") as td:
        src, dst = os.path.join(td, "in.png"), os.path.join(td, "out.webp")
        img.save(src)
        cmd = [
            "cwebp",
            "-quiet",
            "-size",
            str(budget),
            "-m",
            str(flags.get("method", 6)),
        ]
        if "sns" in flags:
            cmd += ["-sns", str(flags["sns"])]
        subprocess.run([*cmd, src, "-o", dst], capture_output=True, check=False)
        data = Path(dst).read_bytes() if os.path.exists(dst) else b""
    return len(data), data


def encoder(codec: str, img: Image.Image, flags: dict):
    """Return ``(encode(setting) -> (size, data), settings ascending in size)``."""
    _register_plugins()
    f = dict(flags)
    f.pop("sns", None)
    if codec == "jpeg":
        arith = f.pop("arithmetic", False)

        def enc(v):
            n, d = _save(img, "JPEG", quality=v, optimize=True, **f)
            if arith:
                d = jpegtran_arith(d)
                n = len(d)
            return n, d

        return enc, QUALITIES
    if codec == "jpeg2000":
        return (
            lambda v: _save(
                img, "JPEG2000", quality_mode="rates", quality_layers=[v], **f
            ),
            J2K_RATES,
        )
    if codec == "jpeg_xl" and "effort" in f:
        f["effort"] = min(f["effort"], 9)  # effort 10 is an expert option of libjxl
    if codec == "webp":
        f.setdefault("method", 6)
    return (lambda v: _save(img, PIL_FORMATS[codec], quality=v, **f)), QUALITIES


def check_tools(cells: list[dict]) -> None:
    """Fail early when a cell needs ``jpegtran`` or ``cwebp`` and it is missing."""
    need = set()
    for c in cells:
        if c["codec"] == "jpeg" and c["flags"].get("arithmetic"):
            need.add("jpegtran")
        if c["codec"] == "webp" and "sns" in c["flags"]:
            need.add("cwebp")
    missing = sorted(t for t in need if shutil.which(t) is None)
    if missing:
        raise SystemExit(f"required command-line tools not found: {missing}")


# ------------------------------------------------------------------ cells
def cell_id(c: dict) -> str:
    """Stable, file-name-safe identifier of a cell."""
    fl = "-".join(f"{k}{v}" for k, v in sorted(c["flags"].items())) or "def"
    return f"{c['codec']}_r{c['res']}_{c['color']}_{c['manip']}_{fl}_b{c['budget']}"


def make_cell(codec, res, color, manip, flags=None, arm="grid") -> dict:
    """Cell definition (budget 1024 B)."""
    return {
        "codec": codec,
        "res": res,
        "color": color,
        "manip": manip,
        "flags": flags or {},
        "budget": BUDGET,
        "arm": arm,
    }


def cells_for(stage: str, cells_file: str | None = None) -> list[dict]:
    """Cell list of a stage."""
    if stage == "baseline":
        return [
            make_cell("none", r, "color", "none", arm="baseline") for r in RESOLUTIONS
        ]
    if stage == "A":
        return [
            make_cell(c, r, col, m)
            for c, r, col, m in itertools.product(CODECS, RESOLUTIONS, COLORS, MANIPS)
        ]
    if stage == "exp1":
        out = [make_cell(c, r, "color", "none", arm="ours") for c, r in OURS.items()]
        out += [
            make_cell(c, r, col, m, f, arm=f"annex{ax}")
            for ax, c, r, col, m, f in ANNEX
        ]
        # engine-matched control of the cwebp-encoded annex arms
        out.append(
            make_cell(
                "webp", OURS["webp"], "color", "none", {"method": 6, "sns": 50}, "ours"
            )
        )
        return out
    if stage == "jpegai":
        return [
            make_cell("jpeg_ai", r, c, m)
            for r, c, m in itertools.product(JPEGAI_RESOLUTIONS, COLORS, MANIPS)
        ]
    if stage in ("B", "confirm"):
        name = "stage_b_cells.json" if stage == "B" else "confirm_cells.json"
        path = cells_json(name, cells_file)
        if not path.is_file():
            raise SystemExit(f"{path} not found; run stageb.py first")
        return json.loads(path.read_text())
    raise ValueError(stage)


def models_for(stage: str) -> list[str]:
    """Matchers of a stage."""
    if stage in ("A", "B"):
        return ["arcface_antelopev2", THEIRS]
    return [*ANCHORS, THEIRS]


# ------------------------------------------------------------------ per-crop work
_W: dict = {}


def _init(cache: str) -> None:
    """Per-worker state; the crop caches are opened lazily per resolution."""
    global _W  # noqa: PLW0603
    _W = {
        "cache": Path(cache),
        "crops": {},
        "hints": {},
        "lmk": np.load(Path(cache) / "lmk106.npy", mmap_mode="r"),
    }


def _crops(res: int) -> np.ndarray:
    if res not in _W["crops"]:
        _W["crops"][res] = np.load(_W["cache"] / f"crops_{res}.npy", mmap_mode="r")
    return _W["crops"][res]


def _to_input(img: np.ndarray) -> np.ndarray:
    if img.shape[0] == MODEL_INPUT:
        return img
    return cv2.resize(img, (MODEL_INPUT, MODEL_INPUT), interpolation=cv2.INTER_LINEAR)


def _one(task: tuple[dict, int]) -> tuple[int, int, bool, np.ndarray]:
    """Manipulate, encode to the budget, decode, resize to 112 px."""
    c, i = task
    arr = manipulate(np.asarray(_crops(c["res"])[i]), c["manip"], _W["lmk"][i])
    if c["codec"] == "none":  # uncompressed reference at this resolution
        return 0, 0, True, _to_input(arr)
    img = Image.fromarray(arr)
    if c["color"] == "gray":
        img = img.convert("L")
    flags, key = c["flags"], cell_id(c)
    if c["codec"] == "webp" and "sns" in flags:
        size, data = cwebp(img, c["budget"], flags)
        fit, fixed = 0 < size <= c["budget"], size
    else:
        enc, settings = encoder(c["codec"], img, flags)
        idx, r = hinted_binary_search_fit(
            enc, settings, c["budget"], _W["hints"].get(key)
        )
        size, data, fit = r.size, r.payload, r.fitted
        _W["hints"][key] = idx
        fixed = enc(FIXED_Q.get(c["codec"], FIXED_Q[None]))[0]
    if not data:
        return size, fixed, False, np.zeros((MODEL_INPUT, MODEL_INPUT, 3), np.uint8)
    return size, fixed, fit, _to_input(decode_image(data))


# ------------------------------------------------------------------ driver
def _embed(model, crops: np.ndarray, name: str) -> np.ndarray:
    bs = EMB_BATCH.get(name, EMB_BATCH_DEFAULT)
    return np.concatenate(
        [model.embed(list(crops[j : j + bs])) for j in range(0, len(crops), bs)]
    )


def _save_emb(path: Path, emb: np.ndarray, full: int) -> None:
    arr = np.full((full, emb.shape[1]), np.nan, np.float16)
    arr[: len(emb)] = emb.astype(np.float16)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


def _todo(ds: str, cells: list[dict], models: list[str], n: int) -> list:
    """(cell, matchers to run) for every cell that is not complete."""
    emb_dir = annex_dir(ds) / "emb"
    known = set()
    for q in annex_dir(ds).glob("cells*.parquet"):
        known |= set(pd.read_parquet(q).cell_id)

    def stale(c, m) -> bool:
        p = emb_dir / f"{cell_id(c)}__{m}.npy"
        if not p.exists():
            return True
        return int(np.isfinite(np.load(p, mmap_mode="r")).all(1).sum()) < n

    todo = [(c, [m for m in models if stale(c, m)]) for c in cells]
    return [(c, m) for c, m in todo if m or cell_id(c) not in known]


def _cell_row(c: dict, sizes, fixed, fits, n: int, cid: str) -> dict:
    return {
        **c,
        "flags": json.dumps(c["flags"]),
        "cell_id": cid,
        "bytes_median": float(np.median(sizes)),
        "bytes_fixed_q": float(np.median(fixed)),
        "fit_rate": float(fits.mean()),
        "n": n,
    }


def run(
    ds: str,
    stage: str,
    cells: list[dict],
    models: list[str],
    cache: Path,
    workers: int,
    device: str | None,
    limit: int = 0,
) -> None:
    """Compress and embed every outstanding cell of a CPU stage.

    Cell ``k + 1`` is compressed by the worker pool while cell ``k`` is embedded,
    which hides the GPU work inside the compression of the next cell.
    """
    from face1kb import fr  # noqa: PLC0415

    out = annex_dir(ds) / "emb"
    out.mkdir(parents=True, exist_ok=True)
    full = len(pd.read_parquet(cache / "meta.parquet"))
    n = min(limit, full) if limit else full
    todo = _todo(ds, cells, models, n)
    log.info(
        "%s stage %s: %d of %d cells outstanding", ds, stage, len(todo), len(cells)
    )
    if not todo:
        return
    check_tools([c for c, _m in todo])
    rows = []
    rows_path = annex_dir(ds) / f"cells_{stage}_{os.getpid()}.parquet"
    ctx = multiprocessing.get_context("fork")
    with (
        ProcessPoolExecutor(
            workers, ctx, initializer=_init, initargs=(str(cache),)
        ) as ex,
        ThreadPoolExecutor(1) as pre,
    ):
        # start the workers before the matchers are loaded, so that they do not
        # inherit a copy of torch and the models
        list(ex.map(int, range(workers * 2)))
        loaded = {m: fr.load(m, device) for m in models}

        def submit(k):
            c = todo[k][0]
            return pre.submit(
                lambda: list(ex.map(_one, [(c, i) for i in range(n)], chunksize=8))
            )

        pending = submit(0)
        for k, (c, mods) in enumerate(tqdm(todo, desc=f"{ds} cells")):
            res = pending.result()
            pending = submit(k + 1) if k + 1 < len(todo) else None
            crops = np.stack([r[3] for r in res])
            sizes = np.array([r[0] for r in res], float)
            fixed = np.array([r[1] for r in res], float)
            fits = np.array([r[2] for r in res], bool)
            cid = cell_id(c)
            for m in mods:
                _save_emb(out / f"{cid}__{m}.npy", _embed(loaded[m], crops, m), full)
            rows.append(_cell_row(c, sizes, fixed, fits, n, cid))
            pd.DataFrame(rows).to_parquet(rows_path)  # rewritten after every cell


def run_jpegai(
    ds: str,
    cells: list[dict],
    models: list[str],
    cache: Path,
    device: str | None,
    limit: int,
) -> None:
    """JPEG-AI arm: encode, reconstruct and embed in this process (GPU)."""
    from face1kb import fr  # noqa: PLC0415
    from face1kb.baselines import jpeg_ai  # noqa: PLC0415

    jpeg_ai.check_setup()
    out = annex_dir(ds) / "emb"
    out.mkdir(parents=True, exist_ok=True)
    full = len(pd.read_parquet(cache / "meta.parquet"))
    n = min(limit, full) if limit else full
    todo = [
        c
        for c in cells
        if not all((out / f"{cell_id(c)}__{m}.npy").exists() for m in models)
    ]
    log.info("%s stage jpegai: %d of %d cells outstanding", ds, len(todo), len(cells))
    if not todo:
        return
    loaded = {m: fr.load(m, device) for m in models}
    lmk = np.load(cache / "lmk106.npy", mmap_mode="r")
    rows = []
    rows_path = annex_dir(ds) / f"cells_jpegai_{os.getpid()}.parquet"
    for c in todo:
        cid = cell_id(c)
        crops = np.load(cache / f"crops_{c['res']}.npy", mmap_mode="r")
        recs = np.zeros((n, MODEL_INPUT, MODEL_INPUT, 3), np.uint8)
        sizes = np.zeros(n)
        hint = None
        for i in tqdm(range(n), desc=cid):
            arr = manipulate(np.asarray(crops[i]), c["manip"], lmk[i])
            if c["color"] == "gray":
                arr = np.asarray(Image.fromarray(arr).convert("L").convert("RGB"))
            sizes[i], recs[i], bpp = _jpegai_one(jpeg_ai, arr, c["budget"], hint)
            hint = bpp or hint
        for m in models:
            _save_emb(out / f"{cid}__{m}.npy", _embed(loaded[m], recs, m), full)
        rows.append(
            {
                **c,
                "flags": json.dumps(c["flags"]),
                "cell_id": cid,
                "bytes_median": float(np.median(sizes[sizes < 1 << 20])),
                "bytes_fixed_q": float("nan"),
                "fit_rate": float((sizes <= c["budget"]).mean()),
                "n": n,
            }
        )
        pd.DataFrame(rows).to_parquet(rows_path)


def _jpegai_one(jpeg_ai, arr: np.ndarray, budget: int, hint: int | None):
    """Encode one crop at the largest plausible target that fits ``budget``.

    Returns ``(size, 112 px reconstruction, target bpp x 100)``; a crop without an
    accepted encode gives ``(2**30, zeros, 0)``.
    """
    recon: dict[int, np.ndarray] = {}

    def encode(bpp):
        data, rec = jpeg_ai.encode_bpp(arr, bpp, return_recon=True)
        if data is None or rec is None:
            return None
        recon[bpp] = rec
        return len(data), data

    fit = jpeg_ai.hinted_fit(encode, budget, arr.shape[0] * arr.shape[1], hint=hint)
    if fit is None:
        return jpeg_ai.FAILED_SIZE, np.zeros((MODEL_INPUT, MODEL_INPUT, 3), np.uint8), 0
    return fit.size, _to_input(recon[fit.setting]), fit.setting


def main(argv: list[str] | None = None) -> int:
    """Run one stage of the sweep on one dataset."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--stage",
        required=True,
        choices=["baseline", "exp1", "A", "B", "confirm", "jpegai"],
    )
    ap.add_argument("--dataset", default="colorferet", choices=config.DATASETS)
    ap.add_argument("--models", default="", help="default: the matchers of the stage")
    ap.add_argument("--cells", default=None, help="cell list of stage B / confirm")
    ap.add_argument("--only", default="", help="run only these cell ids")
    ap.add_argument("--shard", default="0/1", help="k/K: every K-th cell from k")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="first N crops (default: all; 500 for the jpegai stage; 0: all)",
    )
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--device", default=None, help="torch device of the matchers")
    ap.add_argument("--cache-root", default=None, help="prep.py --cache-root")
    args = ap.parse_args(argv)
    setup_logging()
    if (
        config.LAYOUT == "legacy"
        and config.WORK_ROOT.resolve() == config.DATA_ROOT.resolve()
    ):
        raise SystemExit(
            "FACE1KB_LAYOUT=legacy: point FACE1KB_WORK_ROOT away from the data store"
        )

    cells = cells_for(args.stage, args.cells)
    models = parse_list(args.models) or models_for(args.stage)
    if args.only:
        only = set(parse_list(args.only))
        cells = [c for c in cells if cell_id(c) in only]
    k, tot = (int(x) for x in args.shard.split("/"))
    cells = cells[k::tot]
    cache = cache_dir(args.dataset, args.cache_root)
    if not (cache / "meta.parquet").is_file():
        raise SystemExit(f"{cache}/meta.parquet not found; run prep.py first")
    log.info(
        "%s stage %s: %d cells x %d matchers",
        args.dataset,
        args.stage,
        len(cells),
        len(models),
    )
    if args.stage == "jpegai":
        limit = 500 if args.limit is None else args.limit
        run_jpegai(args.dataset, cells, models, cache, args.device, limit)
    else:
        run(
            args.dataset,
            args.stage,
            cells,
            models,
            cache,
            args.workers,
            args.device,
            args.limit or 0,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
