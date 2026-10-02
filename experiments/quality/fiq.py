# SPDX-License-Identifier: MIT
r"""No-reference face image quality (FIQ) of the original and reconstructed crops.

The measure follows SER-FIQ (Terhoerst et al., CVPR 2020) with the stochasticity
moved from the network to the input: the FIQ of an image is the mean pairwise
cosine of the ``arcface_antelopev2`` embeddings of ``T = 10`` mildly augmented copies
(horizontal flip with p = 0.5, a random crop of 90-100 % of the side resized back,
gamma 0.85-1.15 and brightness 0.92-1.08). The same augmentation parameters are
applied to an original crop and to every reconstruction of it. All images are
resized to the 112 px matcher input (bilinear) before augmentation.

Per (dataset, res, budget) cell of the grid {112, 224} px x {1024, 512} B, the
sample is ``n = 250`` crops drawn (seed 0) from the crops that exist for every
stored baseline codec of the cell; the augmentation parameters use seed 1. Baseline
reconstructions are decoded from the stored bitstreams
(:func:`face1kb.baselines.decode_file`, with the decoded caches); the two face1kb
codecs are encoded and decoded on the fly with the released weights
(``paper_compat=True``, i.e. the stored paper bitstreams).

Output (``--out-dir``, default ``$FACE1KB_OUTPUT_ROOT/quality``)::

    fiq_<dataset>.csv   dataset, res, budget, codec, n, fiq_mean, fiq_median,
                        fiq_drop_vs_original   (codec "original" = the crops)

``fiq_report.py`` renders the table and figure. With ``--codecs`` (or a subset of
``--datasets``) the new rows replace the matching (dataset, codec) rows of existing
CSVs in ``out-dir``. Needs a CUDA GPU.

Examples
--------
    python experiments/quality/fiq.py
    python experiments/quality/fiq.py --datasets colorferet --cells 112/1024
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import face1kb
from face1kb import config, fr
from face1kb.baselines import decode_file, extension
from face1kb.data.cli_utils import parse_list, setup_logging

log = logging.getLogger("quality.fiq")

#: Matcher of the FIQ measure.
MATCHER = "arcface_antelopev2"
#: Matcher input size (px).
EMB_RES = 112
#: (res, budget) cells.
GRID = ((112, 1024), (112, 512), (224, 1024), (224, 512))
#: Baseline codecs read from disk (the sample is drawn from their common crops).
DISK_CODECS = (
    "jpeg",
    "jpeg2000",
    "webp",
    "jpeg_xl",
    "avif",
    "heif",
    "jpeg_fzt",
    "jpeg_ai",
)
#: face1kb codecs, encoded on the fly.
OURS = ("ours_fast", "ours_accurate")
#: Embedding batch size.
EMBED_BATCH = 256


# ---------------------------------------------------------------- sampling
def _crop_keys(folder: Path, ext: str) -> set[tuple[str, str]]:
    return {(p.parent.name, p.stem) for p in folder.rglob(f"*{ext}")}


def present_disk_codecs(dataset: str, res: int, budget: int) -> list[str]:
    """Baseline codecs with a compressed folder for the cell."""
    return [
        c
        for c in DISK_CODECS
        if config.compressed_dir(dataset, res, budget, c).is_dir()
    ]


def draw_sample(dataset: str, res: int, budget: int, codecs, n: int, seed: int):
    """Seeded shuffle of the crops present for every codec; the first ``n``."""
    common = _crop_keys(config.aligned_dir(dataset, res), ".png")
    for c in codecs:
        common &= _crop_keys(
            config.compressed_dir(dataset, res, budget, c), extension(c)
        )
    ordered = sorted(common)
    np.random.default_rng(seed).shuffle(ordered)
    return ordered[: min(n, len(ordered))]


# ---------------------------------------------------------------- augmentation
def aug_params(n_images: int, t: int, seed: int) -> list[list[dict]]:
    """``t`` augmentation parameter sets per image (drawn in image-major order)."""
    rng = np.random.default_rng(seed)
    return [
        [
            {
                "flip": bool(rng.random() < 0.5),
                "scale": float(rng.uniform(0.90, 1.00)),
                "dx": float(rng.random()),
                "dy": float(rng.random()),
                "gamma": float(rng.uniform(0.85, 1.15)),
                "bright": float(rng.uniform(0.92, 1.08)),
            }
            for _ in range(t)
        ]
        for _ in range(n_images)
    ]


def apply_aug(img: np.ndarray, p: dict) -> np.ndarray:
    """Apply one parameter set to an ``HxWx3`` uint8 image."""
    import cv2  # noqa: PLC0415

    out = img
    if p["flip"]:
        out = np.ascontiguousarray(out[:, ::-1, :])
    h, w = out.shape[:2]
    ch, cw = max(1, int(round(h * p["scale"]))), max(1, int(round(w * p["scale"])))
    if ch < h or cw < w:
        y0, x0 = int(round(p["dy"] * (h - ch))), int(round(p["dx"] * (w - cw)))
        out = cv2.resize(
            out[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR
        )
    f = out.astype(np.float32) / 255.0
    f = np.power(np.clip(f, 0, 1), p["gamma"]) * p["bright"]
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


def image_fiq(emb: np.ndarray) -> float:
    """Mean pairwise cosine of the embeddings of one image's augmented copies."""
    en = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim = en @ en.T
    return float(sim[np.triu_indices(sim.shape[0], k=1)].mean())


# ---------------------------------------------------------------- reconstruction
class Reconstructor:
    """Original crops, stored baseline reconstructions and on-the-fly face1kb ones."""

    def __init__(self, dataset: str, device: str):
        self.dataset = dataset
        self.device = device
        self._codecs: dict = {}

    def original(self, res: int, subject: str, stem: str) -> np.ndarray:
        from PIL import Image  # noqa: PLC0415

        path = config.aligned_dir(self.dataset, res) / subject / f"{stem}.png"
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def __call__(self, codec, res, budget, subject, stem) -> np.ndarray | None:
        if codec == "original":
            return self.original(res, subject, stem)
        try:
            if codec in OURS:
                c = self._codecs.get(codec)
                if c is None:
                    c = self._codecs[codec] = face1kb.load(
                        config.OURS_CODECS[codec], device=self.device
                    )
                data = c.encode(
                    self.original(res, subject, stem), budget, paper_compat=True
                )
                return c.decode(data)
            path = (
                config.compressed_dir(self.dataset, res, budget, codec)
                / subject
                / f"{stem}{extension(codec)}"
            )
            if not path.exists():
                return None
            dec = decode_file(path, res=res, device=self.device, cache=True)
            return dec if dec.shape[:2] == (res, res) else None
        except Exception as exc:  # noqa: BLE001 - a broken crop is left out
            log.warning(
                "  reconstruction failed %s %d/%d %s/%s: %s",
                codec,
                res,
                budget,
                subject,
                stem,
                exc,
            )
            return None


def score_codec(recon, embedder, codec, res, budget, samples, aug, t) -> dict:
    """``{sample index: FIQ}`` of one codec over the sample."""
    import cv2  # noqa: PLC0415

    crops, owners = [], []
    for i, (subj, stem) in enumerate(samples):
        tile = recon(codec, res, budget, subj, stem)
        if tile is None:
            continue
        if tile.shape[:2] != (EMB_RES, EMB_RES):
            tile = cv2.resize(tile, (EMB_RES, EMB_RES), interpolation=cv2.INTER_LINEAR)
        tile = np.ascontiguousarray(tile[:, :, :3], dtype=np.uint8)
        for k in range(t):
            crops.append(apply_aug(tile, aug[i][k]))
            owners.append(i)
    if not crops:
        return {}
    emb = np.concatenate(
        [
            embedder.embed(np.stack(crops[s : s + EMBED_BATCH]))
            for s in range(0, len(crops), EMBED_BATCH)
        ],
        0,
    ).astype(np.float32)
    owners = np.asarray(owners)
    return {int(i): image_fiq(emb[owners == i]) for i in np.unique(owners)}


def run_cell(dataset, res, budget, *, n, t, seed, embedder, recon, codec_filter=None):
    """FIQ rows (original first) of one grid cell."""
    disk = present_disk_codecs(dataset, res, budget)
    if codec_filter is not None:
        disk = [c for c in disk if c in codec_filter]
    samples = draw_sample(dataset, res, budget, disk, n, seed)
    aug = aug_params(len(samples), t, seed + 1)
    codecs = disk + [c for c in OURS if codec_filter is None or c in codec_filter]
    log.info("[%s %dpx/%dB] n=%d codecs=%s", dataset, res, budget, len(samples), codecs)
    orig = score_codec(recon, embedder, "original", res, budget, samples, aug, t)
    if not orig:
        log.warning("  no original crops: cell skipped")
        return []
    rows = [
        {
            "dataset": dataset,
            "res": res,
            "budget": budget,
            "codec": "original",
            "n": len(orig),
            "fiq_mean": float(np.mean(list(orig.values()))),
            "fiq_median": float(np.median(list(orig.values()))),
            "fiq_drop_vs_original": 0.0,
        }
    ]
    for c in codecs:
        per = score_codec(recon, embedder, c, res, budget, samples, aug, t)
        if not per:
            log.warning("  %s: no decodable crops", c)
            continue
        vals = np.array(list(per.values()))
        shared = [i for i in per if i in orig]
        drop = float(np.mean([orig[i] - per[i] for i in shared])) if shared else np.nan
        rows.append(
            {
                "dataset": dataset,
                "res": res,
                "budget": budget,
                "codec": c,
                "n": int(vals.size),
                "fiq_mean": float(vals.mean()),
                "fiq_median": float(np.median(vals)),
                "fiq_drop_vs_original": drop,
            }
        )
        log.info("  %s: n=%d mean=%.4f drop=%+.4f", c, vals.size, vals.mean(), drop)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Measure FIQ over the grid and write fiq_<dataset>.csv."""
    import pandas as pd  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default="colorferet,kk")
    ap.add_argument(
        "--cells",
        default=",".join(f"{r}/{b}" for r, b in GRID),
        help="comma list of res/budget cells",
    )
    ap.add_argument("--codecs", default="", help="restrict to these codecs and merge")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--t", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--matcher", default=MATCHER)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    setup_logging()

    datasets = parse_list(args.datasets)
    cells = [tuple(int(v) for v in c.split("/")) for c in parse_list(args.cells)]
    cfilter = set(parse_list(args.codecs)) or None
    embedder = fr.load(args.matcher, device=args.device)
    rows = []
    for ds in datasets:
        recon = Reconstructor(ds, args.device)
        for res, budget in cells:
            rows += run_cell(
                ds,
                res,
                budget,
                n=args.n,
                t=args.t,
                seed=args.seed,
                embedder=embedder,
                recon=recon,
                codec_filter=cfilter,
            )
    df = pd.DataFrame(rows)
    out_dir = args.out_dir or config.output_dir("quality")
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfilter is not None or set(datasets) != set(config.DATASETS):
        # replace the matching (dataset, codec) rows of the existing CSVs
        old = [
            pd.read_csv(out_dir / f"fiq_{ds}.csv")
            for ds in config.DATASETS
            if (out_dir / f"fiq_{ds}.csv").exists()
        ]
        old_df = pd.concat(old, ignore_index=True) if old else pd.DataFrame()
        keys = {(ds, c) for ds in datasets for c in (cfilter or set(df.codec))}
        if not old_df.empty:
            old_df = old_df[
                ~old_df.apply(lambda r: (r["dataset"], r["codec"]) in keys, axis=1)
            ]
        new = df[df.apply(lambda r: (r["dataset"], r["codec"]) in keys, axis=1)]
        df = pd.concat([old_df, new], ignore_index=True)
    for ds in config.DATASETS:
        part = df[df.dataset == ds] if len(df) else df
        if len(part):
            part.to_csv(out_dir / f"fiq_{ds}.csv", index=False)
            log.info("wrote %s (%d rows)", out_dir / f"fiq_{ds}.csv", len(part))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
