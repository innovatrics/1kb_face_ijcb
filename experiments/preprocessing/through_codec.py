# SPDX-License-Identifier: MIT
r"""Preprocessing measured through the codec (AI-Solutions-KK, 224 px, 1024 B).

Each preprocessing operator is applied to the aligned 224 px crop, the result is
compressed to the byte budget and decoded, and the reconstruction is resized to
112 px (bilinear) and embedded::

    aligned crop --(operator)--> --(codec @ budget)--> decode --> 112 px --> matcher

Per (operator, codec, matcher) the script reports the verification EER over the
pairs of ``pairs.parquet`` whose two members are both in the sample, and the mean
identity cosine of the reconstruction to the *unprocessed* crop (the same matcher's
embedding of the aligned crop resized to 112 px). ``delta_eer_vs_std`` is the EER
difference to the ``std`` (no preprocessing) row of the same codec and matcher.

The sample is a seeded uniform draw of ``--n`` index rows
(``numpy.random.default_rng(seed).choice(N, n, replace=False)``, sorted).

Operators come from :mod:`face1kb.data.preprocess` (``std``, ``A1``-``A4``, ``B2``
MediaPipe background flattening; ``B1``/``C1``/``C2`` need BiRefNet). Codecs:
``webp``, ``avif`` and the other classical codecs (benchmark binary search),
``jpeg_fzt`` (even qualities 2..94) and ``ours_accurate`` (face1kb-ACCURATE,
``paper_compat=True``; a failed decode gives a mid-grey frame).

Phases (``--phase``):

``encode``
    Reconstructs and caches the decoded 112 px crops of every (operator, codec) unit
    that is not cached yet (``<cache>/crops_n<n>_s<seed>_<op>_<codec>.npy``). Only
    this phase needs the codec; Ours-ACCURATE needs a GPU (about 0.6 s per crop).
``aggregate``
    Embeds the cached units with every matcher and writes
    ``OUTPUT_ROOT/accuracy/preproc_through_codec_kk.csv``.
``all``
    Both.

``--montage`` (with ``aggregate``/``all``) or ``--montage-only`` also renders
``OUTPUT_ROOT/figures/preproc_through_codec_kk.png``: operators (columns) on the
crops given with ``--montage-samples`` (input row and decoded row per crop), decoded
with the codec of lowest mean ArcFace EER. The figure shows dataset faces; keep it
local.

The paper: ``--n 800 --seed 0``, operators std,A1,A2,A3,A4,B2, codecs
webp,avif,ours_accurate, matchers arcface_antelopev2,edgeface_xs.

Examples
--------
    python experiments/preprocessing/through_codec.py --phase encode --codecs webp
    python experiments/preprocessing/through_codec.py --phase encode \
        --codecs ours_accurate --operators A1
    python experiments/preprocessing/through_codec.py --phase aggregate
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from face1kb import config
from face1kb.baselines import classical, jpeg_fzt
from face1kb.data.cli_utils import parse_list, setup_logging
from face1kb.data.preprocess import BG_FILL, OPERATORS
from face1kb.eval import verification

log = logging.getLogger("preproc_through_codec")

DATASET = "kk"
OPERATORS_PAPER: tuple[str, ...] = ("std", "A1", "A2", "A3", "A4", "B2")
CODECS_PAPER: tuple[str, ...] = ("webp", "avif", "ours_accurate")
MODELS_PAPER: tuple[str, ...] = ("arcface_antelopev2", "edgeface_xs")
CODECS: tuple[str, ...] = (*classical.CODECS, "jpeg_fzt", "ours_accurate")
EMBED_BATCH = 256
#: Embedding width assumed for matchers without a ``dim`` attribute.
EMB_DIM = 512
OUT_NAME = "preproc_through_codec_kk.csv"
COLUMNS = (
    "operator",
    "codec",
    "res",
    "budget",
    "model",
    "eer",
    "id_cos_vs_original",
    "delta_eer_vs_std",
    "n_images",
)
PRETTY_OP = {
    "std": "std (none)",
    "A1": "A1 edge-pres.",
    "A2": "A2 bilateral",
    "A3": "A3 chroma-red.",
    "A4": "A4 NLM denoise",
    "B1": "B1 bg-BiRefNet",
    "B2": "B2 bg-flatten",
    "C1": "C1 bg+denoise",
    "C2": "C2 foveated",
}
PRETTY_CODEC = {
    "jpeg": "JPEG",
    "jpeg2000": "JPEG 2000",
    "webp": "WebP",
    "jpeg_xl": "JPEG XL",
    "avif": "AVIF",
    "heif": "HEIF",
    "jpeg_fzt": "JPEG-FzT",
    "ours_accurate": "Ours-ACCURATE",
}


# ------------------------------------------------------------------ sample
def select_subset(n_all: int, n: int, seed: int) -> np.ndarray:
    """Sorted seeded uniform sample of ``n`` of the ``n_all`` index rows."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_all, min(n, n_all), replace=False))


def restrict_pairs(sample: np.ndarray, n_all: int):
    """Mated and non-mated pairs with both members in ``sample``."""
    inside = np.zeros(n_all, bool)
    inside[sample] = True
    i1, i2, lab = verification.load_pairs(DATASET)
    keep = inside[i1] & inside[i2]
    pos = keep & (lab == 1)
    neg = keep & (lab == 0)
    return (i1[pos], i2[pos]), (i1[neg], i2[neg])


def sample_tag(n: int, seed: int, limit: int = 0) -> str:
    """Cache tag of a sample: ``n<n>_s<seed>`` (``_l<limit>`` for a truncated one)."""
    return f"n{n}_s{seed}" + (f"_l{limit}" if limit else "")


def cached_pairs(cache: Path, tag: str, sample: np.ndarray, n_all: int):
    """:func:`restrict_pairs`, cached as ``pairs_<tag>.npz``."""
    path = cache / f"pairs_{tag}.npz"
    if path.exists():
        d = np.load(path)
        return (d["p1"], d["p2"]), (d["n1"], d["n2"])
    pos, neg = restrict_pairs(sample, n_all)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(path, p1=pos[0], p2=pos[1], n1=neg[0], n2=neg[1])
    return pos, neg


def to_112(rgb: np.ndarray) -> np.ndarray:
    """Bilinear resize to the 112 x 112 matcher input."""
    if rgb.shape[:2] != (112, 112):
        import cv2  # noqa: PLC0415

        rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


# ------------------------------------------------------------------ codecs
class Reconstructor:
    """Budget-fitted encode + decode of one codec (``uint8`` RGB in and out)."""

    def __init__(self, codec: str, budget: int, device: str = "cuda"):
        if codec not in CODECS:
            raise ValueError(f"unknown codec {codec!r}; choose from {CODECS}")
        self.codec, self.budget = codec, int(budget)
        self._ours = None
        if codec == "ours_accurate":
            import face1kb  # noqa: PLC0415

            self._ours = face1kb.load("accurate", device=device)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if self._ours is not None:
            data = self._ours.encode(rgb, self.budget, paper_compat=True)
            x_hat, h = self._ours.decode_tensor(data)
            if x_hat is None:  # identity-only container: mid-grey frame
                return np.full((h.res, h.res, 3), BG_FILL, dtype=np.uint8)
            x = x_hat[..., : h.res, : h.res][0].permute(1, 2, 0).cpu().numpy()
            return (x * 255).round().astype(np.uint8)
        if self.codec == "jpeg_fzt":
            data, _ = jpeg_fzt.encode_to_budget(
                rgb, self.budget, qualities=jpeg_fzt.QUALITIES_EVEN
            )
            return jpeg_fzt.decode(data, (rgb.shape[1], rgb.shape[0]))
        data, _ = classical.encode_to_budget(rgb, self.codec, self.budget)
        return classical.decode(data)


class Operators:
    """Apply preprocessing operators, loading each segmenter once."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._mediapipe = None
        self._birefnet = None

    def __call__(self, rgb: np.ndarray, op: str) -> np.ndarray:
        from face1kb.data import preprocess  # noqa: PLC0415

        if op == "B2" and self._mediapipe is None:
            self._mediapipe = preprocess.MediaPipeSegmenter()
        if op in ("B1", "C1", "C2") and self._birefnet is None:
            self._birefnet = preprocess.BiRefNetSegmenter(device=self.device)
        return preprocess.apply_operator(
            rgb, op, birefnet=self._birefnet, mediapipe=self._mediapipe
        )

    def close(self) -> None:
        if self._mediapipe is not None:
            self._mediapipe.close()
            self._mediapipe = None


# ------------------------------------------------------------------ data
def load_originals(rel_paths, sample: np.ndarray, res: int) -> list[np.ndarray]:
    """Unprocessed aligned crops of the sample (sample order)."""
    from PIL import Image  # noqa: PLC0415

    src = config.aligned_dir(DATASET, res)
    out = []
    for g in sample:
        with Image.open(src / rel_paths[g]) as im:
            out.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    return out


def crops_cache(cache: Path, tag: str, op: str, codec: str) -> Path:
    """Cache file of one (operator, codec) unit: ``crops_<tag>_<op>_<codec>.npy``."""
    return cache / f"crops_{tag}_{op}_{codec}.npy"


def embed(model, crops: list[np.ndarray]) -> np.ndarray:
    """Embed 112 px crops in batches of :data:`EMBED_BATCH`."""
    out = np.empty((len(crops), getattr(model, "dim", EMB_DIM)), dtype=np.float32)
    for s in range(0, len(crops), EMBED_BATCH):
        chunk = crops[s : s + EMBED_BATCH]
        out[s : s + len(chunk)] = np.asarray(model.embed(chunk), dtype=np.float32)
    return out


# ------------------------------------------------------------------ phases
def phase_encode(args, rel_paths, sample, cache: Path) -> None:
    """Reconstruct and cache the decoded 112 px crops of each missing unit."""
    from tqdm import tqdm  # noqa: PLC0415

    todo = [
        (op, c)
        for c in args.codecs
        for op in args.operators
        if not crops_cache(cache, args.tag, op, c).exists()
    ]
    if not todo:
        log.info("[encode] all requested units are cached")
        return
    cache.mkdir(parents=True, exist_ok=True)
    originals = load_originals(rel_paths, sample, args.res)
    ops = Operators(args.device)
    recon = {}
    try:
        for op, codec in todo:
            if codec not in recon:
                recon[codec] = Reconstructor(codec, args.budget, args.device)
            t0 = time.perf_counter()
            crops = np.empty((len(sample), 112, 112, 3), dtype=np.uint8)
            for i in tqdm(range(len(sample)), desc=f"{op:4s} x {codec}", unit="img"):
                crops[i] = to_112(recon[codec](ops(originals[i], op)))
            path = crops_cache(cache, args.tag, op, codec)
            tmp = path.with_name(path.stem + ".part.npy")
            np.save(tmp, crops)
            os.replace(tmp, path)
            dt = time.perf_counter() - t0
            log.info(
                "[encode] cached %s x %s (%.0f s, %.0f ms/img)",
                op,
                codec,
                dt,
                dt / len(sample) * 1000,
            )
    finally:
        ops.close()


def phase_aggregate(args, rel_paths, sample, cache: Path) -> pd.DataFrame:
    """Embed the cached units and write the aggregate CSV."""
    from face1kb import fr  # noqa: PLC0415

    missing = [
        (op, c)
        for c in args.codecs
        for op in args.operators
        if not crops_cache(cache, args.tag, op, c).exists()
    ]
    if missing:
        raise SystemExit(f"[aggregate] missing crop caches for {missing}")
    n_all = len(rel_paths)
    pos_idx, neg_idx = cached_pairs(cache, args.tag, sample, n_all)
    log.info(
        "[aggregate] N=%d mated=%d non-mated=%d",
        len(sample),
        *(pos_idx[0].size, neg_idx[0].size),
    )
    orig112 = [to_112(c) for c in load_originals(rel_paths, sample, args.res)]
    models = {m: fr.load(m, device=args.device) for m in args.models}
    ref = {m: embed(models[m], orig112) for m in args.models}
    rows = []
    for codec in args.codecs:
        for op in args.operators:
            crops = np.load(crops_cache(cache, args.tag, op, codec))
            crops_list = [crops[i] for i in range(len(crops))]
            for m in args.models:
                rec = embed(models[m], crops_list)
                rn, on = verification.unit_rows(rec), verification.unit_rows(ref[m])
                id_cos = float((rn * on).sum(axis=1).mean())
                emb = np.full((n_all, rec.shape[1]), np.nan, dtype=np.float32)
                emb[sample] = rec
                eer = verification.eer(
                    verification.cosine_scores(emb, *pos_idx),
                    verification.cosine_scores(emb, *neg_idx),
                )
                rows.append(
                    {
                        "operator": op,
                        "codec": codec,
                        "res": args.res,
                        "budget": args.budget,
                        "model": m,
                        "eer": eer,
                        "id_cos_vs_original": id_cos,
                        "n_images": len(sample),
                    }
                )
                log.info(
                    "  %-4s %-14s %-20s eer=%6.3f%% id_cos=%.4f",
                    op,
                    codec,
                    m,
                    eer * 100,
                    id_cos,
                )
    df = pd.DataFrame(rows)
    std = df[df.operator == "std"].set_index(["codec", "model"])["eer"].to_dict()
    df["delta_eer_vs_std"] = [
        r.eer - std.get((r.codec, r.model), np.nan) for r in df.itertuples()
    ]
    return df[list(COLUMNS)]


# ------------------------------------------------------------------ montage
def parse_samples(spec: str | None) -> list[tuple[str, str]]:
    """``"subject/stem[:label],..."`` -> ``[(rel_path, label)]``."""
    out = []
    for item in parse_list(spec):
        rel, _, label = item.partition(":")
        rel = rel if rel.endswith(".png") else rel + ".png"
        out.append((rel, label or Path(rel).parent.name))
    return out


def build_montage(df: pd.DataFrame, args, out: Path) -> None:
    """Operators applied to a few crops and their reconstructions (local figure)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    samples = parse_samples(args.montage_samples)
    if not samples:
        raise SystemExit("--montage-samples is required for the montage")
    d = df[df.model == "arcface_antelopev2"]
    mean_eer = d.groupby("codec")["eer"].mean()
    best = mean_eer.idxmin()
    log.info("montage codec = %s (mean ArcFace EER %.3f%%)", best, mean_eer.min() * 100)
    recon = Reconstructor(best, args.budget, args.device)
    ops = Operators(args.device)
    nr, nc = len(samples) * 2, len(args.operators)
    fig, axes = plt.subplots(
        nr, nc, figsize=(1.55 * nc, 1.55 * nr + 0.4), squeeze=False
    )
    try:
        for si, (rel, label) in enumerate(samples):
            with Image.open(config.aligned_dir(DATASET, args.res) / rel) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            for ci, op in enumerate(args.operators):
                proc = ops(rgb, op)
                for k, img in enumerate((proc, recon(proc))):
                    ax = axes[si * 2 + k][ci]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.imshow(img)
                    if si == 0 and k == 0:
                        ax.set_title(PRETTY_OP.get(op, op), fontsize=7.5, pad=3)
                    if ci == 0:
                        ax.set_ylabel(
                            f"{label}\n{'input' if k == 0 else 'decoded'}", fontsize=7
                        )
    finally:
        ops.close()
    fig.suptitle(
        f"Preprocessing through {PRETTY_CODEC.get(best, best)} — KK "
        f"{args.res}px @ {args.budget}B",
        fontsize=10,
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("wrote %s", out)


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=("encode", "aggregate", "all"), default="all")
    ap.add_argument("--n", type=int, default=800, help="sample size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--limit", type=int, default=0, help="use the first N of the sample"
    )
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--operators", default=",".join(OPERATORS_PAPER))
    ap.add_argument("--codecs", default=",".join(CODECS_PAPER))
    ap.add_argument("--models", default=",".join(MODELS_PAPER))
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="decoded-crop cache (default: WORK_ROOT/kk/preproc_through_codec)",
    )
    ap.add_argument("--out", default=None, help="default: OUTPUT_ROOT/accuracy/...")
    ap.add_argument("--montage", action="store_true", help="also render the figure")
    ap.add_argument(
        "--montage-only",
        action="store_true",
        help="render the figure from an existing CSV (no embedding)",
    )
    ap.add_argument(
        "--montage-samples",
        default=None,
        help="crops of the figure: 'subject/stem[:label],...' (index rel_paths)",
    )
    ap.add_argument("--montage-out", default=None, help="default: OUTPUT_ROOT/figures")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)
    args.operators = parse_list(args.operators)
    args.codecs = parse_list(args.codecs)
    args.models = parse_list(args.models)
    bad = [o for o in args.operators if o not in OPERATORS]
    bad += [c for c in args.codecs if c not in CODECS]
    if bad:
        raise SystemExit(f"unknown operator(s)/codec(s): {bad}")

    out = Path(args.out) if args.out else config.output_dir("accuracy") / OUT_NAME
    montage_out = (
        Path(args.montage_out)
        if args.montage_out
        else config.output_dir("figures") / "preproc_through_codec_kk.png"
    )
    if args.montage_only:
        build_montage(pd.read_csv(out), args, montage_out)
        return 0
    if args.cache_dir:
        cache = Path(args.cache_dir)
    elif config.LAYOUT == "legacy":
        raise SystemExit("FACE1KB_LAYOUT=legacy: pass --cache-dir")
    else:
        cache = config.work_dir(DATASET) / "preproc_through_codec"

    rel_paths = [str(p) for p in config.read_index(DATASET).rel_path]
    sample = select_subset(len(rel_paths), args.n, args.seed)
    if args.limit:
        sample = sample[: args.limit]
    args.tag = sample_tag(args.n, args.seed, args.limit)
    log.info(
        "N=%d (n=%d, seed=%d) operators=%s codecs=%s",
        len(sample),
        args.n,
        args.seed,
        args.operators,
        args.codecs,
    )
    if args.phase in ("encode", "all"):
        phase_encode(args, rel_paths, sample, cache)
    if args.phase in ("aggregate", "all"):
        df = phase_aggregate(args, rel_paths, sample, cache)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        log.info("wrote %s (%d rows)", out, len(df))
        if args.montage:
            build_montage(df, args, montage_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
