# SPDX-License-Identifier: MIT
r"""Decoded-crop montages of the paper (Figure 1 and Figures 8, 24-28).

Each montage shows how the codecs reconstruct the same aligned crops at a byte
budget, next to the lossless crop. Bitstreams are read from
``config.compressed_dir`` and decoded with :func:`face1kb.baselines.decode_file`
(decoded PNG caches are used when present); a missing file renders as an ``n/a``
tile. Tile captions give the codec, budget and resolution and, where the quality
CSVs have the cell, its dataset-median PSNR / SSIM; the abstract strip is captioned
with the metrics of the shown crop itself (PSNR, SSIM with an 11 px Gaussian window,
LPIPS-AlexNet).

========================  ==============================================================
file                      content
========================  ==============================================================
visual_codec_grid_cf/kk   3 samples x original + 12 codecs, 224 px / 1024 B (Figure 24)
visual_ours_cf/kk         3 samples x original, WebP, AVIF, JPEG-AI, Ours-FAST,
                          Ours-ACCURATE, 112 px / 1024 B (Figure 25)
visual_budget_sweep       one KK sample, WebP / Ours-ACCURATE x 1024 / 512 B, 112 px
                          (Figure 26)
visual_resolution_sweep   one KK sample, WebP 1024 B at 64 / 112 / 224 px (Figure 27)
budget_224_cf/kk          3 samples x WebP, Ours-ACCURATE, Ours-FAST at 1024 and
                          512 B, 224 px (Figure 28)
comparison_1kb            one Color FERET sample, all codecs, 112 px / 1024 B (Figure 8)
abstract_codec_strip      the abstract-page strip (Figure 1)
========================  ==============================================================

The samples of the paper figures are the defaults: Color FERET images by NIST file
name, AI-Solutions-KK images by their row id in ``index.csv``; ``--cf-samples`` /
``--kk-samples`` choose others (``<id or subject/stem>:<row label>``, comma
separated). The default row labels are neutral ("Sample 1" to "Sample 3"); the paper
captions its rows with image attributes, which are not shipped with the code. The
images show dataset faces: they are written to ``--fig-dir`` (default
``$FACE1KB_OUTPUT_ROOT/figures``) for local use and must not be committed or
redistributed. Decoding the face1kb, CompressAI and JPEG-AI tiles needs a CUDA
GPU.

Examples
--------
    python experiments/quality/visual_quality.py --from-results
    python experiments/quality/visual_quality.py --only comparison_1kb \
        --cf-samples 00054/00054_931230_fa_a:sample
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from face1kb import config
from face1kb.baselines import decode_file, extension
from face1kb.data.cli_utils import setup_logging
from face1kb.report import style

log = logging.getLogger("quality.visual")

#: Default Color FERET samples: (subject/stem, row label).
CF_SAMPLES = (
    ("00045/00045_931230_fa_a", "Sample 1"),
    ("00054/00054_931230_fa_a", "Sample 2"),
    ("00066/00066_931230_fa_a", "Sample 3"),
)
#: Default AI-Solutions-KK samples: (index.csv row id, row label).
KK_SAMPLES = (
    (14847, "Sample 1"),
    (15065, "Sample 2"),
    (9867, "Sample 3"),
)
#: Sample of the abstract strip (a crop of the 300-image CompressAI subset).
ABSTRACT_SAMPLE = ("00002/00002_940422_fa", "Color FERET frontal")
#: Tile labels.
LABELS = {
    "original": "Original",
    "jpeg": "JPEG",
    "jpeg2000": "JPEG 2000",
    "webp": "WebP",
    "jpeg_xl": "JPEG-XL",
    "avif": "AVIF",
    "heif": "HEIF",
    "jpeg_fzt": "JPEG-FzT",
    "jpeg_ai": "JPEG-AI",
    "neural_bmshj2018": "bmshj2018",
    "neural_mbt2018_mean": "mbt2018",
    "ours_fast": "Ours-FAST",
    "ours_accurate": "Ours-ACC",
}
#: Full roster of the codec grids and the all-codec montage.
ROSTER = (
    "original",
    "jpeg",
    "jpeg2000",
    "webp",
    "jpeg_xl",
    "avif",
    "heif",
    "jpeg_fzt",
    "jpeg_ai",
    "neural_bmshj2018",
    "neural_mbt2018_mean",
    "ours_fast",
    "ours_accurate",
)
SHOWCASE = ("original", "webp", "avif", "jpeg_ai", "ours_fast", "ours_accurate")
BUDGET_224 = (
    ("webp", 224, 1024),
    ("webp", 224, 512),
    ("ours_accurate", 224, 1024),
    ("ours_accurate", 224, 512),
    ("ours_fast", 224, 1024),
    ("ours_fast", 224, 512),
)
FIGURES = (
    "visual_codec_grid_cf",
    "visual_codec_grid_kk",
    "visual_ours_cf",
    "visual_ours_kk",
    "budget_224_cf",
    "budget_224_kk",
    "visual_budget_sweep",
    "visual_resolution_sweep",
    "comparison_1kb",
    "abstract_codec_strip",
)
DPI = style.FIGURE_DPI["visual_grid"]


# ---------------------------------------------------------------- inputs
def resolve_samples(dataset: str, specs) -> list[tuple[str, str]]:
    """``[(rel_stem, label)]`` from ``(index id | "subject/stem", label)`` specs."""
    index = None
    out = []
    for key, label in specs:
        if isinstance(key, int) or str(key).isdigit():
            if index is None:
                index = config.read_index(dataset)
            rel = index.loc[index["id"] == int(key), "rel_path"]
            if rel.empty:
                raise SystemExit(f"{dataset}: no index row with id {key}")
            key = str(Path(rel.iloc[0]).with_suffix(""))
        out.append((str(key), label))
    return out


def parse_samples(value: str) -> list[tuple[str, str]]:
    """Parse ``key:label,key:label`` (the label defaults to the key)."""
    out = []
    for item in value.split(","):
        if item.strip():
            key, _, label = item.strip().partition(":")
            out.append((key, label or key))
    return out


class Tiles:
    """Decoded tiles of one dataset (uint8 ``res x res x 3`` or ``None``)."""

    def __init__(self, dataset: str, device: str):
        self.dataset = dataset
        self.device = device

    def __call__(self, codec: str, res: int, budget: int, rel: str):
        from PIL import Image  # noqa: PLC0415

        if codec == "original":
            p = config.aligned_dir(self.dataset, res) / f"{rel}.png"
            if not p.exists():
                return None
            with Image.open(p) as im:
                return np.asarray(im.convert("RGB"), dtype=np.uint8)
        p = config.compressed_dir(self.dataset, res, budget, codec) / (
            rel + extension(codec)
        )
        if not p.exists():
            return None
        try:
            dec = decode_file(p, res=res, device=self.device, cache=True)
        except Exception as exc:  # noqa: BLE001 - a broken tile renders as n/a
            log.warning("decode failed %s %d %d %s: %s", codec, res, budget, rel, exc)
            return None
        return dec if dec is not None and dec.shape[:2] == (res, res) else None


def load_quality(root: Path):
    """Concatenated ``quality_<dataset>.csv`` frames (empty if none exist)."""
    import pandas as pd  # noqa: PLC0415

    parts = [
        pd.read_csv(root / f"quality_{ds}.csv")
        for ds in config.DATASETS
        if (root / f"quality_{ds}.csv").exists()
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def median_metrics(q, dataset: str, codec: str, res: int, budget: int):
    """Dataset-median ``(psnr, ssim)`` of a cell, or ``(None, None)``."""
    if q.empty or codec == "original" or codec not in LABELS:
        return None, None
    row = q[
        (q.dataset == dataset)
        & (q.codec == codec)
        & (q.res == res)
        & (q.budget == budget)
    ]
    if row.empty:
        return None, None
    return float(row.psnr.iloc[0]), float(row.ssim.iloc[0])


def caption(codec: str, res: int, budget: int, q, dataset: str) -> str:
    name = LABELS.get(codec, codec)
    if codec == "original":
        return f"{name}\n{res}px (lossless)"
    psnr, ssim = median_metrics(q, dataset, codec, res, budget)
    head = f"{name}\n{budget}B @ {res}px"
    return f"{head}\n{psnr:.1f}dB / {ssim:.3f}" if psnr is not None else head


class CropMetrics:
    """PSNR, SSIM (``pytorch_msssim``, 11 px Gaussian) and LPIPS-AlexNet of one crop."""

    def __init__(self):
        self._lpips = None

    def __call__(self, ref: np.ndarray, rec: np.ndarray) -> tuple[float, float, float]:
        import torch  # noqa: PLC0415
        from pytorch_msssim import ssim as ssim_fn  # noqa: PLC0415

        if self._lpips is None:
            import lpips  # noqa: PLC0415

            self._lpips = lpips.LPIPS(net="alex", verbose=False).eval()
        a = torch.from_numpy(np.array(ref)).permute(2, 0, 1)[None].float()
        b = torch.from_numpy(np.array(rec)).permute(2, 0, 1)[None].float()
        a, b = a / 255, b / 255
        mse = torch.mean((a - b) ** 2).item()
        psnr = 99.0 if mse < 1e-9 else float(-10 * np.log10(mse))
        s = float(ssim_fn(a, b, data_range=1.0).item())
        with torch.no_grad():
            lp = float(self._lpips(a * 2 - 1, b * 2 - 1).item())
        return psnr, s, lp


# ---------------------------------------------------------------- montages
def _na(ax) -> None:
    ax.imshow(np.full((8, 8, 3), 235, dtype=np.uint8))
    ax.text(
        0.5,
        0.5,
        "n/a",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=8,
        color="#888",
    )


def montage(rows, cols, dataset, tiles, q, title, out: Path) -> int:
    """Rows of samples x columns of ``(codec, res, budget)``; returns n/a count."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    nr, nc = len(rows), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(1.6 * nc, 1.6 * nr + 0.5), squeeze=False)
    missing = 0
    for ri, (rel, rlabel) in enumerate(rows):
        for ci, (codec, res, budget) in enumerate(cols):
            ax = axes[ri][ci]
            ax.set_xticks([])
            ax.set_yticks([])
            tile = tiles(codec, res, budget, rel)
            if tile is None:
                _na(ax)
                missing += 1
            else:
                ax.imshow(tile)
            if ri == 0:
                ax.set_title(
                    caption(codec, res, budget, q, dataset),
                    fontsize=7,
                    color="#111",
                    pad=4,
                )
            if ci == 0:
                ax.set_ylabel(rlabel, fontsize=7, color="#111")
    fig.suptitle(title, fontsize=11, color="#111", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return missing


def budget_sweep(sample, tiles, q, out: Path) -> int:
    """2 x 2: WebP / Ours-ACCURATE x 1024 / 512 B at 112 px (one KK sample)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rel, _label = sample
    rowspec = (("webp", "WebP"), ("ours_accurate", "Ours-ACC"))
    budgets = (1024, 512)
    fig, axes = plt.subplots(2, 2, figsize=(1.6 * 2, 1.6 * 2 + 0.5), squeeze=False)
    missing = 0
    for ri, (codec, rlabel) in enumerate(rowspec):
        for ci, budget in enumerate(budgets):
            ax = axes[ri][ci]
            ax.set_xticks([])
            ax.set_yticks([])
            tile = tiles(codec, 112, budget, rel)
            if tile is None:
                _na(ax)
                missing += 1
            else:
                ax.imshow(tile)
            if ri == 0:
                ax.set_title(f"{budget}B", fontsize=8, color="#111", pad=4)
            if ci == 0:
                ax.set_ylabel(rlabel, fontsize=8, color="#111")
            psnr, ssim = median_metrics(q, "kk", codec, 112, budget)
            if psnr is not None:
                ax.set_xlabel(f"{psnr:.1f}dB/{ssim:.3f}", fontsize=6, color="#444")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return missing


def all_codecs(
    sample,
    dataset,
    tiles,
    q,
    out: Path,
    *,
    ncol: int = 6,
    suptitle: str | None = "Per-codec reconstruction at 112px / 1024B "
    "(one Color FERET sample)",
    tile_in: tuple[float, float] = (1.95, 2.25),
    crop_metrics: CropMetrics | None = None,
) -> int:
    """One sample, every codec at 112 px / 1024 B, ``ncol`` tiles per row.

    With ``crop_metrics`` the tiles are captioned with the metrics of this crop
    instead of the cell medians. Codecs without a tile are left out.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rel, _label = sample
    shown = []
    for c in ROSTER:
        t = tiles(c, 112, 1024, rel)
        if t is not None:
            shown.append((c, t))
    if not shown:
        log.warning("%s: no tile of %s", out.name, rel)
        return 1
    nrow = int(np.ceil(len(shown) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(tile_in[0] * ncol, tile_in[1] * nrow + (0.5 if suptitle else 0.0)),
        squeeze=False,
        constrained_layout=True,
    )
    ref = next((t for c, t in shown if c == "original"), None)
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= len(shown):
            ax.axis("off")
            continue
        c, t = shown[i]
        ax.imshow(t)
        if crop_metrics is not None and c == "original":
            cap = f"{LABELS.get(c, c)}\n112px (lossless)"
        elif crop_metrics is not None and ref is not None:
            psnr, ssim, lp = crop_metrics(ref, t)
            cap = (
                f"{LABELS.get(c, c)}\n1024B @ 112px\n"
                f"{psnr:.1f}dB / {ssim:.3f}\nLPIPS {lp:.3f}"
            )
        else:
            cap = caption(c, 112, 1024, q, dataset)
        ax.set_title(cap, fontsize=7.5, pad=3)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return 0


def build(name: str, fig_dir: Path, samples: dict, tiles: dict, q) -> int:
    """Render one figure; returns its n/a tile count."""
    out = fig_dir / f"{name}.png"
    grid = lambda codecs, res: [(c, res, 1024) for c in codecs]  # noqa: E731
    ds_of = {"cf": "colorferet", "kk": "kk"}
    names = {"colorferet": "Color FERET", "kk": "AI-Solutions-KK"}
    for key, ds in ds_of.items():
        if name == f"visual_codec_grid_{key}":
            return montage(
                samples[ds],
                grid(ROSTER, 224),
                ds,
                tiles[ds],
                q,
                f"Codec comparison — {names[ds]}, 224px @ 1024B",
                out,
            )
        if name == f"visual_ours_{key}":
            return montage(
                samples[ds],
                grid(SHOWCASE, 112),
                ds,
                tiles[ds],
                q,
                f"Learned codec head-to-head — {names[ds]}, 112px @ 1024B",
                out,
            )
        if name == f"budget_224_{key}":
            return montage(
                samples[ds],
                list(BUDGET_224),
                ds,
                tiles[ds],
                q,
                f"Budget degradation at 224px — {names[ds]} (1024B vs 512B)",
                out,
            )
    if name == "visual_budget_sweep":
        return budget_sweep(samples["kk"][1], tiles["kk"], q, out)
    if name == "visual_resolution_sweep":
        return montage(
            [samples["kk"][1]],
            [("webp", 64, 1024), ("webp", 112, 1024), ("webp", 224, 1024)],
            "kk",
            tiles["kk"],
            q,
            "Resolution sweep — KK mid-skin sample, WebP @ 1024B",
            out,
        )
    if name == "comparison_1kb":
        return all_codecs(
            samples["colorferet"][1], "colorferet", tiles["colorferet"], q, out
        )
    if name == "abstract_codec_strip":
        return all_codecs(
            samples["abstract"][0],
            "colorferet",
            tiles["colorferet"],
            q,
            out,
            ncol=7,
            suptitle=None,
            tile_in=(1.7, 2.15),
            crop_metrics=CropMetrics(),
        )
    raise ValueError(f"unknown figure {name!r}; choose from {FIGURES}")


def main(argv: list[str] | None = None) -> int:
    """Render the requested montages."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--quality-root",
        type=Path,
        default=None,
        help="folder with quality_<dataset>.csv for the captions "
        "(default: $FACE1KB_OUTPUT_ROOT/quality)",
    )
    src.add_argument(
        "--from-results",
        action="store_true",
        help="caption with the shipped paper results (<repo>/results/quality)",
    )
    ap.add_argument("--only", default="", help=f"comma list of {FIGURES}")
    ap.add_argument("--cf-samples", default="", help="<subject/stem>:<label>,...")
    ap.add_argument("--kk-samples", default="", help="<index id>:<label>,...")
    ap.add_argument("--abstract-sample", default="", help="<subject/stem>[:label]")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fig-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    setup_logging()
    if not args.device.startswith("cuda"):
        ap.error("the face1kb, CompressAI and JPEG-AI tiles must be decoded on CUDA")
    style.use_agg()

    qroot = (
        config.results_dir("quality")
        if args.from_results
        else args.quality_root or config.output_dir("quality")
    )
    q = load_quality(qroot)
    names = [n for n in args.only.split(",") if n] or list(FIGURES)
    need = {n: n for n in names}
    samples = {
        "colorferet": resolve_samples(
            "colorferet", parse_samples(args.cf_samples) or CF_SAMPLES
        ),
        "abstract": resolve_samples(
            "colorferet", parse_samples(args.abstract_sample) or [ABSTRACT_SAMPLE]
        ),
    }
    if any(
        n.endswith("_kk") or n in ("visual_budget_sweep", "visual_resolution_sweep")
        for n in need
    ):
        samples["kk"] = resolve_samples(
            "kk", parse_samples(args.kk_samples) or KK_SAMPLES
        )
    tiles = {ds: Tiles(ds, args.device) for ds in config.DATASETS}
    fig_dir = args.fig_dir or config.output_dir("figures")
    missing = 0
    for name in names:
        n = build(name, fig_dir, samples, tiles, q)
        missing += n
        log.info("wrote %s (%d n/a tiles)", fig_dir / f"{name}.png", n)
    log.info("done; the montages show dataset faces: keep them local")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
