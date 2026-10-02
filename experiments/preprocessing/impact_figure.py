# SPDX-License-Identifier: MIT
"""WebP headroom of the preprocessing variants on clean Color FERET crops.

For the first ``--n`` crops of the Color FERET index (the paper: 4, all of subject
00001) and each preprocessing variant (``std``, ``A1``-``A4``, ``B1``, ``B2``,
``C1``, ``C2``; crops in ``aligned_112<suffix>``), the crop is WebP-encoded with a
descending quality scan (quality 94, 92, ..., 2, ``method=6``) that stops at the first
file within the budget (the quality-2 file when none fits). The decoded crop is scored
against the variant's own uncompressed crop with PSNR, SSIM and LPIPS
(:func:`face1kb.eval.quality.make_metrics`).

Outputs:

* ``OUTPUT_ROOT/ablation/preprocessing_impact.csv`` -- mean WebP quality, bytes,
  PSNR, SSIM and LPIPS per variant (also printed); with the default ``--n`` these are
  crops of one subject, so keep the file local;
* ``OUTPUT_ROOT/figures/preprocessing_impact.png`` -- the first crop of every variant
  (top) and its WebP reconstruction annotated with bytes and WebP quality (bottom).
  It shows a dataset face; keep it local.

Examples
--------
    python experiments/preprocessing/impact_figure.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from face1kb import config
from face1kb.baselines import classical
from face1kb.data.cli_utils import setup_logging

log = logging.getLogger("preprocessing_impact")

DATASET = "colorferet"
VARIANTS: tuple[str, ...] = ("std", "A1", "A2", "A3", "A4", "B1", "B2", "C1", "C2")
LABEL = {
    "std": "std (orig)",
    "A1": "A1 edge-mild",
    "A2": "A2 bilat-strong",
    "A3": "A3 chroma",
    "A4": "A4 NLM",
    "B1": "B1 bg BiRefNet",
    "B2": "B2 bg MediaPipe",
    "C1": "C1 bg+denoise",
    "C2": "C2 foveated",
}
#: WebP qualities of the descending scan.
QUALITIES = tuple(range(94, 0, -2))


def variant_dir(v: str) -> Path:
    """Crop folder of preprocessing variant ``v``."""
    return config.aligned_dir(DATASET, 112, "" if v == "std" else f"_{v}")


def webp_descending(img, budget: int = 1024) -> tuple[np.ndarray, int, int]:
    """First WebP quality (from 94 down) within ``budget``: ``(decoded, bytes, q)``."""
    last = None
    for q in QUALITIES:
        data = classical.encode_setting(img, "webp", q)
        last = (data, len(data), q)
        if len(data) <= budget:
            break
    data, n, q = last
    return classical.decode(data), n, q


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=4, help="first N crops of the index")
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--csv", default=None, help="default: OUTPUT_ROOT/ablation/...")
    ap.add_argument("--figure", default=None, help="default: OUTPUT_ROOT/figures/...")
    ap.add_argument(
        "--figure-crop",
        default=None,
        help="rel_path of the crop shown in the figure (default: the first crop)",
    )
    args = ap.parse_args(argv)
    setup_logging()
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from face1kb.eval.quality import make_metrics  # noqa: PLC0415

    rels = list(config.read_index(DATASET).rel_path[: args.n])
    all_metrics = make_metrics(args.device)
    metrics = {k: all_metrics[k] for k in ("psnr", "ssim", "lpips")}

    def t(arr):
        x = torch.from_numpy(np.asarray(arr, np.float32) / 255).permute(2, 0, 1)
        return x.unsqueeze(0).to(args.device)

    def load(v, rel):
        with Image.open(variant_dir(v) / rel) as im:
            return im.convert("RGB")

    rows = []
    for v in VARIANTS:
        for rel in rels:
            src = load(v, rel)
            dec, nbytes, q = webp_descending(src, args.budget)
            rec = {"variant": v, "bytes": nbytes, "webp_q": q}
            with torch.no_grad():
                for name, fn in metrics.items():
                    rec[name] = fn(t(dec), t(src)).item()
            rows.append(rec)
    g = (
        pd.DataFrame(rows)
        .groupby("variant")[["webp_q", "bytes", "psnr", "ssim", "lpips"]]
        .mean()
        .reindex(list(VARIANTS))
    )
    csv = (
        Path(args.csv)
        if args.csv
        else config.output_dir("ablation") / ("preprocessing_impact.csv")
    )
    csv.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(csv)
    log.info("per-variant means (n=%d):\n%s", len(rels), g.round(3).to_string())

    fig, ax = plt.subplots(2, len(VARIANTS), figsize=(17, 4.2))
    shown = args.figure_crop or rels[0]
    for c, v in enumerate(VARIANTS):
        src = load(v, shown)
        dec, nbytes, q = webp_descending(src, args.budget)
        ax[0, c].imshow(src)
        ax[0, c].set_title(LABEL[v], fontsize=8)
        ax[0, c].axis("off")
        ax[1, c].imshow(dec)
        ax[1, c].set_title(f"{nbytes}B q{q}", fontsize=8)
        ax[1, c].axis("off")
    ax[0, 0].set_ylabel("preprocessed")
    ax[1, 0].set_ylabel("WebP@1kB")
    plt.tight_layout()
    fig_path = (
        Path(args.figure)
        if args.figure
        else config.output_dir("figures") / "preprocessing_impact.png"
    )
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=95, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s and %s (dataset face: keep the figure local)", csv, fig_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
