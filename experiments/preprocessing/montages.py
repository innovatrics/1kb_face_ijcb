# SPDX-License-Identifier: MIT
"""Alignment montages of the crop-tightness ablation (Color FERET, CPU only).

Writes two figures under ``OUTPUT_ROOT/figures``:

``alignment_check.png``
    The fixed ArcFace 5-point template (scaled to each crop size) drawn over
    aligned crops: six subjects, a frontal / half-left / half-right pose, and one
    crop at 64, 96, 112, 168 and 224 px. The dots mark where alignment placed the
    eye, nose and mouth-corner reference points; they are the template, not
    landmarks detected on the crop.
``alignment_variants.png``
    The crop-tightness presets standard / tight / mid / fill of six subjects
    (``aligned_112``, ``aligned_112_{tight,mid,fill}``).

Crops are picked deterministically: in index order, the first colour crop (mean
absolute channel difference above 6 grey levels) of each new subject, restricted to
the requested pose. The figures show dataset faces; keep them local.

Examples
--------
    python experiments/preprocessing/montages.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from face1kb import config
from face1kb.data.alignment import ARCFACE_DST_112
from face1kb.data.cli_utils import setup_logging

log = logging.getLogger("alignment_montages")

DATASET = "colorferet"
PRESETS: tuple[tuple[str, str], ...] = (
    ("", "standard (0.31)"),
    ("_tight", "tight (0.40)"),
    ("_mid", "mid (0.46)"),
    ("_fill", "fill (0.52)"),
)


def _rgb(path: Path) -> np.ndarray | None:
    from PIL import Image  # noqa: PLC0415

    if not path.exists():
        return None
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), np.uint8)


def is_colour(arr: np.ndarray) -> bool:
    """Return True for a colour crop (mean |R-G| + |G-B| above 6 grey levels)."""
    a = arr.astype(np.float32)
    return (
        np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()
    ) > 6.0


def pick(index, n: int, poses=("frontal image",)) -> list[tuple[str, str]]:
    """First colour crop of each new subject in index order: ``[(rel_path, pose)]``."""
    idx = index[index.pose.isin(poses)] if poses else index
    std = config.aligned_dir(DATASET, 112)
    out, seen = [], set()
    for r in idx.itertuples():
        subj = str(Path(r.rel_path).parent)
        if subj in seen:
            continue
        arr = _rgb(std / r.rel_path)
        if arr is None or not is_colour(arr):
            continue
        out.append((r.rel_path, r.pose))
        seen.add(subj)
        if len(out) >= n:
            break
    return out


def _overlay(ax, arr, scale=1.0):
    ax.imshow(arr)
    pts = np.asarray(ARCFACE_DST_112) * scale
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=14,
        c="#39FF14",
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )


def _off(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def alignment_check(index, out: Path) -> None:
    """Template overlay across subjects, poses and resolutions."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    subjects = pick(index, 6)
    poses = (
        pick(index, 1, ("frontal image",))
        + pick(index, 1, ("half left",))
        + pick(index, 1, ("half right",))
    )
    res_rel = subjects[0][0]
    std = config.aligned_dir(DATASET, 112)
    fig, ax = plt.subplots(3, 6, figsize=(11, 6), squeeze=False)
    for c, (rel, _) in enumerate(subjects):
        arr = _rgb(std / rel)
        if arr is not None:
            _overlay(ax[0][c], arr)
        _off(ax[0][c])
    ax[0][0].set_ylabel("template overlay\n(subjects)", fontsize=8)
    for c in range(6):
        _off(ax[1][c])
        ax[1][c].axis("off")
    for c, (rel, pose) in enumerate(poses):
        arr = _rgb(std / rel)
        if arr is not None:
            ax[1][c].axis("on")
            _overlay(ax[1][c], arr)
            ax[1][c].set_title(str(pose), fontsize=8)
        _off(ax[1][c])
    ax[1][0].set_ylabel("poses", fontsize=8)
    for c in range(6):
        _off(ax[2][c])
        ax[2][c].axis("off")
    for c, res in enumerate(config.RESOLUTIONS):
        arr = _rgb(config.aligned_dir(DATASET, res) / res_rel)
        if arr is not None:
            ax[2][c].axis("on")
            _overlay(ax[2][c], arr, scale=res / 112.0)
            ax[2][c].set_title(f"{res}px", fontsize=8)
        _off(ax[2][c])
    ax[2][0].set_ylabel("resolutions", fontsize=8)
    fig.suptitle(
        "Landmark-to-ArcFace alignment check (Color FERET, coloured); "
        "green = fixed 5-point template",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def alignment_variants(index, out: Path) -> None:
    """Crop-tightness presets of six subjects."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    subjects = pick(index, 6)
    fig, ax = plt.subplots(
        len(PRESETS),
        len(subjects),
        figsize=(1.5 * len(subjects), 1.6 * len(PRESETS)),
        squeeze=False,
    )
    for r, (suffix, label) in enumerate(PRESETS):
        d = config.aligned_dir(DATASET, 112, suffix)
        for c, (rel, _) in enumerate(subjects):
            arr = _rgb(d / rel)
            if arr is not None:
                ax[r][c].imshow(arr)
            _off(ax[r][c])
        ax[r][0].set_ylabel(label, fontsize=8)
    fig.suptitle(
        "Crop-tightness presets at 112px (coloured Color FERET subjects)", fontsize=9
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None, help="default: OUTPUT_ROOT/figures")
    args = ap.parse_args(argv)
    setup_logging()
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    out_dir = Path(args.out_dir) if args.out_dir else config.output_dir("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = config.read_index(DATASET)
    if "pose" not in index.columns:
        raise SystemExit("the Color FERET index has no pose column")
    alignment_check(index, out_dir / "alignment_check.png")
    alignment_variants(index, out_dir / "alignment_variants.png")
    log.info(
        "wrote %s and %s (dataset faces: keep them local)",
        *(out_dir / "alignment_check.png", out_dir / "alignment_variants.png"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
