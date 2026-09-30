# SPDX-License-Identifier: MIT
r"""JPEG-AI decoder operation-point figure (fig:jpegai-tradeoff; CPU).

Reads ``<input-root>/codec_comparison/jpegai_decoders.csv`` (the 24-crop 224 px
timing study of the three JPEG-AI operation points SOP / BOP / HOP, also the source
of tab:jpegai-speed) and draws two panels:

* (a) median decode wall-clock per operation point, CPU (one thread) vs GPU, as
  grouped bars labelled in ms;
* (b) PSNR against the GPU decode time as a scatter (the three points are not
  ordered along either axis, so they are not joined).

Decode times are rounded to whole ms and PSNR to 0.01 dB, the precision of the
table. Writes ``jpegai_decoder_tradeoff.png`` and ``.pdf`` (150 dpi) into
``--fig-dir`` (default ``$FACE1KB_OUTPUT_ROOT/figures``).

Examples
--------
    python experiments/codec_comparison/jpegai_tradeoff.py --from-results
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from face1kb import config
from face1kb.data.cli_utils import setup_logging
from face1kb.report import style

log = logging.getLogger("codec_comparison.jpegai_tradeoff")

#: Operation points in plotting order.
OP_POINTS = ("SOP", "BOP", "HOP")
CPU_COLOR = "#4C72B0"
GPU_COLOR = "#DD8452"
POINT_COLOR = "#55A868"


def load_points(path: Path) -> dict[str, tuple[int, int, float]]:
    """``{op: (cpu decode ms, gpu decode ms, psnr dB)}`` from jpegai_decoders.csv."""
    rows: dict[tuple[str, str], dict] = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            rows[(r["op_point"], r["device"])] = r
    out = {}
    for op in OP_POINTS:
        cpu, gpu = rows[(op, "cpu")], rows[(op, "gpu")]
        out[op] = (
            round(float(cpu["decode_ms_median"])),
            round(float(gpu["decode_ms_median"])),
            round(float(gpu["psnr"]), 2),
        )
    return out


def draw(points: dict, out_stem: Path, gpu_label: str, n: int, res: int) -> list[Path]:
    """Draw both panels and save ``<out_stem>.png`` and ``.pdf``."""
    style.use_agg()
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    names = list(points)
    xs = range(len(names))
    cpu = [points[k][0] for k in names]
    gpu = [points[k][1] for k in names]
    b1 = ax.bar(
        [x - 0.2 for x in xs], cpu, 0.4, label="CPU (1 thread)", color=CPU_COLOR
    )
    b2 = ax.bar([x + 0.2 for x in xs], gpu, 0.4, label=gpu_label, color=GPU_COLOR)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%d", fontsize=9, padding=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names)
    ax.set_ylabel("Decode time (ms), median")
    ax.set_title("(a) Decode wall-clock per op. point")
    ax.set_ylim(0, max(cpu + gpu) * 1.18)
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    for name, (_cpu, g, psnr) in points.items():
        ax.scatter(g, psnr, s=70, color=POINT_COLOR, zorder=3)
        ax.annotate(
            name, (g, psnr), textcoords="offset points", xytext=(8, -3), fontsize=10
        )
    ax.set_xlabel("GPU decode time (ms)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("(b) Quality vs. GPU decode cost")
    ax.grid(alpha=0.3)
    ax.margins(x=0.18, y=0.22)

    fig.suptitle(
        f"JPEG-AI operation points, sub-1 kB face crops ({res}x{res}, n={n})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=150)
        written.append(p)
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    """Render the JPEG-AI trade-off figure."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="folder holding codec_comparison/ (default: $FACE1KB_OUTPUT_ROOT)",
    )
    src.add_argument("--from-results", action="store_true", help="read <repo>/results")
    ap.add_argument(
        "--gpu-label",
        default="GPU (RTX 6000)",
        help="legend entry of the GPU bars (the published timings: Quadro RTX 6000)",
    )
    ap.add_argument("--fig-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    setup_logging()

    base = config.RESULTS_ROOT if args.from_results else args.input_root
    csv_path = (base or config.OUTPUT_ROOT) / "codec_comparison" / "jpegai_decoders.csv"
    points = load_points(csv_path)
    with csv_path.open() as fh:
        n = int(next(csv.DictReader(fh))["n"])
    fig_dir = args.fig_dir or config.output_dir("figures")
    for p in draw(points, fig_dir / "jpegai_decoder_tradeoff", args.gpu_label, n, 224):
        log.info("wrote %s", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
