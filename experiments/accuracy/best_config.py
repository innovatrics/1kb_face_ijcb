# SPDX-License-Identifier: MIT
r"""Best configuration per codec and budget: the resolution with the lowest FNMR.

Reads ``accuracy/metrics.csv`` and writes ``best_config.tex`` (Section 5.4,
``tab:best-config``). For every codec and budget (1024 / 512 B) on Color FERET, the
FNMR at FMR = 1e-4 of the clean cells is averaged over ArcFace and LVFace-L at each
resolution (64-224 px, only resolutions scored by both matchers), and the
resolution with the lowest mean is reported as ``<FNMR %> @<res>``. The lowest and
highest value of each budget column are shaded (bold / underlined) with the rules
of :func:`face1kb.report.latex.shade_table`.

The published table was assembled by hand from ``metrics.csv``; this generator
reproduces it except for one cell: JPEG-AI at 1024 B is ``0.71 @224`` in the data,
the paper prints ``0.73 @224``.

Examples
--------
    python experiments/accuracy/best_config.py --from-results
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from face1kb import config
from face1kb.report.latex import shade_table

log = logging.getLogger("best_config")

HEADER = (
    "% best-configuration (optimal resolution per codec/budget), Color~FERET, "
    "ArcFace+LVFace-L mean"
)
DATASET = "colorferet"
MODELS = ("arcface_antelopev2", "lvface_l")
BUDGETS = (1024, 512)
CODECS = [
    ("jpeg_ai", "JPEG-AI"),
    ("ours_accurate", "Ours-ACCURATE"),
    ("ours_fast", "Ours-FAST"),
    ("webp", "WebP"),
    ("avif", "AVIF"),
    ("jpeg_xl", "JPEG~XL"),
    ("heif", "HEIF"),
    ("jpeg_fzt", "JPEG-FzT"),
    ("jpeg", "JPEG"),
    ("jpeg2000", "JPEG~2000"),
]


def best_resolution(df: pd.DataFrame, codec: str, budget: int):
    """``(mean FNMR@1e-4, res)`` of the best resolution, or None."""
    d = df[
        (df.dataset == DATASET)
        & (df.kind == "compressed")
        & (df.suffix.fillna("") == "")
        & df.model.isin(MODELS)
        & (df.codec == codec)
        & (df.budget == budget)
        & df["fnmr_0.0001"].notna()
    ]
    g = d.groupby("res")["fnmr_0.0001"].agg(["mean", "count"])
    g = g[g["count"] == len(MODELS)]
    if g.empty:
        return None
    res = g["mean"].idxmin()
    return float(g["mean"][res]), int(res)


def table(df: pd.DataFrame) -> str:
    """Shaded LaTeX tabular."""
    lines = [
        HEADER,
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Codec & " + " & ".join(f"${b}$\\,B" for b in BUDGETS) + " \\\\",
        " & " + " & ".join("FNMR@$10^{-4}$ \\%\\,@res" for _ in BUDGETS) + " \\\\",
        "\\midrule",
    ]
    for codec, label in CODECS:
        cells = []
        for b in BUDGETS:
            got = best_resolution(df, codec, b)
            cells.append("--" if got is None else f"{100 * got[0]:.2f} @{got[1]}")
        lines.append(f"{label:<14}& " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return shade_table("\n".join(lines), lower_better=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the best-configuration table."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--input-root", default=None, help="default: outputs/")
    src.add_argument(
        "--from-results", action="store_true", help="read the shipped results/"
    )
    ap.add_argument("--metrics", default=None, help="default: accuracy/metrics.csv")
    ap.add_argument("--out-dir", default=None, help="default: outputs/tables")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = (
        config.RESULTS_ROOT
        if args.from_results
        else Path(args.input_root or config.OUTPUT_ROOT)
    )
    df = pd.read_csv(Path(args.metrics or root / "accuracy" / "metrics.csv"))
    out_dir = Path(args.out_dir) if args.out_dir else config.output_dir("tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "best_config.tex"
    out.write_text(table(df))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
