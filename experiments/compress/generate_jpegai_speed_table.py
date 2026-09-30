# SPDX-License-Identifier: MIT
r"""Table of the three JPEG-AI operation points (``tab:jpegai-speed``).

Renders ``<input-root>/codec_comparison/jpegai_decoders.csv`` -- per operation point
(SOP, BOP, HOP) and device (cpu, gpu): median encode / decode ms, median bytes and
the reconstruction quality (PSNR, SSIM, LPIPS, ArcFace identity cosine) of 24 Color
FERET crops at 224 px -- as the paper table: CPU and GPU decode, GPU encode, PSNR,
SSIM, id-cos and bytes. Shading per column: the time columns rank lower-is-better,
the quality columns higher-is-better (on the unrounded values); the byte column is
not ranked.

The CSV is a shipped aggregate of the paper; this repository has no script that
re-measures it (``benchmark_jpegai_speed.py`` times the HOP profile per resolution,
``measure_jpegai_kk.py`` measures the three operation points on AI-Solutions-KK).

Input: ``--input-root`` (default: the shipped ``results/``). Output:
``OUTPUT_ROOT/tables/jpegai_speed.tex``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from face1kb import config
from face1kb.report import latex

log = logging.getLogger("jpegai_speed_table")

OP_POINTS = ("SOP", "BOP", "HOP")
#: Synthesis-transform depth of each operation point.
SYNTH = {"SOP": "1", "BOP": "2", "HOP": "3"}
#: (device, CSV column, format, lower_is_better or None for unranked).
COLUMNS = (
    ("cpu", "decode_ms_median", "{:.0f}", True),
    ("gpu", "decode_ms_median", "{:.0f}", True),
    ("gpu", "encode_ms_median", "{:.0f}", True),
    ("gpu", "psnr", "{:.2f}", False),
    ("gpu", "ssim", "{:.3f}", False),
    ("gpu", "id_cos", "{:.3f}", False),
    ("gpu", "bytes_median", "{:.0f}", None),
)
STEM = "jpegai_speed"
HEADER = (
    "\\begin{tabular}{l c c c c c c c c}",
    "\\toprule",
    "Op.\\ point & Synth. & \\multicolumn{2}{c}{Decode (ms)} & Encode (ms) & PSNR & "
    "SSIM & id-cos & Bytes \\\\",
    "\\cmidrule(lr){3-4}",
    " & transf. & CPU & GPU & GPU & (dB) & & (ArcFace) & \\\\",
    "\\midrule",
)


def render(df: pd.DataFrame) -> str:
    """Colour-only LaTeX table of the operation points."""
    vals = {}
    for op in OP_POINTS:
        row = []
        for dev, col, _fmt, _dir in COLUMNS:
            r = df[(df.op_point == op) & (df.device == dev)]
            row.append(float(r.iloc[0][col]) if not r.empty else None)
        vals[op] = row
    lines = list(HEADER)
    for op in OP_POINTS:
        cells = [op, SYNTH[op]]
        for j, (_dev, _col, fmt, lower) in enumerate(COLUMNS):
            v = vals[op][j]
            if v is None:
                cells.append("--")
                continue
            txt = fmt.format(v)
            col = [vals[o][j] for o in OP_POINTS if vals[o][j] is not None]
            if lower is not None and len(set(col)) > 1:
                best = min(col) if lower else max(col)
                worst = max(col) if lower else min(col)
                if v == best:
                    txt = latex.shade_best(txt, emphasis=False)
                elif v == worst:
                    txt = latex.shade_worst(txt, emphasis=False)
            cells.append(txt)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input-root", default=None, help="default: results/")
    ap.add_argument("--output-root", default=None, help="override FACE1KB_OUTPUT_ROOT")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    in_root = Path(args.input_root or config.RESULTS_ROOT)
    df = pd.read_csv(in_root / "codec_comparison" / "jpegai_decoders.csv")
    out_root = Path(args.output_root) if args.output_root else config.OUTPUT_ROOT
    out = out_root / "tables" / f"{STEM}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex.finalize_table(STEM, render(df)))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
