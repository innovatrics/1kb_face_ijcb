# SPDX-License-Identifier: MIT
"""Worst-case identity tail: p5 and median per-image identity cosine (Section 7.8).

For JPEG-AI, Ours-ACCURATE, Ours-FAST, WebP and AVIF at 112 px and 512 B, on both
datasets, the cosine between the ArcFace embedding of every reconstruction and that
of its own aligned crop is computed (:func:`face1kb.eval.verification
.paired_cosines`; crops missing on either side are dropped), and its 5th percentile
(:func:`face1kb.eval.verification.linear_quantile`) and median are reported.

Outputs:

* ``quality/idcos_tail_<budget>.csv`` (under ``--csv-dir``, default
  ``config.output_dir("quality")``): ``dataset, codec, budget, n, p5, median``;
* ``idcos_tail.tex`` (``tab:idcos-tail``): p5 and median per dataset, highest value
  of a column shaded green and bold, lowest red and underlined.

The computation needs the ArcFace embedding arrays ``aligned_112`` and
``<codec>_112_<budget>`` (``experiments/embed/compute_embeddings.py``). With
``--render-only`` the table is written from an existing CSV (e.g. the shipped
``results/quality/idcos_tail_512.csv`` with ``--from-results``); ``--from-results``
and ``--input-root`` imply ``--render-only``. A computation that finds no embedding
pairs for any cell writes nothing and exits with status 1.

Examples
--------
    python experiments/accuracy/idcos_tail.py
    python experiments/accuracy/idcos_tail.py --render-only --from-results
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from face1kb import config
from face1kb.report.latex import BEST_FILL, WORST_FILL, finalize_table

log = logging.getLogger("idcos_tail")

#: Matcher of the identity cosine (independent of the codec's EdgeFace side stream).
MODEL = "arcface_antelopev2"
RES = 112
CODECS = [
    ("jpeg_ai", "JPEG-AI"),
    ("ours_accurate", "Ours-ACCURATE"),
    ("ours_fast", "Ours-FAST"),
    ("webp", "WebP"),
    ("avif", "AVIF"),
]


def compute(budget: int, datasets=config.DATASETS) -> pd.DataFrame:
    """Return ``n``, ``p5`` and ``median`` of the per-image cosine, per cell."""
    from face1kb.eval.embeddings import load_embeddings  # noqa: PLC0415
    from face1kb.eval.verification import (  # noqa: PLC0415
        linear_quantile,
        paired_cosines,
    )

    rows = []
    for ds in datasets:
        ref = load_embeddings(ds, MODEL, config.embedding_tag(RES))
        for codec, _ in CODECS:
            tag = config.embedding_tag(RES, "", codec, budget)
            rec = load_embeddings(ds, MODEL, tag)
            if ref is None or rec is None:
                log.warning("[%s] %s: embeddings missing", ds, codec)
                cos = np.empty(0)
            else:
                cos = paired_cosines(ref, rec)
            rows.append(
                {
                    "dataset": ds,
                    "codec": codec,
                    "budget": budget,
                    "n": int(cos.size),
                    "p5": float(linear_quantile(cos, 0.05)) if cos.size else np.nan,
                    "median": float(np.median(cos)) if cos.size else np.nan,
                }
            )
            r = rows[-1]
            log.info(
                "  %-11s %-14s n=%6d p5=%.3f median=%.3f",
                ds,
                codec,
                r["n"],
                r["p5"],
                r["median"],
            )
    return pd.DataFrame(rows)


def _fmt(v: float) -> str:
    return "--" if not np.isfinite(v) else f"{v:.3f}"


def table(df: pd.DataFrame) -> str:
    """Colour-shaded LaTeX tabular (higher is better)."""
    cols = [(d, k) for d in ("colorferet", "kk") for k in ("p5", "median")]
    best, worst = {}, {}
    for c in cols:
        vals = [
            df.loc[(df.dataset == c[0]) & (df.codec == name), c[1]].squeeze()
            for name, _ in CODECS
        ]
        vals = [float(v) for v in vals if np.isfinite(v)]
        if len(vals) > 1:
            best[c], worst[c] = max(vals), min(vals)
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "& \\multicolumn{2}{c}{Color~FERET} & "
        "\\multicolumn{2}{c}{AI-Solutions-KK} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Codec & p5 (worst) & median & p5 (worst) & median \\\\",
        "\\midrule",
    ]
    for codec, label in CODECS:
        cells = []
        for c in cols:
            row = df[(df.dataset == c[0]) & (df.codec == codec)]
            v = float(row[c[1]].iloc[0]) if not row.empty else np.nan
            txt = _fmt(v)
            if np.isfinite(v) and c in best:
                if abs(v - best[c]) < 1e-9:
                    txt = f"\\cellcolor{{{BEST_FILL}}}{txt}"
                elif abs(v - worst[c]) < 1e-9:
                    txt = f"\\cellcolor{{{WORST_FILL}}}{txt}"
            cells.append(txt)
        lines.append(f"    {label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Compute the tail statistics (or read them) and write the CSV and table."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument(
        "--render-only", action="store_true", help="write the table from a CSV"
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--input-root", default=None, help="CSV root (--render-only)")
    src.add_argument(
        "--from-results", action="store_true", help="read the shipped results/"
    )
    ap.add_argument("--csv-dir", default=None, help="default: outputs/quality")
    ap.add_argument("--out-dir", default=None, help="default: outputs/tables")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    name = f"idcos_tail_{args.budget}.csv"
    if args.from_results or args.input_root:
        args.render_only = True
    if args.render_only:
        root = (
            config.RESULTS_ROOT
            if args.from_results
            else Path(args.input_root or config.OUTPUT_ROOT)
        )
        df = pd.read_csv(root / "quality" / name)
    else:
        df = compute(args.budget)
        if not (df["n"] > 0).any():
            log.error(
                "no embeddings found for any cell; nothing written (render the "
                "shipped table with --from-results)"
            )
            return 1
        csv_dir = Path(args.csv_dir) if args.csv_dir else config.output_dir("quality")
        csv_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_dir / name, index=False)
        log.info("wrote %s", csv_dir / name)
    out_dir = Path(args.out_dir) if args.out_dir else config.output_dir("tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "idcos_tail.tex"
    out.write_text(finalize_table("idcos_tail", table(df)))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
