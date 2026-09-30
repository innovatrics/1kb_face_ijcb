# SPDX-License-Identifier: MIT
r"""Budget compliance of the stored bitstreams: size summary, fit-rate tables, figures.

Each codec controls its rate differently, so the achieved size at a byte budget varies
per crop. This script ``stat()``\ s every stored bitstream of the benchmark grid
(``WORK_ROOT/<dataset>/compressed/<res>px_<budget>B/<codec>/``; only the file sizes
are read) and writes

* ``OUTPUT_ROOT/codec_comparison/file_size_summary.csv``: per ``(dataset, res,
  budget, codec)`` cell ``n, min, mean, median, max, p5, p95, over_budget_pct,
  pct_holding`` (``pct_holding`` = % of files with ``size <= budget``);
* ``OUTPUT_ROOT/tables/fit_rate_<dataset>_<budget>.tex``: % of crops holding the
  budget, codec x resolution (paper tables ``tab:fit-kk``, ``tab:fit-cf``,
  ``tab:fit-1024``);
* ``OUTPUT_ROOT/figures/file_size_compliance.png`` (``fig:size-compliance``),
  ``file_size_boxplots.png`` (``fig:size-box``, 112 px panels) and
  ``file_size_boxplots_all.png`` (every cell).

``--from-summary`` re-renders the tables and the compliance figure from an existing
summary CSV (default: the shipped ``results/codec_comparison/file_size_summary.csv``
with ``--from-results``) without any bitstream; the boxplots need the files.

Table rules: a cell with fewer than :data:`MIN_N` files reads ``--``; a codec with
no reported cell is left out; per resolution column the lowest rounded value is
shaded worst, and the highest only if it is unique (most codecs hold 100 %); a
column whose rounded values all tie is not shaded. The emphasis marker of the paper
tables is added by :func:`face1kb.report.latex.finalize_table`.

Examples
--------
    python experiments/compress/generate_file_size_boxplots.py
    python experiments/compress/generate_file_size_boxplots.py --from-summary \
        --from-results
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from face1kb import config
from face1kb.report import latex, style

log = logging.getLogger("file_sizes")

#: Minimum number of files before a compliance percentage is reported.
MIN_N = 30
#: Resolution of the paper's boxplot figure.
REPORT_RES = 112
#: Codecs in table order.
CODECS = (
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
#: Row labels of the fit-rate tables and figures (the paper abbreviates ACCURATE).
NAME = {**style.CODEC_LABELS, "ours_accurate": "Ours-ACC"}
SUMMARY_COLUMNS = (
    "dataset",
    "res",
    "budget",
    "codec",
    "n",
    "min",
    "mean",
    "median",
    "max",
    "p5",
    "p95",
    "over_budget_pct",
    "pct_holding",
)


# ------------------------------------------------------------------- scanning
def cell_sizes(dataset: str, res: int, budget: int, codec: str) -> np.ndarray:
    """Sizes (bytes) of the stored bitstreams of one cell."""
    from face1kb.baselines import extension  # noqa: PLC0415

    d = config.compressed_dir(dataset, res, budget, codec)
    if not d.is_dir():
        return np.array([], dtype=np.int64)
    return np.array(
        [p.stat().st_size for p in d.rglob(f"*{extension(codec)}") if p.is_file()],
        dtype=np.int64,
    )


def summary_row(dataset: str, res: int, budget: int, codec: str, s) -> dict:
    """Summary statistics of one cell's sizes ``s`` (non-empty)."""
    s = np.asarray(s)
    return {
        "dataset": dataset,
        "res": res,
        "budget": budget,
        "codec": codec,
        "n": int(s.size),
        "min": int(s.min()),
        "mean": float(s.mean()),
        "median": float(np.median(s)),
        "max": int(s.max()),
        "p5": float(np.percentile(s, 5)),
        "p95": float(np.percentile(s, 95)),
        "over_budget_pct": float(100 * np.mean(s > budget)),
        "pct_holding": float(100 * np.mean(s <= budget)),
    }


def scan(datasets, resolutions, budgets, codecs=CODECS):
    """Return ``(summary DataFrame, panels)``; panels hold the raw sizes per cell."""
    rows, panels = [], []
    for ds in datasets:
        for res in resolutions:
            for budget in budgets:
                cell = {}
                for codec in codecs:
                    s = cell_sizes(ds, res, budget, codec)
                    if s.size == 0:
                        continue
                    cell[codec] = s
                    rows.append(summary_row(ds, res, budget, codec, s))
                if cell:
                    panels.append((ds, res, budget, cell))
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS)), panels


# ------------------------------------------------------------------- tables
def column_extremes(mat: dict, codecs: list, ncol: int):
    """Per column the unique rounded maximum (or None) and the rounded minimum.

    Ranking uses the printed (rounded) value; a column whose rounded values all tie
    gets ``(None, None)``.
    """
    hi, lo = [], []
    for j in range(ncol):
        col = [round(mat[c][j]) for c in codecs if mat[c][j] is not None]
        if len(set(col)) < 2:
            hi.append(None)
            lo.append(None)
            continue
        top = max(col)
        hi.append(top if col.count(top) == 1 else None)
        lo.append(min(col))
    return hi, lo


def fit_cell(v, hi, lo) -> str:
    """One compliance cell, shaded against its column's extremes."""
    if v is None:
        return "--"
    v = round(v)
    cell = f"{v:.0f}"
    if hi is not None and v == hi:
        return latex.shade_best(cell, emphasis=False)
    if lo is not None and v == lo:
        return latex.shade_worst(cell, emphasis=False)
    return cell


def fit_table(df: pd.DataFrame, dataset: str, budget: int, resolutions) -> str:
    """Colour-only LaTeX fit-rate table of one ``(dataset, budget)``."""
    sub = df[(df.dataset == dataset) & (df.budget == budget)]
    codecs = [c for c in CODECS if c in set(sub.codec)]
    lines = [
        "% generated by generate_file_size_boxplots.py",
        "\\begin{tabular}{l" + "r" * len(resolutions) + "}",
        "\\toprule",
        "Codec & " + " & ".join(f"{r}\\,px" for r in resolutions) + " \\\\",
        "\\midrule",
    ]
    mat = {}
    for c in codecs:
        row = []
        for r in resolutions:
            v = sub[(sub.codec == c) & (sub.res == r)]
            if v.empty or int(v.iloc[0]["n"]) < MIN_N:
                row.append(None)
            else:
                row.append(float(v.iloc[0]["pct_holding"]))
        mat[c] = row
    codecs = [c for c in codecs if any(v is not None for v in mat[c])]
    hi, lo = column_extremes(mat, codecs, len(resolutions))
    for c in codecs:
        vals = [fit_cell(v, hi[j], lo[j]) for j, v in enumerate(mat[c])]
        lines.append(NAME.get(c, c) + " & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_fit_tables(df, out_dir: Path, datasets, budgets, resolutions) -> list[Path]:
    """Write ``fit_rate_<dataset>_<budget>.tex`` (with the paper's emphasis)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ds in datasets:
        for b in budgets:
            stem = f"fit_rate_{ds}_{b}"
            path = out_dir / f"{stem}.tex"
            path.write_text(
                latex.finalize_table(stem, fit_table(df, ds, b, resolutions))
            )
            paths.append(path)
    log.info("wrote %d fit-rate tables to %s", len(paths), out_dir)
    return paths


# ------------------------------------------------------------------- figures
def write_boxplots(panels, path: Path, ncol: int, base: int) -> None:
    """Achieved-size boxplots (whiskers at min/max, mean marker, budget line)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if not panels:
        return
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.8 * nrow), squeeze=False)
    for k, (ds, res, budget, cell) in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        names = list(cell)
        ax.boxplot(
            [cell[n] for n in names], showfliers=False, showmeans=True, whis=(0, 100)
        )
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels([NAME.get(n, n) for n in names], rotation=90, fontsize=base)
        ax.tick_params(axis="y", labelsize=base)
        ax.axhline(budget, ls="--", c="r", lw=1.4)
        ax.set_title(f"{ds} {res}px / {budget}B", fontsize=base + 2)
        ax.set_ylabel("bytes", fontsize=base + 1)
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=style.FIGURE_DPI["file_size"])
    plt.close(fig)
    log.info("wrote %s (%d panels)", path, len(panels))


def write_compliance_heatmap(df, path: Path, datasets, budgets, resolutions) -> None:
    """Heatmap of the % of crops holding the budget (same ``MIN_N`` floor)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cells = [(ds, b) for ds in datasets for b in budgets]
    fig, axes = plt.subplots(
        1, len(cells), figsize=(5.0 * len(cells), 4.6), squeeze=False
    )
    fig.subplots_adjust(wspace=0.45)
    im = None
    for j, (ds, b) in enumerate(cells):
        ax = axes[0][j]
        sub = df[(df.dataset == ds) & (df.budget == b)]
        codecs = [c for c in CODECS if c in set(sub.codec)]
        mat = np.full((len(codecs), len(resolutions)), np.nan)
        for ci, c in enumerate(codecs):
            for ri, r in enumerate(resolutions):
                v = sub[(sub.codec == c) & (sub.res == r)]
                if not v.empty and int(v.iloc[0]["n"]) >= MIN_N:
                    mat[ci, ri] = v.iloc[0]["pct_holding"]
        im = ax.imshow(
            mat, vmin=0, vmax=100, cmap=style.HEATMAP_CMAPS["fit_rate"], aspect="auto"
        )
        ax.set_xticks(range(len(resolutions)))
        ax.set_xticklabels([str(r) for r in resolutions], fontsize=7)
        ax.set_yticks(range(len(codecs)))
        ax.set_yticklabels([NAME.get(c, c) for c in codecs], fontsize=7)
        ax.set_xlabel("resolution (px)", fontsize=8)
        ax.set_title(f"{ds} / {b}B", fontsize=9)
        for ci in range(len(codecs)):
            for ri in range(len(resolutions)):
                if not np.isnan(mat[ci, ri]):
                    ax.text(
                        ri,
                        ci,
                        f"{mat[ci, ri]:.0f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                    )
    fig.suptitle(
        "% of crops holding the byte budget (green = always fits)", fontsize=10
    )
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, label="% <= budget")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=style.FIGURE_DPI["file_size"], bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)


# ------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--from-summary",
        action="store_true",
        help="render tables and the compliance figure from a summary CSV (no scan)",
    )
    ap.add_argument(
        "--from-results",
        action="store_true",
        help="read the summary from the shipped results/ (with --from-summary)",
    )
    ap.add_argument("--input-root", default=None, help="root holding the summary CSV")
    ap.add_argument("--output-root", default=None, help="override FACE1KB_OUTPUT_ROOT")
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument("--resolutions", default=",".join(map(str, config.RESOLUTIONS)))
    ap.add_argument("--budgets", default=",".join(map(str, config.BUDGETS)))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    style.use_agg()

    out_root = Path(args.output_root) if args.output_root else config.OUTPUT_ROOT
    datasets = [d for d in args.datasets.split(",") if d]
    resolutions = [int(r) for r in args.resolutions.split(",") if r]
    budgets = [int(b) for b in args.budgets.split(",") if b]
    fig_dir = out_root / "figures"

    if args.from_summary:
        in_root = Path(
            args.input_root
            or (config.RESULTS_ROOT if args.from_results else config.OUTPUT_ROOT)
        )
        src = in_root / "codec_comparison" / "file_size_summary.csv"
        if not src.is_file():
            raise SystemExit(f"no summary {src}; run without --from-summary first")
        df = pd.read_csv(src)
    else:
        df, panels = scan(datasets, resolutions, budgets)
        if df.empty:
            raise SystemExit("no stored bitstreams found under FACE1KB_WORK_ROOT")
        dst = out_root / "codec_comparison" / "file_size_summary.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst, index=False)
        log.info("wrote %s (%d cells)", dst, len(df))
        if not args.no_figures:
            write_boxplots(
                [p for p in panels if p[1] == REPORT_RES],
                fig_dir / "file_size_boxplots.png",
                ncol=2,
                base=13,
            )
            write_boxplots(panels, fig_dir / "file_size_boxplots_all.png", 4, 8)
    if not args.no_figures:
        write_compliance_heatmap(
            df, fig_dir / "file_size_compliance.png", datasets, budgets, resolutions
        )
    write_fit_tables(df, out_root / "tables", datasets, budgets, resolutions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
