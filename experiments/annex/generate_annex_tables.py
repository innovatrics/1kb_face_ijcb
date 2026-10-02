# SPDX-License-Identifier: MIT
r"""LaTeX tables of the ISO/IEC 29794-5 annex study.

Reads the aggregates of ``score.py`` and ``ci.py``:

* ``scores.csv``: every cell scored at its own population (Experiment 1 arms and
  the confirmed winners at the full population of both datasets);
* ``scores_pop2000.csv``: the Color FERET cells scored on the 2000-crop prefix the
  grid ran on (like-for-like comparison of our configuration, the annexes, the grid
  and the Stage-B flag sweep);
* ``ci.csv``: the paired subject-level bootstrap;

and writes eight tables (EER in %, symmetric protocol, ``all`` population):

=====================  =================================================================
``annex_exp1.tex``     ``tab:annex-exp1``: annex configurations against ours, both
                       datasets; self-similarity and EER under ``cvlface_vit_b`` and EER
                       averaged over the four anchors
``annex_ci.tex``       ``tab:annex-ci``: paired bootstrap differences against ours;
                       ``$^\dagger$`` marks an interval that excludes zero
``annex_joint.tex``    ``tab:annex-joint``: ours, the best annex configuration and the
                       best grid/Stage-B cell (mean of ``arcface_antelopev2`` and
                       ``cvlface_vit_b``, 2000-crop prefix)
``annex_axes.tex``     ``tab:annex-axes``: mean EER over the grid at each level of each
                       axis
``annex_manip.tex``    ``tab:annex-manip``: manipulations at each codec's best
                       resolution and colour
``annex_selfsim.tex``  ``tab:annex-selfsim``: Spearman correlation of self-similarity
                       and EER under ``cvlface_vit_b``, and the cells each picks
``annex_bytes.tex``    ``tab:annex-bytes``: bytes freed by greyscale at a fixed quality
``annex_jpegai.tex``   ``tab:annex-jpegai``: the JPEG-AI arm
=====================  =================================================================

The grid is selected structurally (default flags, classical codec, grid resolution
and manipulation), not by the ``arm`` column: a grid cell that is also an
Experiment-1 or winner cell carries that arm instead.

Input: ``OUTPUT_ROOT/annex/`` (``--input-dir``, or the shipped paper results with
``--from-results``). Output: ``OUTPUT_ROOT/annex/`` (``--out-dir``). CPU, seconds.

Examples
--------
    python experiments/annex/generate_annex_tables.py --from-results
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from face1kb import config
from face1kb.data.cli_utils import setup_logging

log = logging.getLogger("annex.tables")

THEIRS = "cvlface_vit_b"
PAIR = ["arcface_antelopev2", THEIRS]
#: Our recommended resolution per codec (colour, no manipulation, default flags).
OURS = {
    "jpeg": 96,
    "webp": 96,
    "avif": 112,
    "heif": 96,
    "jpeg_xl": 112,
    "jpeg2000": 224,
}
GRID_RES = (56, 64, 80, 96, 112, 168, 224)
GRID_MANIP = ("none", "rectangle_mean", "ofiq_landmarks", "mean")


def _is_grid(d: pd.DataFrame) -> pd.Series:
    """Select the Stage-A grid structurally.

    The ``arm`` column cannot be used: a grid cell that is also an Exp-1 or a
    winner cell was re-recorded under that arm, which silently dropped e.g. the
    JPEG 96 px colour/no-manipulation cell out of the axis tables.
    """
    return (
        (d["flags"] == "{}")
        & d.codec.isin(OURS)
        & d.res.isin(GRID_RES)
        & d.manip.isin(GRID_MANIP)
    )


NAME = {
    "jpeg": "JPEG",
    "jpeg2000": "JPEG~2000",
    "jpeg_xl": "JPEG~XL",
    "jpeg_ai": "JPEG-AI",
    "webp": "WebP",
    "avif": "AVIF",
    "heif": "HEIF",
}


def load(src: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``scores.csv`` and ``scores_pop2000.csv``; add the ``kind`` column.

    ``kind`` is ``ours`` for our recommended configuration (colour, no
    manipulation, default flags, the codec's recommended resolution) and the
    ``arm`` otherwise. The uncompressed baseline rows are dropped.
    """
    a = pd.read_csv(src / "scores.csv")
    b = pd.read_csv(src / "scores_pop2000.csv")
    for d in (a, b):
        is_ours = d.apply(
            lambda r: r["flags"] == "{}"
            and r.color == "color"
            and r.manip == "none"
            and OURS.get(r.codec) == r.res,
            axis=1,
        )
        d["kind"] = np.where(is_ours, "ours", d.arm)
    return a[a.codec != "none"], b[b.codec != "none"]


def _cfg(r) -> str:
    f = (
        ""
        if r["flags"] == "{}"
        else ", " + ", ".join(f"{k}={v}" for k, v in json.loads(r["flags"]).items())
    )
    m = {
        "none": "--",
        "rectangle_mean": "rect",
        "ofiq_landmarks": "contour",
        "mean": "full blur",
    }[r.manip]
    return f"{r.res:.0f}\\,px {r.color}, {m}{f}".replace("_", "\\_")


def _wr(out: Path, name: str, head: str, rows: list[str], comment: str) -> None:
    body = "\n".join(rows)
    (out / name).write_text(
        f"% {comment}\n\\begin{{tabular}}{{{head}}}\n\\toprule\n{body}\n"
        f"\\bottomrule\n\\end{{tabular}}\n"
    )
    log.info("wrote %s", out / name)


def exp1(a: pd.DataFrame, out: Path) -> None:
    """Annex configurations vs ours, both datasets, both objectives."""
    x = a[a.kind.isin(["ours", "annexE", "annexF"])]
    th = x[x.model == THEIRS].set_index(["dataset", "codec", "kind"])
    an = x[x.model != THEIRS].groupby(["dataset", "codec", "kind"]).all_sym_eer.mean()
    rows = [
        r"Codec & Parameters & \multicolumn{3}{c}{Color~FERET} & "
        r"\multicolumn{3}{c}{AI-Solutions-KK} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r" & & self-sim & EER$_\text{their}$ & EER$_\text{anch}$ "
        r"& self-sim & EER$_\text{their}$ & EER$_\text{anch}$ \\",
        r"\midrule",
    ]
    for c in sorted(OURS):
        for k, lbl in (("ours", "ours"), ("annexE", "Annex~E"), ("annexF", "Annex~F")):
            if ("colorferet", c, k) not in th.index and ("kk", c, k) not in th.index:
                continue
            cells = []
            cfg = ""
            for ds in ("colorferet", "kk"):
                if (ds, c, k) not in th.index:
                    cells += ["--"] * 3
                    continue
                r = th.loc[(ds, c, k)]
                r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
                cfg = cfg or _cfg(r)
                cells += [
                    f"{r.self_sim:.3f}",
                    f"{100 * r.all_sym_eer:.2f}",
                    f"{100 * an[(ds, c, k)]:.2f}",
                ]
            rows.append(
                f"{NAME[c] if lbl == 'ours' else ''} & {lbl}: {cfg} & "
                + " & ".join(cells)
                + r" \\"
            )
    _wr(
        out,
        "annex_exp1.tex",
        "llcccccc",
        rows,
        "Exp 1: annex parameters vs ours, EER %, symmetric protocol",
    )


def joint(b: pd.DataFrame, out: Path) -> None:
    """Ours vs annex vs the joint sweep, like-for-like on the selection set."""
    x = b[b.model.isin(PAIR)]
    g = (
        x.groupby(["codec", "kind", "cell_id"])
        .agg(
            eer=("all_sym_eer", "mean"),
            res=("res", "first"),
            color=("color", "first"),
            manip=("manip", "first"),
            flags=("flags", "first"),
        )
        .reset_index()
    )
    rows = [
        r"Codec & Ours & Annex (best of E/F) & \multicolumn{2}{c}{Joint sweep "
        r"(this work)} \\",
        r"\cmidrule(lr){4-5}",
        r" & EER \% & EER \% & configuration & EER \% \\",
        r"\midrule",
    ]
    for c in sorted(OURS):
        y = g[g.codec == c]
        o = y[y.kind == "ours"].eer.min() * 100
        n = y[y.kind.isin(["annexE", "annexF"])].eer.min() * 100
        w = y[y.kind.isin(["grid", "stageB"])].nsmallest(1, "eer").iloc[0]
        rows.append(
            f"{NAME[c]} & {o:.3f} & {n:.3f} & {_cfg(w)} & "
            f"\\textbf{{{100 * w.eer:.3f}}} \\\\"
        )
    _wr(
        out,
        "annex_joint.tex",
        "lccrc",
        rows,
        "Exp 2: joint sweep vs both parameter sets, Color FERET 2000-crop prefix",
    )


def axes(b: pd.DataFrame, out: Path) -> None:
    """Marginal axis effects and the manipulation reversal."""
    g = b[_is_grid(b) & b.model.isin(PAIR)]
    k = g.groupby(["codec", "res", "color", "manip"]).all_sym_eer.mean().reset_index()
    rows = [
        r"Axis & Level & EER \% & Level & EER \% & Level & EER \% & Level & EER \% \\",
        r"\midrule",
    ]
    for ax, lv in (("res", [56, 64, 80, 96, 112, 168, 224]),):
        m = k.groupby(ax).all_sym_eer.mean()
        cells = [f"{v} & {100 * m[v]:.3f}" for v in lv]
        for i in range(0, len(cells), 4):
            rows.append(
                ("Resolution (px)" if i == 0 else "")
                + " & "
                + " & ".join(cells[i : i + 4])
                + " & " * 2 * (4 - len(cells[i : i + 4]))
                + r" \\"
            )
    for ax, lbl in (("color", "Colour"), ("manip", "Manipulation")):
        m = k.groupby(ax).all_sym_eer.mean()
        cells = [f"{str(v).replace('_', ' ')} & {100 * e:.3f}" for v, e in m.items()]
        rows.append(
            lbl + " & " + " & ".join(cells) + " & " * 2 * (4 - len(cells)) + r" \\"
        )
    _wr(
        out,
        "annex_axes.tex",
        "l" + "lc" * 4,
        rows,
        "Stage A marginal axis effects (mean EER over all cells)",
    )

    best = k.loc[k.groupby("codec").all_sym_eer.idxmin()][["codec", "res", "color"]]
    p = (
        k.merge(best, on=["codec", "res", "color"]).pivot_table(
            index="codec", columns="manip", values="all_sym_eer"
        )
        * 100
    )
    cols = ["none", "rectangle_mean", "ofiq_landmarks", "mean"]
    rows = [
        r"Codec & at best res/colour & \multicolumn{4}{c}{manipulation, EER \%} \\",
        r"\cmidrule(lr){3-6}",
        r" & & none & rect & contour & full blur \\",
        r"\midrule",
    ]
    for c in p.index:
        v = [p.loc[c, m] for m in cols]
        lo = min(v)
        rows.append(
            f"{NAME[c]} & {best[best.codec == c].res.iloc[0]:.0f}\\,px "
            f"{best[best.codec == c].color.iloc[0]} & "
            + " & ".join((f"\\textbf{{{x:.3f}}}" if x == lo else f"{x:.3f}") for x in v)
            + r" \\"
        )
    _wr(
        out,
        "annex_manip.tex",
        "llcccc",
        rows,
        "Manipulation held at each codec's best resolution/colour",
    )


def selfsim(b: pd.DataFrame, out: Path) -> None:
    """Tabulate whether their objective picks the EER-optimal configuration."""
    th = b[_is_grid(b) & (b.model == THEIRS)]
    rows = [
        r"Codec & $\rho$(self-sim, EER) & self-similarity's pick & EER \% & "
        r"EER-optimal pick & EER \% \\",
        r"\midrule",
    ]
    for c in sorted(OURS):
        x = th[th.codec == c]
        rho = spearmanr(x.self_sim, x.all_sym_eer).statistic
        s, e = x.loc[x.self_sim.idxmax()], x.loc[x.all_sym_eer.idxmin()]
        rows.append(
            f"{NAME[c]} & ${rho:+.2f}$ & {_cfg(s)} & {100 * s.all_sym_eer:.3f} & "
            f"{_cfg(e)} & \\textbf{{{100 * e.all_sym_eer:.3f}}} \\\\"
        )
    rho = spearmanr(th.self_sim, th.all_sym_eer).statistic
    rows.append(r"\midrule")
    rows.append(
        f"All cells & ${rho:+.2f}$ & \\multicolumn{{4}}{{l}}{{"
        f"$n={len(th)}$, $p<10^{{-60}}$}} \\\\"
    )
    _wr(
        out,
        "annex_selfsim.tex",
        "lclclc",
        rows,
        "Their self-similarity objective vs verification EER over the 336-cell grid",
    )


def bytes_freed(b: pd.DataFrame, out: Path) -> None:
    """Tabulate what greyscale buys in bytes at a fixed quality setting."""
    g = b[_is_grid(b) & (b.manip == "none") & b.model.isin(PAIR)]
    k = g.groupby(["codec", "res", "color"]).bytes_fixed_q.mean().unstack()
    k["saved"] = 100 * (1 - k.gray / k.color)
    res = sorted(k.index.get_level_values(1).unique())
    rows = [r"Codec & " + " & ".join(f"{r}\\,px" for r in res) + r" \\", r"\midrule"]
    for c in sorted(OURS):
        v = [k.loc[(c, r), "saved"] if (c, r) in k.index else np.nan for r in res]
        rows.append(f"{NAME[c]} & " + " & ".join(f"{x:.0f}" for x in v) + r" \\")
    _wr(
        out,
        "annex_bytes.tex",
        "l" + "c" * len(res),
        rows,
        "Bytes freed by greyscale at a fixed quality setting (% of the colour file)",
    )


def jpegai(b: pd.DataFrame, out: Path) -> None:
    """Tabulate the JPEG-AI arm (subsampled population)."""
    j = b[(b.codec == "jpeg_ai") & b.model.isin(PAIR)]
    k = (
        j.groupby(["res", "color", "manip"])
        .agg(
            eer=("all_sym_eer", "mean"),
            ss=("self_sim", "mean"),
            b=("bytes_median", "mean"),
        )
        .reset_index()
        .sort_values("eer")
    )
    rows = [
        r"Resolution & Colour & Manipulation & bytes & self-sim & EER \% \\",
        r"\midrule",
    ]
    for _, r in k.iterrows():
        m = {
            "none": "--",
            "rectangle_mean": "rect",
            "ofiq_landmarks": "contour",
            "mean": "full blur",
        }[r.manip]
        rows.append(
            f"{r.res:.0f}\\,px & {r.color} & {m} & {r.b:.0f} & "
            f"{r.ss:.3f} & {100 * r.eer:.3f} \\\\"
        )
    _wr(
        out,
        "annex_jpegai.tex",
        "lllccc",
        rows,
        "JPEG-AI arm of the joint grid, 500-crop subsample",
    )


def ci_table(src: Path, out: Path) -> None:
    """Tabulate the paired subject-bootstrap deltas against our configuration."""
    d = pd.read_csv(src / "ci.csv")
    rows = [
        r"Codec & \multicolumn{3}{c}{Color~FERET} & "
        r"\multicolumn{3}{c}{AI-Solutions-KK} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r" & ours & $\Delta$ annex & $\Delta$ sweep & ours & $\Delta$ annex "
        r"& $\Delta$ sweep \\",
        r"\midrule",
    ]
    for c in sorted(OURS):
        cells = []
        for ds in ("colorferet", "kk"):
            x = d[(d.dataset == ds) & (d.codec == c)]
            o = x[x.kind == "ours"]
            cells.append(f"{100 * o.eer.iloc[0]:.3f}" if len(o) else "--")
            for sel in (
                x[x.kind.isin(["annexE", "annexF"])].nsmallest(1, "eer"),
                x[x.kind == "winner"],
            ):
                if not len(sel):
                    cells.append("--")
                    continue
                r = sel.iloc[0]
                txt = f"${100 * r.d:+.3f}$ \\,[{100 * r.d_lo:+.2f},{100 * r.d_hi:+.2f}]"
                sig = r.d_lo > 0 or r.d_hi < 0  # CI excludes zero
                cells.append(txt + ("$^\\dagger$" if sig else ""))
        rows.append(f"{NAME[c]} & " + " & ".join(cells) + r" \\")
    _wr(
        out,
        "annex_ci.tex",
        "lcccccc",
        rows,
        "Paired subject-level bootstrap (200 reps) of EER differences against ours",
    )


def main(argv: list[str] | None = None) -> int:
    """Write every annex-study table."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--input-dir", default=None, help="folder of the three CSVs")
    src.add_argument(
        "--from-results", action="store_true", help="read the shipped paper results"
    )
    ap.add_argument("--out-dir", default=None, help="default OUTPUT_ROOT/annex")
    args = ap.parse_args(argv)
    setup_logging()
    if args.input_dir:
        src_dir = Path(args.input_dir)
    elif args.from_results:
        src_dir = config.results_dir("annex")
    else:
        src_dir = config.output_dir("annex")
    for name in ("scores.csv", "scores_pop2000.csv", "ci.csv"):
        if not (src_dir / name).is_file():
            raise SystemExit(f"{src_dir / name} not found")
    out = Path(args.out_dir) if args.out_dir else config.output_dir("annex")
    out.mkdir(parents=True, exist_ok=True)
    a, b = load(src_dir)
    exp1(a, out)
    joint(b, out)
    axes(b, out)
    selfsim(b, out)
    bytes_freed(b, out)
    jpegai(b, out)
    ci_table(src_dir, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
