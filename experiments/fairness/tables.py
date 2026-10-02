# SPDX-License-Identifier: MIT
r"""LaTeX tables of the fairness section from the fairness CSVs (CPU only).

Reads ``<input-root>/fairness/`` (the outputs of ``subgroup.py``, ``fmr_fairness.py``
and ``disparity_ci.py``; ``--from-results`` reads the shipped paper results) and
writes one ``tabular`` per table into ``--out-dir`` (default
``$FACE1KB_OUTPUT_ROOT/tables``):

======================  ===================  =====================================
file                    paper table          content
======================  ===================  =====================================
fairness_subgroup_cf    tab:fairness-        Color FERET subgroup EER (%),
                        subgroup-cf          edgeface_xs, 112 px / 1024 B, pose
                                             and ethnicity
fairness_subgroup_kk    tab:fairness-        AI-Solutions-KK Monk skin-tone
                        subgroup-kk          subgroup EER (%), edgeface_xs,
                                             112 px / 1024 B
fairness_disparity_kk   tab:fairness-        KK skin-tone disparity (pp), uniform
                        disparity-kk         MST5/6/7/9/10 basis, 4 anchors, 1024 B
fairness_disparity_cf   tab:fairness-        Color FERET ethnicity and pose
                        disparity-cf         disparity (pp), 4 anchors, 1024 B,
                                             shaded
disparity_ci_kk         tab:disparity-ci     KK amplification ratio with its
                                             bootstrap CI
fairness_disparity_512  tab:fairness-512     512 B disparity with the 1024 B value
                                             in brackets (published unshaded)
fmr_fairness_cf         tab:fmr-fairness-cf  Color FERET ethnicity differential
                                             FMR (pp) at overall FMR 1e-2,
                                             1024 B, shaded
======================  ===================  =====================================

The first three are typed inline in the paper; their emitters produce the same rows
and values (the inline copies differ only in whitespace). Shaded tables follow the
paper convention (green best, red worst, bold / underline marker) through
:mod:`face1kb.report.latex`; the ``base (aligned)`` row never competes.

Examples
--------
    python experiments/fairness/tables.py --from-results
    python experiments/fairness/tables.py --tables fairness_disparity_cf --res 112
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from face1kb import config
from face1kb.data.cli_utils import parse_list, setup_logging
from face1kb.eval.fairness import MONK_UNIFORM_BASIS, disparity_pp, subgroup_spread_pp
from face1kb.report import latex

log = logging.getLogger("fairness.tables")

#: Anchor order of the disparity tables (model, header).
ANCHORS_CF = (
    ("arcface_antelopev2", "ArcFace"),
    ("lvface_l", "LVFace-L"),
    ("topofr_r100", "TopoFR"),
    ("edgeface_xs", "EdgeFace$^{\\S}$"),
)
#: Anchor order of the 512 B and KK 1024 B disparity tables (model, header).
ANCHORS_LOWER = (
    ("arcface_antelopev2", "arcface"),
    ("edgeface_xs", "edgeface"),
    ("lvface_l", "lvface"),
    ("topofr_r100", "topofr"),
)
#: Anchor order of the FMR table (model, header).
ANCHORS_FMR = (
    ("arcface_antelopev2", "ArcFace"),
    ("lvface_l", "LVFace-L"),
    ("topofr_r100", "TopoFR"),
    ("edgeface_xs", "EdgeFace"),
)
#: Anchor order of the disparity-CI table (model, header).
ANCHORS_CI = (
    ("arcface_antelopev2", "ArcFace"),
    ("lvface_l", "LVFace-L"),
    ("topofr_r100", "TopoFR-R100"),
    ("edgeface_xs", "EdgeFace-XS"),
)
#: Codec rows of the disparity tables (codec, label).
CODEC_ROWS = (
    ("aligned", "base (aligned)"),
    ("webp", "WebP"),
    ("avif", "AVIF"),
    ("heif", "HEIF"),
    ("jpeg_ai", "JPEG-AI"),
    ("jpeg", "JPEG"),
    ("jpeg_xl", "JPEG~XL"),
    ("jpeg_fzt", "JPEG-FzT"),
    ("jpeg2000", "JPEG~2000"),
    ("ours_accurate", "Ours-ACCURATE$^{\\S}$"),
    ("ours_fast", "Ours-FAST$^{\\S}$"),
)
#: Candidate rows of the FMR table; rows that print 0.00 everywhere are dropped.
FMR_ROWS = CODEC_ROWS + (
    ("neural_bmshj2018", "bmshj2018"),
    ("neural_mbt2018_mean", "mbt2018"),
)
#: Codec rows of the disparity-CI table (codec, label).
CI_ROWS = (
    ("webp", "WebP"),
    ("jpeg2000", "JPEG~2000"),
    ("ours_accurate", "Ours-ACCURATE"),
    ("ours_fast", "Ours-FAST"),
)
#: Columns of the subgroup-EER tables (codec, header).
SUBGROUP_COLUMNS = (
    ("aligned", "base"),
    ("webp", "WebP"),
    ("avif", "AVIF"),
    ("jpeg", "JPEG"),
    ("jpeg2000", "JPEG~2000"),
    ("ours_accurate", "Ours-ACC"),
    ("ours_fast", "Ours-FAST"),
)
#: Row blocks of the Color FERET subgroup table: (attribute, title, subgroups).
SUBGROUP_BLOCKS_CF = (
    ("pose", "Pose", ("frontal", "quarter", "half", "profile")),
    (
        "skin_tone",
        "Ethnicity",
        ("White", "Hispanic", "Asian", "Black", "Pacific-Islander"),
    ),
)
#: Subgroups of the AI-Solutions-KK subgroup table.
SUBGROUPS_KK = ("MST5", "MST6", "MST7", "MST8", "MST9", "MST10")
#: Printed decimals of the disparity tables.
DP = 2
#: First comment line of a generated table, and of the published copy
#: (``--paper-header``), by table.
HEADER = "% generated by experiments/fairness/tables.py"
PAPER_HEADERS = {
    "fairness_disparity_cf": "% generated by generate_fairness_disparity_cf.py",
    "fairness_disparity_512": "% generated by generate_fairness_512.py",
}

TABLES = (
    "fairness_subgroup_cf",
    "fairness_subgroup_kk",
    "fairness_disparity_kk",
    "fairness_disparity_cf",
    "disparity_ci_kk",
    "fairness_disparity_512",
    "fmr_fairness_cf",
)


# ------------------------------------------------------------ subgroup EER tables
def _subgroup_eer(df, codec: str, attribute: str, subgroup: str, *, model, res, budget):
    b = 0 if codec == "aligned" else budget
    r = df[
        (df.model == model)
        & (df.res == res)
        & (df.budget == b)
        & (df.codec == codec)
        & (df.attribute == attribute)
        & (df.subgroup == subgroup)
    ]
    return float(r.eer.iloc[0]) * 100 if len(r) else np.nan


def _pct(v: float) -> str:
    return "---" if np.isnan(v) else f"{v:.2f}"


def _subgroup_lines(df, blocks, *, model, res, budget) -> list[str]:
    width = max(len(g) for _a, _t, groups in blocks for g in groups)
    ncol = 1 + len(SUBGROUP_COLUMNS)
    lines = [
        "\\begin{tabular}{l" + "r" * len(SUBGROUP_COLUMNS) + "}",
        "\\toprule",
        "Subgroup & " + " & ".join(h for _c, h in SUBGROUP_COLUMNS) + " \\\\",
        "\\midrule",
    ]
    for i, (attribute, title, groups) in enumerate(blocks):
        if i:
            lines.append("\\midrule")
        if title:
            lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{title}}}}}\\\\")
        for g in groups:
            cells = [
                _pct(
                    _subgroup_eer(
                        df, c, attribute, g, model=model, res=res, budget=budget
                    )
                )
                for c, _h in SUBGROUP_COLUMNS
            ]
            lines.append(f"{g.ljust(width)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


def table_subgroup_cf(root: Path, res: int, budget: int, model: str) -> str:
    """Color FERET subgroup EER table (pose and ethnicity blocks)."""
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(root / "fairness_colorferet.csv")
    lines = _subgroup_lines(df, SUBGROUP_BLOCKS_CF, model=model, res=res, budget=budget)
    return "\n".join(lines) + "\n"


def table_subgroup_kk(root: Path, res: int, budget: int, model: str) -> str:
    """AI-Solutions-KK Monk skin-tone subgroup EER table."""
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(root / "fairness_kk.csv")
    blocks = (("skin_tone", "", SUBGROUPS_KK),)
    lines = _subgroup_lines(df, blocks, model=model, res=res, budget=budget)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------ disparity tables
def table_disparity_kk(root: Path, res: int, budget: int) -> str:
    """KK skin-tone disparity (pp) on the uniform Monk basis, all four anchors."""
    import pandas as pd  # noqa: PLC0415

    kk = pd.read_csv(root / "fairness_kk.csv")
    present = set(kk.codec.unique())
    width = max(len(lab) for _c, lab in CODEC_ROWS)
    lines = [
        "\\begin{tabular}{l" + "r" * len(ANCHORS_LOWER) + "}",
        "\\toprule",
        "Codec & " + " & ".join(h for _m, h in ANCHORS_LOWER) + " \\\\",
        "\\midrule",
    ]
    for codec, label in CODEC_ROWS:
        if codec not in present:
            continue
        cells = [
            _pct(subgroup_spread_pp(kk, model=m, codec=codec, budget=budget, res=res))
            for m, _h in ANCHORS_LOWER
        ]
        lines.append(f"{label.ljust(width)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _cf_block(df, attribute: str, title: str, budget: int, res: int) -> list[str]:
    """One attribute block of the Color FERET disparity table (colour only)."""
    sub = df[df.attribute == attribute]
    vals = {}
    for codec, _ in CODEC_ROWS:
        for model, _ in ANCHORS_CF:
            v, n = disparity_pp(
                sub,
                model=model,
                codec=codec,
                budget=budget,
                attribute=attribute,
                res=res,
            )
            if n:
                vals[(codec, model)] = (v, n)
    ncoh = sorted({n for (_v, n) in vals.values()})
    coh = (
        f"{ncoh[0]} subgroups" if len(ncoh) == 1 else f"{ncoh[0]}--{ncoh[-1]} subgroups"
    )
    # extremes over the codec rows at the printed precision (ties all shaded)
    ext = {}
    for model, _ in ANCHORS_CF:
        col = [
            round(v, DP)
            for (c, m), (v, _n) in vals.items()
            if m == model and c != "aligned"
        ]
        if col:
            ext[model] = (min(col), max(col))
    lines = [
        f"\\multicolumn{{{1 + len(ANCHORS_CF)}}}{{l}}{{\\emph{{{title}"
        f" $\\Delta$ ({coh})}}}}\\\\"
    ]
    for codec, label in CODEC_ROWS:
        cells = []
        for model, _ in ANCHORS_CF:
            got = vals.get((codec, model))
            if got is None:
                cells.append("---")
                continue
            v = round(got[0], DP)
            s = f"{v:.{DP}f}"
            if codec != "aligned" and model in ext:
                lo, hi = ext[model]
                if v == lo:
                    s = f"\\cellcolor{{{latex.BEST_FILL}}}{s}"
                elif v == hi:
                    s = f"\\cellcolor{{{latex.WORST_FILL}}}{s}"
            cells.append(s)
        lines.append(f"    {label} & " + " & ".join(cells) + " \\\\")
    return lines


def table_disparity_cf(root: Path, res: int, budget: int, header: str = HEADER) -> str:
    """Color FERET ethnicity / pose disparity table (shaded within each block)."""
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(root / "fairness_disparity_colorferet.csv")
    df = df[df.res == res]
    if df.empty:
        raise SystemExit(f"fairness_disparity_colorferet.csv has no rows at {res} px")
    lines = [
        header,
        "\\begin{tabular}{l" + "r" * len(ANCHORS_CF) + "}",
        "\\toprule",
    ]
    for i, (attribute, title) in enumerate(
        (("skin_tone", "Ethnicity"), ("pose", "Pose"))
    ):
        if i:
            lines.append("\\midrule")
        lines += _cf_block(df, attribute, title, budget, res)
        if i == 0:
            lines.insert(
                len(lines) - len(CODEC_ROWS),
                "Codec & " + " & ".join(n for _m, n in ANCHORS_CF) + " \\\\\n\\midrule",
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return latex.finalize_table("fairness_disparity_cf", "\n".join(lines) + "\n")


def _cell_512(v512: float, v1024: float, reduced: bool = False) -> str:
    if np.isnan(v512):
        return "---"
    mark = "$^{\\ddagger}$" if reduced else ""
    b = f"\\,{{\\scriptsize[{v1024:.2f}]}}" if not np.isnan(v1024) else ""
    return f"{v512:.2f}{mark}{b}"


def table_disparity_512(
    root: Path, res: int, budgets: tuple[int, int], header: str = HEADER
) -> str:
    """512 B disparity, 1024 B value in brackets (KK skin tone, CF ethnicity / pose)."""
    import pandas as pd  # noqa: PLC0415

    low, high = budgets
    kk = pd.read_csv(root / "fairness_kk.csv")
    cfd = pd.read_csv(root / "fairness_disparity_colorferet.csv")
    kk_present = set(kk.codec.unique())
    cf_present = set(cfd.codec.unique())
    ncol = 1 + len(ANCHORS_LOWER)

    def block(title: str, rows: list[str]) -> list[str]:
        return [
            f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{title}}}}}\\\\",
            "\\midrule",
        ] + rows

    kk_rows = []
    for c, lab in CODEC_ROWS:
        if c not in kk_present:
            continue
        cells = [
            _cell_512(
                subgroup_spread_pp(kk, model=a, codec=c, budget=low, res=res),
                subgroup_spread_pp(kk, model=a, codec=c, budget=high, res=res),
            )
            for a, _h in ANCHORS_LOWER
        ]
        kk_rows.append(f"    {lab} & " + " & ".join(cells) + " \\\\")

    def cf_rows(attr: str, base_n: int) -> list[str]:
        rr = []
        for c, lab in CODEC_ROWS:
            if c not in cf_present:
                continue
            cells = []
            for a, _h in ANCHORS_LOWER:
                v5, n5 = disparity_pp(
                    cfd, model=a, codec=c, budget=low, attribute=attr, res=res
                )
                v10, _ = disparity_pp(
                    cfd, model=a, codec=c, budget=high, attribute=attr, res=res
                )
                cells.append(_cell_512(v5, v10, bool(n5) and n5 < base_n))
            rr.append(f"    {lab} & " + " & ".join(cells) + " \\\\")
        return rr

    lines = [
        header,
        f"% each cell: {low} B disparity Delta (pp); [.] = {high} B value",
        "\\begin{tabular}{l" + "r" * len(ANCHORS_LOWER) + "}",
        "\\toprule",
        "Codec & " + " & ".join(h for _m, h in ANCHORS_LOWER) + " \\\\",
        "\\midrule",
    ]
    lines += block(
        "AI-Solutions-KK Monk skin-tone $\\Delta$ "
        f"(uniform {len(MONK_UNIFORM_BASIS)}-subgroup basis)",
        kk_rows,
    )
    lines += ["\\midrule"]
    for attr, name in (("skin_tone", "ethnicity"), ("pose", "pose")):
        # the subgroup count of the block is the mode over its rows
        r = cfd[(cfd.res == res) & (cfd.attribute == attr)]
        n = int(r.n_subgroups.mode().iloc[0])
        lines += block(
            f"Color~FERET {name} $\\Delta$ ({n} subgroups)", cf_rows(attr, n)
        )
        if attr == "skin_tone":
            lines += ["\\midrule"]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return latex.finalize_table("fairness_disparity_512", "\n".join(lines) + "\n")


def table_fmr_fairness_cf(root: Path, res: int, budget: int, target: float) -> str:
    """Color FERET ethnicity differential FMR (pp) at one target FMR."""
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(root / "fmr_fairness_colorferet.csv")
    ov = df[
        (df.attribute == "skin_tone")
        & (df.subgroup == "__overall__")
        & (df.res == res)
        & np.isclose(df.target_fmr, target)
    ]
    lines = [
        "\\begin{tabular}{l" + "r" * len(ANCHORS_FMR) + "}",
        "\\toprule",
        "Codec & " + " & ".join(h for _m, h in ANCHORS_FMR) + " \\\\",
        "\\midrule",
    ]
    for codec, label in FMR_ROWS:
        b = 0 if codec == "aligned" else budget
        cells = []
        for model, _h in ANCHORS_FMR:
            r = ov[(ov.codec == codec) & (ov.model == model) & (ov.budget == b)]
            cells.append(
                f"{float(r.fmr_disparity_pp.iloc[0]):.2f}" if len(r) else "---"
            )
        if all(c in ("0.00", "---") for c in cells):
            continue  # no threshold at this target (or no data): omitted
        lines.append(f"    {label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return latex.shade_table("\n".join(lines) + "\n", lower_better=True, refs=("base",))


def table_disparity_ci(root: Path) -> str:
    """KK amplification ratio with its cluster-bootstrap 95% CI, per anchor."""
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(root / "disparity_ci_kk.csv")

    def row(codec: str, model: str):
        r = df[(df.codec == codec) & (df.anchor == model)]
        return r.iloc[0] if len(r) else None

    lines = [
        "% subject-level cluster bootstrap (105 KK identities, B=500, 400k impostor "
        "sample)",
        "\\begin{tabular}{l" + "c" * len(ANCHORS_CI) + "}",
        "\\toprule",
        "Codec & " + " & ".join(h for _m, h in ANCHORS_CI) + " \\\\",
        " & " + " & ".join("ratio [95\\% CI]" for _ in ANCHORS_CI) + " \\\\",
        "\\midrule",
    ]
    base = []
    for model, _h in ANCHORS_CI:
        r = row("aligned", model)
        base.append("---" if r is None else f"{r.disparity_pp:.2f}")
    lines += [
        "\\emph{aligned} $\\Delta$ (pp) & " + " & ".join(base) + " \\\\",
        "\\midrule",
    ]
    for codec, label in CI_ROWS:
        cells = []
        for model, _h in ANCHORS_CI:
            r = row(codec, model)
            if r is None or pd.isna(r.get("ratio")):
                cells.append("---")
            else:
                cells.append(f"{r.ratio:.1f} [{r.ratio_lo:.1f},{r.ratio_hi:.1f}]")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return latex.shade_table(
        "\n".join(lines) + "\n", lower_better=True, refs=("aligned",)
    )


def build(name: str, root: Path, args) -> str:
    """Render table ``name`` from the CSVs in ``root``."""
    header = PAPER_HEADERS.get(name, HEADER) if args.paper_header else HEADER
    if name == "fairness_subgroup_cf":
        return table_subgroup_cf(root, args.res, args.budget, args.subgroup_model)
    if name == "fairness_subgroup_kk":
        return table_subgroup_kk(root, args.res, args.budget, args.subgroup_model)
    if name == "fairness_disparity_kk":
        return table_disparity_kk(root, args.res, args.budget)
    if name == "fairness_disparity_cf":
        return table_disparity_cf(root, args.res, args.budget, header)
    if name == "disparity_ci_kk":
        return table_disparity_ci(root)
    if name == "fairness_disparity_512":
        return table_disparity_512(root, args.res, (512, args.budget), header)
    if name == "fmr_fairness_cf":
        return table_fmr_fairness_cf(root, args.res, args.budget, args.target_fmr)
    raise ValueError(f"unknown table {name!r}; choose from {TABLES}")


def main(argv: list[str] | None = None) -> int:
    """Write the requested fairness tables."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="folder holding fairness/ (default: $FACE1KB_OUTPUT_ROOT)",
    )
    src.add_argument(
        "--from-results",
        action="store_true",
        help="read the shipped paper results (<repo>/results)",
    )
    ap.add_argument("--tables", default="all", help=f"'all' or a list of {TABLES}")
    ap.add_argument("--res", type=int, default=112)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--target-fmr", type=float, default=0.01)
    ap.add_argument("--subgroup-model", default="edgeface_xs")
    ap.add_argument(
        "--paper-header",
        action="store_true",
        help="write the first comment line of the published tables",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    setup_logging()

    base = config.RESULTS_ROOT if args.from_results else args.input_root
    root = (base or config.OUTPUT_ROOT) / "fairness"
    names = TABLES if args.tables == "all" else tuple(parse_list(args.tables))
    out_dir = args.out_dir or config.output_dir("tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        tex = build(name, root, args)
        out = out_dir / f"{name}.tex"
        out.write_text(tex)
        log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
