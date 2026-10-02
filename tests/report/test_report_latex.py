# SPDX-License-Identifier: MIT
"""Shared LaTeX shading helper (face1kb.report.latex)."""

from __future__ import annotations

from face1kb.report import latex as L

TABLE = r"""\begin{tabular}{lrrr}
\toprule
Codec & EER & id-cos & Bytes \\
\midrule
aligned & 0.10 & 1.000 & -- \\
JPEG & 2.89 & 0.901 & 1020 \\
WebP & 1.06 & 0.947 & 1011 \\
Ours-ACC & 1.26 & 0.947 & 1003 \\
JPEG 2000 & 37.82 & 0.612 & 1024 \\
\bottomrule
\end{tabular}"""


def test_cell_writers_match_emphasis_pass():
    assert L.shade_best("0.947") == r"\cellcolor{green!25}\textbf{0.947}"
    assert L.shade_worst("37.82") == r"\cellcolor{red!22}\underline{37.82}"
    assert L.shade_best("1.0", emphasis=False) == r"\cellcolor{green!25}1.0"
    row = r"X & \cellcolor{green!25}0.947 & \cellcolor{red!22}37.82$^\dagger$ \\"
    out, n = L.add_emphasis(row)
    assert n == 2
    assert out == (
        r"X & \cellcolor{green!25}\textbf{0.947} & "
        r"\cellcolor{red!22}\underline{37.82$^\dagger$} \\"
    )
    assert L.add_emphasis(out) == (out, 0)  # idempotent
    assert L.strip_emphasis(out) == row


def test_shade_table_lower_is_better_with_reference_row():
    out = L.shade_table(TABLE, lower_better=True, refs=("aligned",))
    rows = {ln.split("&")[0].strip(): ln for ln in out.splitlines() if "&" in ln}
    assert r"\cellcolor{green!25}\textbf{1.06}" in rows["WebP"]
    assert r"\cellcolor{red!22}\underline{37.82}" in rows["JPEG 2000"]
    assert "cellcolor" not in rows["aligned"]  # reference row is not ranked
    # One direction applies to every column: with lower_better the id-cos minimum
    # is "best" and both cells of the tied maximum are "worst".
    assert r"\cellcolor{green!25}\textbf{0.612}" in rows["JPEG 2000"]
    assert r"\cellcolor{red!22}\underline{0.947}" in rows["WebP"]
    assert r"\cellcolor{red!22}\underline{0.947}" in rows["Ours-ACC"]
    assert r"\cellcolor{green!25}\textbf{1003}" in rows["Ours-ACC"]
    assert L.shade_table(out, lower_better=True, refs=("aligned",)) == out


def test_ties_are_all_marked_and_constant_columns_unshaded():
    t = TABLE.replace("1020", "1024").replace("1011", "1024").replace("1003", "1024")
    out = L.shade_table(t, lower_better=False, refs=("aligned",))
    assert out.count(r"\cellcolor{green!25}\textbf{0.947}") == 2
    assert "1024 \\\\" in out and r"\cellcolor{green!25}\textbf{1024}" not in out


def test_published_exceptions():
    assert len(L.NO_EMPHASIS_TABLES) == 13
    assert "frr_far_summary" in L.NO_EMPHASIS_TABLES
    assert "model_effect_kk_1024_224" in L.NO_EMPHASIS_TABLES
    assert not L.emphasis_for("frr_far_colorferet_512")
    assert L.emphasis_for("codec_results")
    colour_only = r"A & \cellcolor{green!25}1.00 \\"
    assert L.finalize_table("frr_far_kk_1024", colour_only) == colour_only
    assert L.finalize_table("codec_results", colour_only) == (
        r"A & \cellcolor{green!25}\textbf{1.00} \\"
    )
    assert L.finalize_table("fairness_disparity_512", colour_only) == colour_only


def test_unshade_columns():
    src = "\\midrule\nSOP & 1 & \\cellcolor{red!22}\\underline{1108} \\\\"
    assert L.unshade_columns(src, (2,)) == "\\midrule\nSOP & 1 & 1108 \\\\"
    speed = (
        "\\midrule\nHOP & 3 & \\cellcolor{red!22}\\underline{1192} & 656 & 1945 & "
        "35.93 & 0.919 & 0.949 & \\cellcolor{green!25}\\textbf{1109} \\\\"
    )
    out = L.finalize_table("jpegai_speed", L.strip_emphasis(speed))
    assert out.endswith("& 1109 \\\\")
    assert r"\cellcolor{red!22}\underline{1192}" in out


def test_cell_value():
    assert L.cell_value(r" \cellcolor{green!25}\textbf{0.57 @224} ") == 0.57
    assert L.cell_value("1.4 [0.8,6.0]") == 1.4
    assert L.cell_value(" -- ") is None
    assert L.cell_value("n/a") is None
