# SPDX-License-Identifier: MIT
"""Table generators of experiments/accuracy on small synthetic metrics frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face1kb.eval.verification import METRICS_COLUMNS

from ._scripts import load


def _row(dataset, model, codec, res, budget, eer, **kw):
    kind = "aligned" if codec == "" else "compressed"
    tag = f"aligned_{res}" if kind == "aligned" else f"{codec}_{res}_{budget}"
    row = {c: np.nan for c in METRICS_COLUMNS}
    row.update(
        dataset=dataset,
        model=model,
        tag=tag,
        kind=kind,
        res=res,
        suffix=np.nan,
        codec=np.nan if kind == "aligned" else codec,
        budget=0 if kind == "aligned" else budget,
        n_pos=kw.pop("n_pos", 1000),
        n_neg=kw.pop("n_neg", 100000),
        eer=eer,
    )
    row.update(kw)
    return row


def _frame(rows):
    return pd.DataFrame(rows, columns=list(METRICS_COLUMNS))


# ------------------------------------------------------------------- rate_eer
def test_rate_eer_table_shading_and_dagger():
    m = load("rate_eer")
    rows = [
        _row("kk", "lvface_l", "", 112, 0, 0.001, eer_lo=0.0005, eer_hi=0.002),
        _row("kk", "lvface_l", "webp", 112, 1024, 0.01),
        _row("kk", "lvface_l", "jpeg", 112, 1024, 0.05),
        _row("kk", "lvface_l", "avif", 112, 1024, 0.02),
        _row("kk", "lvface_l", "webp", 112, 512, 0.03),
    ]
    tex = m.table(_frame(rows), "kk", "lvface_l", {("kk", "jpeg", 112, 1024)}, "% h")
    lines = tex.splitlines()
    assert lines[0] == "% h"
    assert "aligned (ref) & 0.10 (0.05–0.20) &  \\\\" in lines
    assert "WebP & \\cellcolor{green!25}1.00 & 3.00 \\\\" in lines
    assert "JPEG & \\cellcolor{red!22}5.00$^\\dagger$ & -- \\\\" in lines
    assert "AVIF & 2.00 & -- \\\\" in lines
    assert not any(ln.startswith("HEIF") for ln in lines)  # no data: row omitted
    final = m.finalize_table("rate_eer_kk_lvface_l", tex)
    assert "\\cellcolor{red!22}\\underline{5.00$^\\dagger$}" in final


def test_rate_eer_over_budget(tmp_path):
    m = load("rate_eer")
    p = tmp_path / "fs.csv"
    pd.DataFrame(
        {
            "dataset": ["kk", "kk"],
            "res": [112, 112],
            "budget": [512, 1024],
            "codec": ["jpeg", "webp"],
            "pct_holding": [99.4, 99.5],
        }
    ).to_csv(p, index=False)
    assert m.over_budget(p) == {("kk", "jpeg", 112, 512)}
    assert m.over_budget(tmp_path / "missing.csv") == set()


# -------------------------------------------------------------------- frr_far
def _frr_rows(res=112, budget=1024):
    rows = []
    for model in ("arcface_antelopev2", "lvface_l"):
        rows.append(
            _row(
                "colorferet",
                model,
                "",
                res,
                0,
                0.001,
                **{"fnmr_0.001": 0.002, "fnmr_0.0001": 0.004},
            )
        )
        for codec, f4, n_pos in (
            ("webp", 0.02, 1000),
            ("jpeg", 0.50, 1000),
            ("ours_fast", 0.01, 1000),
            ("neural_bmshj2018", 0.001, 30),
        ):
            rows.append(
                _row(
                    "colorferet",
                    model,
                    codec,
                    res,
                    budget,
                    0.01,
                    n_pos=n_pos,
                    **{"fnmr_0.001": f4 / 2, "fnmr_0.0001": f4},
                )
            )
    return rows


def test_frr_table_order_support_and_shading():
    m = load("frr_far")
    tex = m.frr_table(_frame(_frr_rows()), "colorferet", 1024, 112, "% h")
    labels = ("Ours", "WebP", "JPEG", "bm")
    body = [ln for ln in tex.splitlines() if ln.startswith(labels)]
    # ordered by mean FNMR@1e-4; the 30-pair CompressAI codec is marked, not ranked
    assert [ln.split(" & ")[0] for ln in body] == [
        "bmshj2018$^{\\ddagger}$",
        "Ours-FAST",
        "WebP",
        "JPEG",
    ]
    assert "\\cellcolor" not in body[0]
    assert body[1].split(" & ")[2].startswith("\\cellcolor{green!25}1.00")
    assert body[3].split(" & ")[2].startswith("\\cellcolor{red!22}50.00")


def test_frr_pick_prefers_populated_row():
    m = load("frr_far")
    sub = pd.DataFrame({"n_neg": [3, 5000, 2000], "eer": [0.0, 0.1, 0.2]})
    assert m._pick(sub).eer == 0.1
    assert m._pick(sub.iloc[:1]) is None


def test_model_effect_ties_are_all_shaded():
    m = load("frr_far")
    rows = []
    for codec, eer in (("webp", 0.01), ("avif", 0.01), ("jpeg", 0.05)):
        rows.append(_row("kk", "lvface_l", codec, 112, 1024, eer))
    got = m.model_effect_matrix(_frame(rows), "kk", 1024, 112)
    tex = m.model_effect_table(got, "% h")
    row = next(ln for ln in tex.splitlines() if ln.startswith("LVFace-L"))
    assert row.count("green!25") == 2 and row.count("red!22") == 1


def test_summary_table_uses_two_matchers():
    m = load("frr_far")
    tex = m.summary_table(_frame(_frr_rows()), "% h")
    row = next(ln for ln in tex.splitlines() if ln.startswith("Ours-FAST"))
    assert row.split(" & ")[1].endswith("1.00")


# ---------------------------------------------------------------- best_config
def test_best_config_needs_both_matchers():
    m = load("best_config")
    rows = []
    for res, a, b in ((112, 0.02, 0.04), (224, 0.01, 0.01), (168, 0.001, np.nan)):
        rows.append(
            _row(
                "colorferet",
                "arcface_antelopev2",
                "webp",
                res,
                1024,
                0.1,
                **{"fnmr_0.0001": a},
            )
        )
        rows.append(
            _row("colorferet", "lvface_l", "webp", res, 1024, 0.1, **{"fnmr_0.0001": b})
        )
    df = _frame(rows)
    assert m.best_resolution(df, "webp", 1024) == (pytest.approx(0.01), 224)
    assert m.best_resolution(df, "webp", 512) is None
    tex = m.table(df)
    assert "WebP          & 1.00 @224 & -- \\\\" in tex


# ------------------------------------------------------------------- h4_curve
def test_h4_low_support_is_per_budget():
    m = load("h4_curve")
    rows = [
        _row("colorferet", "arcface_antelopev2", c, 112, b, 0.01, n_pos=n)
        for c, b, n in (
            ("webp", 512, 1000),
            ("jpeg", 512, 1000),
            ("neural_bmshj2018", 512, 30),
            ("webp", 768, 1000),
            ("neural_bmshj2018", 768, 1000),
        )
    ]
    weak = m._low_support(_frame(rows), "arcface_antelopev2", [512, 768])
    assert weak == {("neural_bmshj2018", 512)}


# --------------------------------------------------------- significance_tables
def test_significance_formatting():
    m = load("significance_tables")
    assert m._p(0.0) == "$\\approx 0$"
    assert m._p(2.5e-7) == "$2.5\\times10^{-7}$"
    assert m._p(0.3456) == "0.346"
    assert m._chi(3.14159) == "3.14"
    assert m._chi(12345.67) == "12,345.7"


def test_significance_row_selection():
    m = load("significance_tables")
    cell = pd.DataFrame(
        {
            "codec_a": ["aligned", "aligned", "avif", "jpeg", "avif"],
            "codec_b": ["jpeg", "webp", "webp", "webp", "heif"],
            "mcnemar_chi2": [900.0, 50.0, 0.5, 700.0, 2.0],
            "significant": [True, True, False, True, False],
        }
    )
    rows = m._rows_for(cell)
    # the two non-significant pairs, then the largest chi2 (also the aligned one)
    assert list(rows.index) == [2, 4, 0]


# -------------------------------------------------------------- posthoc_stats
def test_posthoc_outputs_format():
    m = load("posthoc_stats")
    res = {
        "n": 3,
        "k": 2,
        "friedman_chi2": 3.0,
        "friedman_p": 0.0833,
        "kendall_w": 1.0,
        "mean_rank": pd.Series({"webp": 1.0, "jpeg": 2.0}),
        "pairs": [("webp", "jpeg", 0.25, 0.25, -1.0)],
    }
    assert m.scalars(res) == (
        "n=3 k=2 friedman_chi2=3.0 friedman_p=8.33e-02 kendall_w=1.000\n"
        "mean_rank: WebP=1.00, JPEG=2.00\n"
    )
    assert "WebP & JPEG & 0.25 & -1.00 & \\textbf{no} \\\\" in m.table(res, "% h")
