# SPDX-License-Identifier: MIT
"""Tests of face1kb.eval.significance (CPU, synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face1kb.eval import significance as S

from . import _reference as R


def _random_arrays(rng):
    n = int(rng.integers(1, 4000))
    yield rng.normal(size=n).astype(np.float32)
    yield rng.integers(0, 15, n).astype(np.float64)  # heavy ties
    yield np.round(rng.normal(size=n), 2).astype(np.float32)
    z = np.concatenate([np.zeros(n // 2), -np.zeros(n - n // 2)])  # +0 / -0 ties
    rng.shuffle(z)
    yield z
    yield np.full(n, 0.25, np.float32)


@pytest.mark.parametrize("seed", range(20))
def test_midrank_identical_to_loop(seed):
    rng = np.random.default_rng(seed)
    for x in _random_arrays(rng):
        got = S.midrank(x)
        ref = R.midrank_loop(x)
        assert got.dtype == np.float64
        assert np.array_equal(got, ref)


def test_midrank_small_cases():
    np.testing.assert_array_equal(
        S.midrank(np.array([3.0, 1.0, 3.0, 2.0])), [3.5, 1, 3.5, 2]
    )
    assert S.midrank(np.array([], np.float32)).size == 0
    # every NaN is its own block (the scores never contain NaN in the tests)
    r = S.midrank(np.array([1.0, np.nan, np.nan]))
    assert r[0] == 1.0 and sorted(r[1:]) == [2.0, 3.0]


def test_delong_auc_is_mann_whitney():
    rng = np.random.default_rng(0)
    pos = np.round(rng.normal(1, 1, 300), 1).astype(np.float32)
    neg = np.round(rng.normal(0, 1, 500), 1).astype(np.float32)
    auc, v10, v01 = S.delong_components(pos, neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    assert auc == pytest.approx(gt + 0.5 * eq, abs=1e-12)
    assert v10.shape == pos.shape and v01.shape == neg.shape
    assert v10.mean() == pytest.approx(auc) and v01.mean() == pytest.approx(auc)


def test_delong_paired_identical_and_different():
    rng = np.random.default_rng(1)
    pos = rng.normal(1, 1, 400)
    neg = rng.normal(0, 1, 600)
    a, b, d, p = S.delong_paired(pos, neg, pos, neg)
    assert d == 0 and p == 1.0 and a == b
    a, b, d, p = S.delong_paired(pos, neg, pos * 0.2 + rng.normal(0, 1, 400), neg)
    assert d > 0 and 0 <= p < 0.05


def test_mcnemar():
    ca = np.array([True] * 30 + [False] * 10 + [True] * 60)
    cb = np.array([False] * 30 + [True] * 10 + [True] * 60)
    b, c, chi2, p, p_exact = S.mcnemar(ca, cb)
    assert (b, c) == (30, 10)
    assert chi2 == (abs(30 - 10) - 1) ** 2 / 40
    from scipy import stats

    assert p == stats.chi2.sf(chi2, 1)
    assert p_exact == stats.binomtest(30, 40, 0.5).pvalue
    assert S.mcnemar(ca, ca) == (0, 0, 0.0, 1.0, 1.0)


def test_bh_fdr():
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.01])
    got = S.bh_fdr(p)
    # manual BH
    order = np.argsort(p, kind="mergesort")
    adj = p[order] * p.size / np.arange(1, p.size + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    exp = np.empty_like(adj)
    exp[order] = np.minimum(adj, 1)
    np.testing.assert_array_equal(got, exp)
    assert got[0] == got[4]  # ties share the adjusted value


def test_correct_family_per_cell_vs_per_model():
    rng = np.random.default_rng(2)
    rows = []
    for res in (112, 224):
        for _ in range(10):
            rows.append(
                {
                    "dataset": "d",
                    "model": "m",
                    "res": res,
                    "budget": 1024,
                    "mcnemar_p": float(rng.random() * 0.1),
                }
            )
    df = pd.DataFrame(rows)
    cell = S.correct_family(df)
    pooled = S.correct_family(df, ("dataset", "model"))
    for res in (112, 224):
        sub = df[df.res == res]
        np.testing.assert_array_equal(
            cell.loc[sub.index, "p_adj"].to_numpy(), S.bh_fdr(sub.mcnemar_p.to_numpy())
        )
    np.testing.assert_array_equal(
        pooled.p_adj.to_numpy(), S.bh_fdr(df.mcnemar_p.to_numpy())
    )
    assert (cell.significant == (cell.p_adj < 0.05)).all()


def test_finalize_cell_uses_common_valid_trials():
    rng = np.random.default_rng(3)
    n_pos, n_neg = 200, 2000
    raws = {}
    for name, shift in (("a", 2.0), ("b", 1.5), ("aligned", 3.0)):
        pos = rng.normal(shift, 1, n_pos).astype(np.float32)
        neg = rng.normal(0, 1, n_neg).astype(np.float32)
        if name == "b":
            pos[:5] = np.nan
            neg[:7] = np.nan
        valid = np.concatenate([~np.isnan(pos), ~np.isnan(neg)])
        raws[name] = {"pos": pos, "neg": neg, "valid": valid}
    fin = S.finalize_cell(raws)
    assert {k: v["correct"].size for k, v in fin.items()} == {
        "a": n_pos + n_neg - 12,
        "b": n_pos + n_neg - 12,
        "aligned": n_pos + n_neg - 12,
    }
    rows = S.cell_rows("d", "m", 112, 1024, fin)
    assert [(r["codec_a"], r["codec_b"]) for r in rows] == [
        ("a", "aligned"),
        ("a", "b"),
        ("aligned", "b"),
    ]
    frame = S.significance_frame(rows)
    assert list(frame.columns) == list(S.SIGNIFICANCE_COLUMNS)


def test_posthoc_helpers():
    rng = np.random.default_rng(4)
    a = np.round(rng.random(14), 3)
    b = np.round(rng.random(14), 3)
    b[3] = a[3]
    assert S.cliffs_delta(a, b) == R.cliffs_delta_loop(a, b)
    assert S.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    M = pd.DataFrame(rng.random((14, 5)), columns=list("vwxyz"))
    ph = S.posthoc(M, pairs=[("v", "w"), ("v", "missing")])
    assert ph["n"] == 14 and ph["k"] == 5 and len(ph["pairs"]) == 1
    assert ph["kendall_w"] == pytest.approx(ph["friedman_chi2"] / (14 * 4))


def test_eer_matrix_complete_matchers_only():
    rows = []
    for m in ("m1", "m2", "edgeface_x"):
        for c in ("webp", "avif"):
            if m == "m2" and c == "avif":
                continue
            rows.append(
                {
                    "dataset": "colorferet",
                    "res": 112,
                    "budget": 1024,
                    "kind": "compressed",
                    "suffix": np.nan,
                    "model": m,
                    "codec": c,
                    "eer": 0.1,
                }
            )
    met = pd.DataFrame(rows)
    M = S.eer_matrix(met, codecs=("webp", "avif", "jpeg"))
    assert list(M.index) == ["edgeface_x", "m1"] and list(M.columns) == ["webp", "avif"]
    M = S.eer_matrix(met, codecs=("webp", "avif"), exclude_prefix="edgeface")
    assert list(M.index) == ["m1"]


def test_eer_matrix_skips_all_nan_codec_and_raises_when_empty():
    rows = [
        {
            "dataset": "kk",
            "res": 224,
            "budget": 1024,
            "kind": "compressed",
            "suffix": "",
            "model": m,
            "codec": c,
            "eer": np.nan if c == "neural_bmshj2018" else 0.1,
        }
        for m in ("m1", "m2")
        for c in ("webp", "avif", "neural_bmshj2018")
    ]
    met = pd.DataFrame(rows)
    M = S.eer_matrix(met, dataset="kk", res=224, budget=1024)
    assert list(M.index) == ["m1", "m2"] and list(M.columns) == ["webp", "avif"]
    with pytest.raises(ValueError, match="no matcher"):
        S.eer_matrix(met, dataset="kk", res=112, budget=1024)
    met.loc[met.codec != "neural_bmshj2018", "eer"] = np.nan
    with pytest.raises(ValueError, match="no matcher"):
        S.eer_matrix(met, dataset="kk", res=224, budget=1024)


def test_significance_frame_empty():
    df = S.significance_frame([])
    assert df.empty and list(df.columns) == list(S.SIGNIFICANCE_COLUMNS)
