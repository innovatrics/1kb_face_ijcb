# SPDX-License-Identifier: MIT
"""Unit and differential tests of face1kb.eval.verification (CPU, synthetic data)."""

from __future__ import annotations

import hashlib
import io
import os

import numpy as np
import pandas as pd
import pytest

from face1kb import config
from face1kb.eval import verification as V

from . import _reference as R


def _scores(rng, n_pos, n_neg, sep=2.5, dtype=np.float32):
    pos = rng.normal(sep, 1.0, n_pos).astype(dtype)
    neg = rng.normal(0.0, 1.0, n_neg).astype(dtype)
    return pos, neg


def _same_ci(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        np.array_equal(a[k], b[k], equal_nan=True) for k in a
    )


def _same(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, tuple):
            if x != y:
                return False
        elif not (x == y or (np.isnan(x) and np.isnan(y))):
            return False
    return True


@pytest.mark.parametrize("seed", range(6))
def test_metrics_match_reference(seed):
    rng = np.random.default_rng(seed)
    n_neg = [50, 3375, 20_000, 120_000, 7, 1][seed]
    pos, neg = _scores(rng, 800 + 97 * seed, n_neg)
    pos[::17] = np.nan  # NaN rows are dropped
    assert _same(V.verification_metrics(pos, neg), R.metrics(pos, neg))
    assert V.eer(pos, neg) == R.eer_only(pos, neg)
    clean_p, clean_n = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    assert V.eer_threshold(clean_p, clean_n) == R.eer_threshold(clean_p, clean_n)


def test_fnmr_nan_rule_and_realised_fmr():
    rng = np.random.default_rng(1)
    pos, neg = _scores(rng, 1000, 3375)
    m = V.verification_metrics(pos, neg)
    # m = round(1e-4 * 3375) = 0 < 10 -> not estimable
    assert np.isnan(m["fnmr_0.0001"]) and np.isnan(m["fmr_realised_0.0001"])
    # m = round(1e-2 * 3375) = 34 -> threshold is the 34th largest impostor score
    assert m["fmr_realised_0.01"] == 34 / 3375
    t = np.sort(neg)[3375 - 34]
    assert m["fnmr_0.01"] == np.mean(pos < t)


def test_empty_side_gives_nan():
    m = V.verification_metrics(np.array([np.nan], np.float32), np.ones(5, np.float32))
    assert m["n_pos"] == 0 and np.isnan(m["eer"]) and np.isnan(m["fnmr_0.001"])
    assert "fmr_realised_0.001" not in m
    assert np.isnan(V.eer(np.array([], np.float32), np.ones(3, np.float32)))


def test_perfect_separation():
    pos = np.linspace(0.6, 0.9, 500, dtype=np.float32)
    neg = np.linspace(-0.2, 0.3, 20_000, dtype=np.float32)
    m = V.verification_metrics(pos, neg)
    assert m["eer"] == 0.0 and m["fnmr_0.001"] == 0.0
    assert np.isnan(m["fnmr_0.0001"])  # round(1e-4 * 20000) = 2 < 10


_NUMPY_VERSION = tuple(int(v) for v in np.__version__.split(".")[:2])
# np.linspace has the float32 semantics of the published run in every NumPy 2
# release; np.quantile only from 2.4 (2.0-2.3 form the virtual index in float32)
_NUMPY2 = _NUMPY_VERSION >= (2, 0)
_NUMPY24 = _NUMPY_VERSION >= (2, 4)


@pytest.mark.parametrize(
    "lo, hi, digest",
    [
        # sha256 prefixes of np.linspace(float32, float32, 4000) under NumPy 2; the
        # first and third grids differ from a float64 grid in 2042 / 2868 points
        (-0.2731, 0.9412, "b3a9d199918a6316"),
        (0.1, 0.1, "824c1b28fe0a599f"),
        (-1.0, 1.0, "16790095b2132acc"),
        (0.3333, 0.3334, "1bf89c0a85fcc4e6"),
    ],
)
def test_eer_grid_golden(lo, hi, digest):
    g = V.eer_grid(np.float32(lo), np.float32(hi))
    assert g.dtype == np.float32 and g.size == V.EER_GRID_POINTS
    assert hashlib.sha256(g.tobytes()).hexdigest()[:16] == digest


@pytest.mark.skipif(not _NUMPY2, reason="the reference is NumPy 2's np.linspace")
def test_eer_grid_matches_numpy2_linspace():
    rng = np.random.default_rng(0)
    for i in range(2000):
        lo, hi = np.sort(rng.uniform(-1, 1, 2)).astype(np.float32)
        if i % 5 == 0:
            hi = lo
        elif i % 5 == 1:
            hi = np.nextafter(lo, np.float32(2))
        num = 4000 if i % 3 else int(rng.integers(2, 5000))
        assert np.array_equal(V.eer_grid(lo, hi, num), np.linspace(lo, hi, num))
    tiny = (np.float32(1e-40), np.float32(3e-40))  # underflowing step
    assert np.array_equal(V.eer_grid(*tiny), np.linspace(*tiny, 4000))


def test_eer_grid_other_dtypes_fall_back():
    a, b = np.float64(0.1), np.float32(0.7)
    np.testing.assert_array_equal(V.eer_grid(a, b), np.linspace(a, b, 4000))


def _quantile_input():
    # exact integer arithmetic, identical on every platform; contains ties
    return ((np.arange(20011) * 7919) % 10007 / 10007 * 1.6 - 0.8).astype(np.float32)


@pytest.mark.parametrize(
    "q, bits",
    # float32 bit patterns of np.quantile(x, q) under NumPy 2.4; for 0.25 and 0.123
    # the float64 interpolation of NumPy 1.x gives a different value
    [(0.99, 0x3F48AA7B), (0.999, 0x3F4C598A), (0.25, 0xBECCD20A), (0.123, 0xBF1A71ED)],
)
def test_linear_quantile_golden(q, bits):
    t = np.asarray(V.linear_quantile(_quantile_input(), q))
    assert t.dtype == np.float32 and int(t.view(np.uint32)) == bits


@pytest.mark.skipif(not _NUMPY24, reason="the reference is NumPy >= 2.4 np.quantile")
def test_linear_quantile_matches_numpy24_quantile():
    rng = np.random.default_rng(1)
    for i in range(3000):
        n = int(rng.integers(1, 3000))
        x = rng.uniform(-1, 1, n).astype(np.float32)
        if i % 3 == 1:
            x = np.round(x, 2).astype(np.float32)
        if i % 97 == 0:
            x[int(rng.integers(0, n))] = np.nan
        for q in (0.99, 0.999, float(rng.uniform()), 0.0, 1.0):
            a, b = V.linear_quantile(x, q), np.quantile(x, q)
            assert np.asarray(a).dtype == np.asarray(b).dtype
            assert a == b or (np.isnan(a) and np.isnan(b))


def test_metrics_frame_empty():
    df = V.metrics_frame([])
    assert df.empty and list(df.columns) == list(V.METRICS_COLUMNS)


@pytest.mark.filterwarnings("ignore:All-NaN slice")
@pytest.mark.parametrize("level", ["subject", "pair"])
def test_bootstrap_matches_reference(level):
    rng = np.random.default_rng(7)
    pos, neg = _scores(rng, 900, 30_000)
    pos[5] = np.nan
    neg[11] = np.nan
    ps = rng.integers(0, 40, pos.size)
    ns = rng.integers(0, 40, neg.size)
    kw = dict(pos_subj=ps, neg_subj=ns, level=level)
    got = V.bootstrap_ci(pos, neg, V.FMR_TARGETS, 25, 10_000, 0, **kw)
    ref = R.bootstrap(pos, neg, V.FMR_TARGETS, 25, 10_000, 0, **kw)
    assert _same_ci(got, ref)
    assert set(got) == {"eer", "fnmr_0.01", "fnmr_0.001", "fnmr_0.0001"}
    assert V.bootstrap_ci(pos, neg, reps=0) == {}


@pytest.mark.filterwarnings("ignore:All-NaN slice")
def test_bootstrap_subject_labels_any_type():
    # string and integer subject labels with the same sort order give the same CIs
    rng = np.random.default_rng(3)
    pos, neg = _scores(rng, 300, 5000)
    ps = rng.integers(1, 30, pos.size)
    ns = rng.integers(1, 30, neg.size)
    a = V.bootstrap_ci(pos, neg, reps=10, pos_subj=ps, neg_subj=ns)
    b = V.bootstrap_ci(
        pos,
        neg,
        reps=10,
        pos_subj=np.array([f"{s:05d}" for s in ps]),
        neg_subj=np.array([f"{s:05d}" for s in ns]),
    )
    assert _same_ci(a, b)


def test_sample_pairs_is_seeded_choice():
    rng = np.random.default_rng(0)
    n = 5000
    i1 = rng.integers(0, 100, n)
    i2 = rng.integers(0, 100, n)
    lab = (rng.random(n) < 0.1).astype(np.int8)
    (p1, p2), (n1, n2) = V.sample_pairs(i1, i2, lab, 1000, seed=0)
    pos = np.flatnonzero(lab == 1)
    neg = np.random.default_rng(0).choice(
        np.flatnonzero(lab == 0), size=1000, replace=False
    )
    assert np.array_equal(p1, i1[pos]) and np.array_equal(p2, i2[pos])
    assert np.array_equal(n1, i1[neg]) and np.array_equal(n2, i2[neg])
    # no sampling when the population is not larger than the request
    (_, _), (m1, _) = V.sample_pairs(i1, i2, lab, 10**6, seed=0)
    assert np.array_equal(m1, i1[lab == 0])


def test_cosine_scores():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(50, 16)).astype(np.float32)
    emb[3] = np.nan
    i1 = rng.integers(0, 50, 200)
    i2 = rng.integers(0, 50, 200)
    s = V.cosine_scores(emb, i1, i2, chunk=37)
    en = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    ref = (en[i1] * en[i2]).sum(axis=1)
    assert s.dtype == np.float32
    np.testing.assert_array_equal(s, ref)
    assert np.isnan(s[(i1 == 3) | (i2 == 3)]).all()


def test_paired_cosines_drops_nan_rows():
    a = np.array([[1, 0], [0, 1], [np.nan, np.nan]], np.float32)
    b = np.array([[1, 0], [1, 0], [1, 0]], np.float32)
    np.testing.assert_allclose(V.paired_cosines(a, b), [1.0, 0.0])


def test_attach_delta_and_metrics_frame():
    rows = [
        {
            "dataset": "d",
            "model": "m",
            "tag": "aligned_112",
            "kind": "aligned",
            "res": 112,
            "suffix": "",
            "codec": "",
            "budget": 0,
            "eer": 0.01,
        },
        {
            "dataset": "d",
            "model": "m",
            "tag": "aligned_112_tight",
            "kind": "aligned",
            "res": 112,
            "suffix": "_tight",
            "codec": "",
            "budget": 0,
            "eer": 0.5,
        },
        {
            "dataset": "d",
            "model": "m",
            "tag": "webp_112_1024",
            "kind": "compressed",
            "res": 112,
            "suffix": "",
            "codec": "webp",
            "budget": 1024,
            "eer": 0.03,
        },
    ]
    df = V.metrics_frame(rows)
    assert list(df.columns) == list(V.METRICS_COLUMNS)
    assert df.eer_aligned.tolist() == [0.01, 0.01, 0.01]
    np.testing.assert_allclose(df.delta_eer, [0.0, 0.49, 0.02])


@pytest.mark.data
def test_recompute_published_cell():
    """Recompute one clean Color FERET cell from the embeddings (point estimates).

    The published ``metrics.csv`` went through one read/write with pandas' default
    float parser (the per-dataset merge), so the recomputed row is passed through
    the same round trip and must then match exactly.
    """
    candidates = [
        config.results_dir("accuracy", "metrics.csv"),
        config.output_dir("accuracy", "metrics.csv"),
    ]
    shipped = next((p for p in candidates if p.is_file()), None)
    emb = config.embeddings_path("colorferet", "arcface_antelopev2", "aligned_112")
    if (
        shipped is None
        or not emb.is_file()
        or not config.pairs_parquet("colorferet").is_file()
    ):
        pytest.skip("needs accuracy/metrics.csv, pairs and embeddings")
    if os.environ.get("FACE1KB_SLOW", "") != "1":
        pytest.skip("set FACE1KB_SLOW=1 (reads the 64M-pair table)")
    rows = V.accuracy_rows(
        "colorferet", ["arcface_antelopev2"], bootstrap=0, tags=["aligned_112"]
    )
    new = pd.read_csv(io.StringIO(V.metrics_frame(rows).to_csv(index=False)))
    exp = pd.read_csv(shipped)
    exp = exp[(exp.dataset == "colorferet") & (exp.model == "arcface_antelopev2")]
    exp = exp[exp.tag == "aligned_112"].iloc[0]
    for k in ("n_pos", "n_neg", "eer", "fnmr_0.01", "fnmr_0.001", "fnmr_0.0001"):
        assert new.iloc[0][k] == exp[k], k


@pytest.mark.parametrize("cpu_fallback", [True, False])
def test_cuda_scorer_cpu_fallback(monkeypatch, cpu_fallback):
    torch = pytest.importorskip("torch")

    def oom(*_a, **_k):
        raise torch.OutOfMemoryError("synthetic OOM")

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(20, 512)).astype(np.float32)
    i1, i2 = rng.integers(0, 20, 50), rng.integers(0, 20, 50)
    monkeypatch.setattr(V, "_CHUNK_OK", {})
    monkeypatch.setattr(torch, "from_numpy", oom)
    score = V.make_scorer("cpu", cpu_fallback=cpu_fallback)
    if cpu_fallback:
        np.testing.assert_array_equal(score(emb, i1, i2), V.cosine_scores(emb, i1, i2))
    else:
        with pytest.raises(RuntimeError, match="cpu_fallback=False"):
            score(emb, i1, i2)
