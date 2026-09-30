# SPDX-License-Identifier: MIT
"""End-to-end runs of the eval drivers on a tiny synthetic dataset (public layout).

The drivers (``accuracy_rows``, ``significance_rows``, ``fairness_rows``,
``fmr_fairness_rows``, ``disparity_ci_rows``) read the index, the pair table, the
labels / attributes and the embedding arrays through ``face1kb.config``; their
results are compared with the reference formulas applied directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face1kb import config
from face1kb.eval import fairness as F
from face1kb.eval import significance as S
from face1kb.eval import verification as V

from . import _reference as R

MODEL = "arcface_antelopev2"
TAGS = {"aligned_112": 0.5, "ours_fast_112_1024": 0.8, "webp_112_1024": 1.0}
N_SUBJ, PER_SUBJ = 12, 6
SAMPLE = 1000


def _write_dataset(dataset: str, rng) -> np.ndarray:
    subj = np.repeat(np.arange(1, N_SUBJ + 1), PER_SUBJ)
    shot = np.tile(np.arange(PER_SUBJ), N_SUBJ)
    rel = [f"{s:05d}/{s:05d}_{k}.png" for s, k in zip(subj, shot)]
    config.dataset_dir(dataset).mkdir(parents=True)
    pd.DataFrame({"id": np.arange(subj.size), "rel_path": rel, "subject": subj}).to_csv(
        config.index_csv(dataset), index=False
    )
    i1, i2 = np.triu_indices(subj.size, 1)
    pd.DataFrame(
        {
            "idx1": i1.astype(np.int32),
            "idx2": i2.astype(np.int32),
            "label": (subj[i1] == subj[i2]).astype(np.int8),
        }
    ).to_parquet(config.pairs_parquet(dataset), index=False)
    centers = rng.normal(size=(N_SUBJ, 512))
    config.embeddings_dir(dataset, MODEL).mkdir(parents=True)
    for tag, noise in TAGS.items():
        e = (centers[subj - 1] + noise * rng.normal(size=(subj.size, 512))).astype(
            np.float32
        )
        if tag == "webp_112_1024":
            e[3] = np.nan  # one crop that failed to embed
        np.save(config.embeddings_path(dataset, MODEL, tag), e)
    return subj


@pytest.fixture
def synth(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(config, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(config, "LAYOUT", "public")
    rng = np.random.default_rng(0)
    subj = {ds: _write_dataset(ds, rng) for ds in ("colorferet", "kk")}
    # Color FERET labels: two races, two genders, frontal / profile
    n = subj["colorferet"].size
    stems = [
        p.split("/")[1][:-4] for p in config.read_index("colorferet").rel_path.tolist()
    ]
    s = subj["colorferet"]
    pd.DataFrame(
        {
            "image": stems,
            "pose": np.where(np.arange(n) % 2 == 0, "frontal image", "profile left"),
            "gender": np.where(s % 2 == 0, "male", "female"),
            "race": np.where(s <= 6, "White", "Asian"),
            "age_from": 20 + 3 * s,
            "age_to": 20 + 3 * s,
        }
    ).to_csv(config.labels_csv("colorferet"), index=False)
    # AI-Solutions-KK attributes: one MST label per identity (first image only)
    k = subj["kk"]
    first = np.r_[True, k[1:] != k[:-1]]
    pd.DataFrame(
        {
            "subject": k,
            "mst_label": np.where(first, np.where(k <= 6, "MST5", "MST6"), None),
            "gender": np.where(k % 3 == 0, "F", "M"),
            "age": 30.0 + k,
        }
    ).to_csv(config.attributes_csv("kk"), index=False)
    return subj


def _reference_trials(dataset: str):
    pairs = pd.read_parquet(config.pairs_parquet(dataset))
    lab = pairs.label.to_numpy()
    pos = np.flatnonzero(lab == 1)
    neg = np.random.default_rng(0).choice(
        np.flatnonzero(lab == 0), size=SAMPLE, replace=False
    )
    i1, i2 = pairs.idx1.to_numpy(), pairs.idx2.to_numpy()
    return (i1[pos], i2[pos]), (i1[neg], i2[neg])


def _scores(dataset, tag, i1, i2):
    e = np.load(config.embeddings_path(dataset, MODEL, tag))
    en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    return (en[i1] * en[i2]).sum(axis=1)


@pytest.mark.filterwarnings("ignore:All-NaN slice")
def test_accuracy_rows_end_to_end(synth):
    rows = V.accuracy_rows(
        "colorferet", [MODEL], sample_nonmated=SAMPLE, bootstrap=8, boot_nonmated=600
    )
    assert [r["tag"] for r in rows] == sorted(TAGS)
    (p1, p2), (n1, n2) = _reference_trials("colorferet")
    subj = synth["colorferet"]
    for r in rows:
        pos = _scores("colorferet", r["tag"], p1, p2)
        neg = _scores("colorferet", r["tag"], n1, n2)
        exp = R.metrics(pos, neg)
        for k, v in exp.items():
            assert r[k] == v or (np.isnan(r[k]) and np.isnan(v)), (r["tag"], k)
        ci = R.bootstrap(
            pos,
            neg,
            V.FMR_TARGETS,
            8,
            600,
            0,
            pos_subj=subj[p1],
            neg_subj=subj[n1],
            level="subject",
        )
        for k, (lo, hi) in ci.items():
            assert np.array_equal(
                [r[f"{k}_lo"], r[f"{k}_hi"]], [lo, hi], equal_nan=True
            )
    by_tag = {r["tag"]: r for r in rows}
    assert by_tag["webp_112_1024"]["n_pos"] == 180 - (PER_SUBJ - 1)
    assert np.isnan(by_tag["aligned_112"]["fnmr_0.0001"])  # round(0.1) < 10
    df = V.metrics_frame(rows)
    assert list(df.columns) == list(V.METRICS_COLUMNS)
    assert (df.eer_aligned == by_tag["aligned_112"]["eer"]).all()
    assert df.codec.tolist() == ["", "ours_fast", "webp"]


def test_significance_rows_end_to_end(synth):
    rows = S.significance_rows("colorferet", [MODEL], sample_nonmated=SAMPLE)
    df = S.significance_frame(rows)
    assert list(df.columns) == list(S.SIGNIFICANCE_COLUMNS)
    assert list(zip(df.codec_a, df.codec_b)) == [
        ("aligned", "ours_fast"),
        ("aligned", "webp"),
        ("ours_fast", "webp"),
    ]
    # trials touching the NaN row of webp are removed from every source
    (p1, p2), (n1, n2) = _reference_trials("colorferet")
    touch = int(((p1 == 3) | (p2 == 3)).sum() + ((n1 == 3) | (n2 == 3)).sum())
    assert (df.n_trials == p1.size + n1.size - touch).all()
    np.testing.assert_array_equal(df.p_adj, S.bh_fdr(df.mcnemar_p.to_numpy()))


def test_row_count_mismatch_raises(synth):
    bad = np.zeros((N_SUBJ * PER_SUBJ - 1, 512), np.float32)
    np.save(config.embeddings_path("colorferet", MODEL, "jpeg_112_1024"), bad)
    with pytest.raises(ValueError, match="jpeg_112_1024"):
        V.accuracy_rows("colorferet", [MODEL], sample_nonmated=SAMPLE, bootstrap=0)
    with pytest.raises(ValueError, match="index length"):
        S.significance_rows("colorferet", [MODEL], sample_nonmated=SAMPLE)


@pytest.mark.parametrize("delta", [-1, 1])
def test_fairness_row_count_mismatch_raises(synth, delta):
    # an array with more rows than the index would otherwise be scored silently
    n = N_SUBJ * PER_SUBJ + delta
    bad = np.zeros((n, 512), np.float32)
    for ds in ("colorferet", "kk"):
        np.save(config.embeddings_path(ds, MODEL, "jpeg_112_1024"), bad)
    with pytest.raises(ValueError, match="jpeg_112_1024"):
        F.fairness_rows("colorferet", [MODEL], sample_nonmated=SAMPLE, min_mated=10)
    with pytest.raises(ValueError, match="jpeg_112_1024"):
        F.fmr_fairness_rows(
            "colorferet", [MODEL], budgets=(1024,), sample_nonmated=SAMPLE
        )
    with pytest.raises(ValueError, match="jpeg_112_1024"):
        F.disparity_ci_rows(
            "kk",
            [MODEL],
            codecs=("aligned", "jpeg"),
            subgroups=("MST5", "MST6"),
            reps=2,
            sample_nonmated=SAMPLE,
        )
    attrs = F.load_attributes("kk")
    short = {k: v[:-1] for k, v in attrs.items()}
    with pytest.raises(ValueError, match="attribute length"):
        F.disparity_ci_rows(
            "kk", [MODEL], codecs=("aligned",), attributes=short, reps=2
        )


def test_significance_rows_no_comparable_cell(synth):
    rows = S.significance_rows(
        "colorferet", [MODEL], budgets=[512], sample_nonmated=SAMPLE
    )
    assert rows == []
    assert S.significance_frame(rows).empty


def test_fairness_rows_end_to_end(synth):
    fair, disp = F.fairness_rows(
        "colorferet", [MODEL], sample_nonmated=SAMPLE, min_mated=10
    )
    fdf = pd.DataFrame(fair)
    assert set(fdf.attribute) == {"pose", "skin_tone", "gender", "age"}
    (p1, p2), (n1, n2) = _reference_trials("colorferet")
    attrs = F.load_attributes("colorferet", N_SUBJ * PER_SUBJ)
    tone = attrs["skin_tone"]
    pos = _scores("colorferet", "aligned_112", p1, p2)
    neg = _scores("colorferet", "aligned_112", n1, n2)
    pm = (tone[p1] == "Asian") & (tone[p2] == "Asian")
    nm = (tone[n1] == "Asian") & (tone[n2] == "Asian")
    row = fdf[
        (fdf.codec == "aligned")
        & (fdf.attribute == "skin_tone")
        & (fdf.subgroup == "Asian")
    ].iloc[0]
    assert (row.n_pos, row.n_neg) == (int(pm.sum()), int(nm.sum()))
    assert row.eer == R.eer_only(pos[pm], neg[nm])
    assert {d["codec"] for d in disp} <= {"aligned", "ours_fast", "webp"}

    fmr = pd.DataFrame(
        F.fmr_fairness_rows(
            "colorferet", [MODEL], budgets=(1024,), sample_nonmated=SAMPLE, min_neg=5
        )
    )
    assert set(fmr.target_fmr) == set(F.FMR_FAIRNESS_TARGETS)
    ov = fmr[(fmr.subgroup == F.OVERALL) & (fmr.attribute == "skin_tone")]
    assert (ov.n_neg == int((tone[n1] == tone[n2]).sum())).all()


def test_disparity_ci_rows_end_to_end(synth):
    rows = F.disparity_ci_rows(
        "kk",
        [MODEL],
        codecs=("aligned", "webp"),
        subgroups=("MST5", "MST6"),
        reps=12,
        sample_nonmated=SAMPLE,
    )
    assert [r["codec"] for r in rows] == ["aligned", "webp"]
    assert "ratio" not in rows[0] and rows[1]["n_boot"] > 0
    assert rows[0]["disp_lo"] <= rows[0]["disp_hi"]
    # the point value: max-min EER over the two tones (mated by identity, impostors
    # by the tone of idx1)
    (p1, p2), (n1, n2) = _reference_trials("kk")
    tone = F.load_attributes("kk")["skin_tone"]
    pos = _scores("kk", "aligned_112", p1, p2)
    neg = _scores("kk", "aligned_112", n1, n2)
    e = [R.eer_only(pos[tone[p1] == g], neg[tone[n1] == g]) for g in ("MST5", "MST6")]
    assert rows[0]["disparity_pp"] == (max(e) - min(e)) * 100
