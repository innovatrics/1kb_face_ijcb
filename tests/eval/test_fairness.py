# SPDX-License-Identifier: MIT
"""Tests of face1kb.eval.fairness (CPU, synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face1kb.eval import fairness as F
from face1kb.eval import verification as V


def test_pose_class():
    assert F.pose_class("frontal image") == "frontal"
    assert F.pose_class("subject faces 15 degree left") == "frontal"
    assert F.pose_class("quarter left - subject faces 67.5 degrees left") == "quarter"
    assert F.pose_class("45 degree right") == "quarter"
    assert F.pose_class("half left") == "half"
    assert F.pose_class("profile right") == "profile"
    assert F.pose_class("75 degree left") == "profile"
    assert F.pose_class(np.nan) is None and F.pose_class("unknown") is None


def test_age_band_edges():
    b = F.age_band(pd.Series([0, 25, 26, 35, 50, 51, 65, 66, 100]))
    assert b.astype(str).tolist() == [
        "<=25",
        "<=25",
        "26-35",
        "26-35",
        "36-50",
        "51-65",
        "51-65",
        ">65",
        ">65",
    ]


def _toy(rng, n_img=60):
    """Index with 6 subjects x 10 images and every mated / non-mated pair."""
    subj = np.repeat(np.arange(6), n_img // 6)
    group = np.array(["A", "A", "B", "B", "C", None], dtype=object)[subj]
    i1, i2 = np.triu_indices(n_img, 1)
    lab = (subj[i1] == subj[i2]).astype(np.int8)
    emb = rng.normal(size=(6, 32))[subj] + 0.8 * rng.normal(size=(n_img, 32))
    return subj, group, i1, i2, lab, emb.astype(np.float32)


def test_subgroup_rows_both_ends_rule():
    rng = np.random.default_rng(0)
    subj, group, i1, i2, lab, emb = _toy(rng)
    (p1, p2), (n1, n2) = V.sample_pairs(i1, i2, lab, 0)
    pos = V.cosine_scores(emb, p1, p2)
    neg = V.cosine_scores(emb, n1, n2)
    rows = F.subgroup_rows("g", group, (p1, p2), (n1, n2), pos, neg, min_mated=10)
    by = {r["subgroup"]: r for r in rows}
    assert set(by) == {F.OVERALL, "A", "B"}  # C has one subject -> no impostor pair
    # A: 2 subjects x C(10,2) mated, 10 x 10 cross-subject impostors
    assert by["A"]["n_pos"] == 90 and by["A"]["n_neg"] == 100
    m = (group[p1] == "A") & (group[p2] == "A")
    k = (group[n1] == "A") & (group[n2] == "A")
    assert by["A"]["eer"] == V.eer(pos[m], neg[k])
    # overall: both ends share a non-null value (None never forms a group)
    assert by[F.OVERALL]["n_pos"] == 5 * 45 and by[F.OVERALL]["n_neg"] == 200
    d = F.disparity_row(rows, min_neg=100)
    assert d["n_subgroups"] == 2 and d["eer_max"] >= d["eer_min"]
    assert F.disparity_row(rows, min_neg=101) is None


def test_fmr_rows():
    rng = np.random.default_rng(1)
    subj, group, i1, i2, lab, emb = _toy(rng)
    (_, _), (n1, n2) = V.sample_pairs(i1, i2, lab, 0)
    neg = V.cosine_scores(emb, n1, n2)
    rows = F.fmr_rows("g", group, (n1, n2), neg, 0.1, min_neg=50)
    ov = [r for r in rows if r["subgroup"] == F.OVERALL][0]
    same = (group[n1] == group[n2]) & np.not_equal(group[n1], None)
    tau = float(V.linear_quantile(neg[same], 0.9))
    assert ov["tau"] == tau and ov["n_neg"] == int(same.sum())
    sub = [r for r in rows if r["subgroup"] != F.OVERALL]
    assert ov["fmr_disparity_pp"] == pytest.approx(
        (max(r["fmr"] for r in sub) - min(r["fmr"] for r in sub)) * 100
    )
    assert F.fmr_rows("g", group, (n1, n2), neg, 0.1, min_neg=10**6) == []
    # a NaN score in the pool: tau NaN and every FMR 0 (published), or dropped
    nan = neg.copy()
    nan[np.flatnonzero(same)[0]] = np.nan
    prop = F.fmr_rows("g", group, (n1, n2), nan, 0.1, min_neg=50)
    assert all(np.isnan(r["tau"]) and r["fmr"] == 0.0 for r in prop)
    omit = F.fmr_rows("g", group, (n1, n2), nan, 0.1, min_neg=50, nan_policy="omit")
    ov = [r for r in omit if r["subgroup"] == F.OVERALL][0]
    assert ov["n_neg"] == int(same.sum()) - 1
    assert ov["tau"] == float(V.linear_quantile(np.delete(neg[same], 0), 0.9))
    with pytest.raises(ValueError):
        F.fmr_rows("g", group, (n1, n2), neg, 0.1, nan_policy="drop")


def test_uniform_disparity_and_masks():
    rng = np.random.default_rng(2)
    pos = rng.normal(2, 1, 400).astype(np.float32)
    neg = rng.normal(0, 1, 4000).astype(np.float32)
    pg = [np.arange(400) % 2 == 0, np.arange(400) % 2 == 1]
    ng = [np.arange(4000) % 2 == 0, np.arange(4000) % 2 == 1]
    allp, alln = np.ones(400, bool), np.ones(4000, bool)
    e = [V.eer(pos[m], neg[k]) for m, k in zip(pg, ng)]
    assert F.uniform_disparity(pos, neg, pg, ng, allp, alln) == (max(e) - min(e)) * 100
    few = np.zeros(400, bool)
    few[:30] = True  # 15 mated per group < 20
    assert np.isnan(F.uniform_disparity(pos, neg, pg, ng, few, alln))
    masks = F.cluster_bootstrap_masks(10, 3, np.random.default_rng(0))
    ref = np.random.default_rng(0)
    for m in masks:
        exp = np.zeros(10, bool)
        exp[ref.choice(10, size=10, replace=True)] = True
        assert np.array_equal(m, exp)


def test_attributes_colorferet_join_on_stem():
    index = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "rel_path": [
                "00001/00001_930831_fa_a.png",
                "00001/00001_930831_pr_a.png",
                "00002/00002_940128_hl.png",
            ],
            "subject": [1, 1, 2],
        }
    )
    labels = pd.DataFrame(
        {
            "image": ["00002_940128_hl", "00001_930831_fa_a", "00001_930831_pr_a", "x"],
            "pose": ["half left", "frontal image", "profile right", "frontal image"],
            "gender": ["female", "male", "male", "male"],
            "race": ["Asian", "White", "White", np.nan],
            "age_from": [30, 50, 50, 20],
            "age_to": [30, 50, 50, 20],
        }
    )
    a = F.attributes_colorferet(index, labels)
    assert a["pose"].tolist() == ["frontal", "profile", "half"]
    assert a["skin_tone"].tolist() == ["White", "White", "Asian"]
    assert a["age"].tolist() == ["36-50", "36-50", "26-35"]
    b = F.attributes_colorferet(
        index, labels.drop(columns=["age_to"]).rename(columns={"age_from": "age"})
    )
    assert b["age"].tolist() == a["age"].tolist()


def test_attributes_kk_propagated_per_identity():
    index = pd.DataFrame({"subject": ["p1", "p1", "p2", "p3"]})
    att = pd.DataFrame(
        {
            "identity": ["p1", "p1", "p1", "p2", "p2"],
            "mst_label": [np.nan, "MST6", "MST7", "MST5", np.nan],
            "gender": ["F", "M", "M", "F", np.nan],
            "age": [20, 30, 40, np.nan, 70],
        }
    )
    a = F.attributes_kk(index, att)
    assert a["skin_tone"][:3].tolist() == ["MST6", "MST6", "MST5"]
    assert pd.isna(a["skin_tone"][3])
    assert a["gender"][:3].tolist() == ["M", "M", "F"]
    assert a["age"][:3].tolist() == ["26-35", "26-35", ">65"]


def test_table_helpers():
    fair = pd.DataFrame(
        {
            "res": 112,
            "attribute": "skin_tone",
            "model": "m",
            "codec": "webp",
            "budget": 1024,
            "subgroup": ["MST5", "MST6", "MST8"],
            "eer": [0.01, 0.03, 0.2],
        }
    )
    assert F.subgroup_spread_pp(
        fair, model="m", codec="webp", budget=1024
    ) == pytest.approx(2.0)
    assert np.isnan(F.subgroup_spread_pp(fair, model="m", codec="avif", budget=1024))
    disp = pd.DataFrame(
        {
            "res": [112],
            "attribute": ["pose"],
            "codec": ["aligned"],
            "model": ["m"],
            "budget": [0],
            "eer_min": [0.0],
            "eer_max": [0.004],
            "n_subgroups": [4],
        }
    )
    v, n = F.disparity_pp(
        disp, model="m", codec="aligned", budget=512, attribute="pose"
    )
    assert v == pytest.approx(0.4) and n == 4
