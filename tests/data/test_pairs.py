# SPDX-License-Identifier: MIT
"""Verification-pair matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face1kb.data import pairs as pr


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 50])
def test_pair_indices_match_triu(n, monkeypatch):
    monkeypatch.setattr(pr, "_BLOCK_ROWS", 5)  # force many blocks
    i, j = pr.pair_indices(n)
    ti, tj = np.triu_indices(n, k=1)
    assert i.dtype == np.int32 and j.dtype == np.int32
    assert np.array_equal(i, ti) and np.array_equal(j, tj)


def test_pair_indices_default_block():
    i, j = pr.pair_indices(300)
    ti, tj = np.triu_indices(300, k=1)
    assert np.array_equal(i, ti) and np.array_equal(j, tj)


def test_build_pairs_labels_and_counts():
    subjects = ["a", "a", "b", "c", "c", "c"]
    df = pr.build_pairs(subjects)
    assert df.dtypes.to_dict() == {
        "idx1": np.dtype("int32"),
        "idx2": np.dtype("int32"),
        "label": np.dtype("int8"),
    }
    assert len(df) == pr.n_pairs(6) == 15
    exp = [int(subjects[a] == subjects[b]) for a, b in zip(df.idx1, df.idx2)]
    assert df.label.tolist() == exp
    assert pr.pair_counts(subjects) == (15, 1 + 3)
    # subject dtype does not matter (zero-padded strings vs ints)
    assert pr.build_pairs(["00001", "00001", "00002"]).equals(pr.build_pairs([1, 1, 2]))


def test_paper_counts_closed_form():
    # 1,515,807 mated of 153,711,811 = C(17534, 2) for AI-Solutions-KK
    assert pr.n_pairs(17534) == 153_711_811
    assert pr.n_pairs(11335) == 64_235_445


def test_parquet_roundtrip(tmp_path):
    df = pr.build_pairs(["x", "y", "x", "y"])
    p = pr.write_pairs(df, tmp_path / "p.parquet")
    back = pr.read_pairs(p)
    assert back.equals(df)


def test_sample_policies():
    subjects = [s for s in "aabbbccccdd"]
    rel = [f"{s}/{k}.png" for k, s in enumerate(subjects)]
    df = pr.build_pairs(subjects)
    n_mated = int(df.label.sum())
    all_mated = pr.sample_pairs(df, rel, None, 5, seed=0)
    assert len(all_mated) == n_mated + 5
    assert all_mated.label.tolist()[:n_mated] == [1] * n_mated
    capped = pr.sample_pairs(df, rel, 3, 4, seed=0)
    assert capped.label.tolist() == [1, 1, 1, 0, 0, 0, 0]
    again = pr.sample_pairs(df, rel, 3, 4, seed=0)
    assert capped.equals(again)
    assert set(capped.columns) == {"name1", "name2", "label"}
    assert pr.SAMPLE_POLICIES == {
        "colorferet": (None, 200_000),
        "kk": (200_000, 200_000),
    }
    # the sample draws with numpy's default_rng exactly like a direct call
    rng = np.random.default_rng(0)
    pos = rng.choice(np.flatnonzero(df.label == 1), size=3, replace=False)
    assert (
        pd.Series(rel).iloc[df.idx1.to_numpy()[pos]].tolist()
        == capped.name1[:3].tolist()
    )
