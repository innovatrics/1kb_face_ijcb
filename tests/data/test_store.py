# SPDX-License-Identifier: MIT
"""Checks against an existing data root (skipped without FACE1KB_DATA_ROOT).

With the released crops these confirm that the index and pairs files on disk are
the ones the builders produce, and that the integer-ratio resolutions are exact
subsamplings of the 224 px crops.
"""

from __future__ import annotations

import numpy as np
import pytest

from face1kb import config
from face1kb.data import build_index, build_pairs, read_pairs

pytestmark = pytest.mark.data


def _stem(paths):
    return [p.rsplit(".", 1)[0] for p in paths]


@pytest.mark.parametrize("dataset", config.DATASETS)
def test_index_matches_scan(dataset):
    root = config.aligned_dir(dataset, 112)
    if not root.is_dir() or not config.index_csv(dataset).exists():
        pytest.skip(f"no aligned_112 / index for {dataset}")
    built = build_index(root, dataset=dataset)
    stored = config.read_index(dataset)
    assert len(built) == len(stored)
    assert _stem(built.rel_path) == _stem(stored.rel_path.astype(str))
    b = built.subject.astype(str).str.lstrip("0")
    s = stored.subject.astype(str).str.lstrip("0")
    assert (b.to_numpy() == s.to_numpy()).all()
    if "pose" in stored.columns:
        assert (built.pose.to_numpy() == stored.pose.to_numpy()).all()


@pytest.mark.parametrize("dataset", config.DATASETS)
def test_pairs_match_index(dataset):
    if not config.pairs_parquet(dataset).exists():
        pytest.skip(f"no pairs for {dataset}")
    stored = read_pairs(config.pairs_parquet(dataset))
    built = build_pairs(config.read_index(dataset).subject)
    for col in ("idx1", "idx2", "label"):
        assert np.array_equal(stored[col].to_numpy(), built[col].to_numpy())


@pytest.mark.parametrize("dataset", config.DATASETS)
def test_112_is_224_subsampled(dataset):
    cv2 = pytest.importorskip("cv2")
    d112, d224 = config.aligned_dir(dataset, 112), config.aligned_dir(dataset, 224)
    if not d112.is_dir() or not d224.is_dir():
        pytest.skip("need aligned_112 and aligned_224")
    rel = config.read_index(dataset).rel_path.tolist()[::997][:12]
    for rp in rel:
        a = cv2.imread(str(d224 / rp))
        b = cv2.imread(str(d112 / rp))
        assert np.array_equal(a[::2, ::2], b), rp
