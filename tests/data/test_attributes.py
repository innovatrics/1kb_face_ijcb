# SPDX-License-Identifier: MIT
"""Per-identity crop selection of the attribute estimator (no models needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face1kb.data import attributes as at
from face1kb.data.crops import MissingCropsError

cv2 = pytest.importorskip("cv2")


def _index(counts):
    rel = [f"{s}/{s}{k:02d}.png" for s, n in counts for k in range(n)]
    return pd.DataFrame(
        {
            "id": range(len(rel)),
            "rel_path": rel,
            "subject": [r.split("/")[0] for r in rel],
        }
    )


def _write_crops(root, index):
    for k, rp in enumerate(index.rel_path):
        p = root / rp
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), np.full((112, 112, 3), k, np.uint8))


class _FakeGenderAge:
    """Age = the grey level of the crop (its index position); fails on level 3."""

    def __init__(self):
        self.calls = 0

    def __call__(self, bgr):
        self.calls += 1
        level = int(bgr[0, 0, 0])
        if level == 3:
            raise RuntimeError("no face")
        return level, "M" if level % 2 else "F"


def test_first_crops_per_identity_in_index_order(tmp_path, monkeypatch):
    # identities listed out of order; "b" has more crops than per_subject_ga
    index = _index([("b", 10), ("a", 2)])
    _write_crops(tmp_path / "112", index)
    mst_calls = []

    def fake_mst(path):
        mst_calls.append(str(path))
        return "MST3", 3, "#ABCDEF"

    monkeypatch.setattr(at, "monk_skin_tone", fake_mst)
    ga = _FakeGenderAge()
    df = at.estimate_attributes(
        index, tmp_path / "112", tmp_path / "224", per_subject_ga=8, genderage=ga
    )
    assert list(df.columns) == list(at.COLUMNS)
    # identities sorted, then the first 8 crops of each in index order
    expected = index.rel_path[index.subject == "a"].tolist()
    expected += index.rel_path[index.subject == "b"].tolist()[:8]
    assert df.rel_path.tolist() == expected
    assert df.image.tolist() == [p.split("/")[1] for p in expected]
    assert ga.calls == 10
    # MST only on the first crop of each identity, in identity order
    assert mst_calls == [
        str(tmp_path / "224" / expected[0]),
        str(tmp_path / "224" / expected[2]),
    ]
    assert df.mst_label.tolist() == ["MST3", "", "MST3"] + [""] * 7
    # genderage failure keeps the row with empty age and gender
    failed = df[df.rel_path == "b/b03.png"].iloc[0]
    assert failed.age == "" and failed.gender == ""
    assert df[df.rel_path == "b/b02.png"].iloc[0].age == 2


def test_subset_no_mst_and_errors(tmp_path):
    index = _index([("a", 3), ("b", 3)])
    _write_crops(tmp_path / "112", index)
    df = at.estimate_attributes(
        index, tmp_path / "112", None, subjects=["b"], genderage=_FakeGenderAge()
    )
    assert df.subject.tolist() == ["b"] * 3
    assert (df.mst_label == "").all() and (df.skin_hex == "").all()
    with pytest.raises(ValueError, match="not in the index"):
        at.estimate_attributes(
            index, tmp_path / "112", None, subjects=["zz"], genderage=_FakeGenderAge()
        )
    # an unreadable crop is an error unless strict=False, which skips it
    (tmp_path / "112" / "a" / "a01.png").unlink()
    with pytest.raises(MissingCropsError, match="1 of 6"):
        at.estimate_attributes(
            index, tmp_path / "112", None, genderage=_FakeGenderAge()
        )
    loose = at.estimate_attributes(
        index, tmp_path / "112", None, genderage=_FakeGenderAge(), strict=False
    )
    assert len(loose) == 5 and "a/a01.png" not in loose.rel_path.tolist()
    # nothing readable: no table
    with pytest.raises(ValueError, match="no attributes"):
        at.estimate_attributes(
            index, tmp_path / "missing", None, genderage=_FakeGenderAge(), strict=False
        )


def test_attributes_roundtrip(tmp_path):
    index = _index([("a", 2)])
    _write_crops(tmp_path / "112", index)
    df = at.estimate_attributes(
        index, tmp_path / "112", None, genderage=_FakeGenderAge()
    )
    back = at.read_attributes(at.write_attributes(df, tmp_path / "attributes.csv"))
    assert back.rel_path.tolist() == df.rel_path.tolist()
    assert back.age.tolist() == [0, 1]


def test_genderage_device_selection(monkeypatch, tmp_path):
    zoo = pytest.importorskip("insightface.model_zoo")
    seen = []

    class _Model:
        def prepare(self, ctx_id):
            seen.append(("prepare", ctx_id))

    def fake_get_model(path, providers=None, provider_options=None):
        seen.append((providers, provider_options))
        return _Model()

    monkeypatch.setattr(zoo, "get_model", fake_get_model)
    at.GenderAge(model_path=tmp_path / "genderage.onnx")
    assert seen[:2] == [(["CPUExecutionProvider"], None), ("prepare", -1)]
    seen.clear()
    at.GenderAge(model_path=tmp_path / "genderage.onnx", ctx_id=2)
    assert seen[0] == (
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        [{"device_id": "2"}, {}],
    )
