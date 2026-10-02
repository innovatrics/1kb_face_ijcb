# SPDX-License-Identifier: MIT
"""Canonical index: crop scan, Color FERET pose codes, joins."""

from __future__ import annotations

import pandas as pd
import pytest

from face1kb.data import index as ix


def _touch(root, rel):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


def test_pose_codes():
    assert ix.cf_pose_code("09001_900101_fa_a") == "fa"
    assert ix.cf_pose_code("09002_910315_hl.png") == "hl"
    assert ix.cf_pose("09003_920606_qr") == "quarter right"
    assert len(ix.CF_POSE_NAMES) == 13
    assert set(ix.CF_POSE_NAMES) == set(ix.CF_POSE_YAW)
    with pytest.raises(ValueError):
        ix.cf_pose_code("Adriana Lima0_0")


def test_scan_sorted_by_subject_then_stem(tmp_path):
    for rel in [
        "b/x10_1.png",
        "b/x1_0.png",
        "a/z.png",
        "a/y.png",
        "a/.hidden.png",
        "a/notes.txt",
        ".cache/q.png",
    ]:
        _touch(tmp_path, rel)
    (tmp_path / "stray.png").write_bytes(b"")
    got = ix.scan_crops(tmp_path)
    assert [(s, st) for s, st, _ in got] == [
        ("a", "y"),
        ("a", "z"),
        ("b", "x10_1"),
        ("b", "x1_0"),
    ]


def test_scan_rejects_duplicate_stems(tmp_path):
    _touch(tmp_path, "a/x.png")
    _touch(tmp_path, "a/x.jpg")
    with pytest.raises(ValueError, match="duplicate stem"):
        ix.scan_crops(tmp_path, exts=(".png", ".jpg"))


def test_scan_rejects_upper_case_extension(tmp_path):
    # readers resolve <subject>/<stem>.png, which does not exist for x.PNG on a
    # case-sensitive file system
    _touch(tmp_path, "a/w.png")
    _touch(tmp_path, "a/x.PNG")
    with pytest.raises(ValueError, match="lower-case extension '.png'"):
        ix.scan_crops(tmp_path)


def test_scan_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        ix.scan_crops(tmp_path / "nope")


def test_build_index_colorferet(tmp_path):
    for rel in [
        "09002/09002_900101_fb.png",
        "09001/09001_900101_hl_a.png",
        "09001/09001_900101_fa_a.png",
    ]:
        _touch(tmp_path, rel)
    df = ix.build_index(tmp_path, dataset="colorferet")
    assert list(df.columns) == ["id", "rel_path", "subject", "pose"]
    assert df.id.tolist() == [0, 1, 2]
    assert df.rel_path.tolist() == [
        "09001/09001_900101_fa_a.png",
        "09001/09001_900101_hl_a.png",
        "09002/09002_900101_fb.png",
    ]
    assert df.subject.tolist() == ["09001", "09001", "09002"]
    assert df.pose.tolist()[1] == "half left"


def test_build_index_kk_has_no_pose(tmp_path):
    _touch(tmp_path, "pins_B/B1_0.png")
    _touch(tmp_path, "pins_A/A0_0.png")
    df = ix.build_index(tmp_path, dataset="kk")
    assert list(df.columns) == ["id", "rel_path", "subject"]
    assert df.subject.tolist() == ["pins_A", "pins_B"]


def test_write_read_keeps_zero_padding(tmp_path):
    _touch(tmp_path / "c", "09001/09001_900101_fa.png")
    df = ix.build_index(tmp_path / "c", dataset="colorferet")
    p = ix.write_index(df, tmp_path / "index.csv")
    back = ix.read_index_csv(p)
    assert back.subject.tolist() == ["09001"]
    assert back.equals(df)


def test_read_index_renames_identity(tmp_path):
    p = tmp_path / "i.csv"
    pd.DataFrame({"id": [0], "rel_path": ["a/b.jpg"], "identity": ["a"]}).to_csv(
        p, index=False
    )
    df = ix.read_index_csv(p)
    assert df.columns.tolist() == ["id", "rel_path", "subject"]
    # indexes listing the source file names point at the PNG crops
    assert df.rel_path.tolist() == ["a/b.png"]
    assert ix.crop_rel_path("pins_Mr. X/Mr. X1_2.jpg") == "pins_Mr. X/Mr. X1_2.png"


def test_compare_image_sets(tmp_path):
    for r in ("112", "224"):
        _touch(tmp_path / r, "a/x.png")
    _touch(tmp_path / "224", "a/y.png")
    rep = ix.compare_image_sets([tmp_path / "112", tmp_path / "224"])
    assert rep[str(tmp_path / "112")] == {"n": 1, "missing": 1, "extra": 0}
    assert rep[str(tmp_path / "224")] == {"n": 2, "missing": 0, "extra": 1}


def test_align_to_index():
    index = pd.DataFrame(
        {"id": [0, 1, 2], "rel_path": ["a/1.png", "a/2.png", "b/1.png"]}
    )
    table = pd.DataFrame(
        {"rel_path": ["b/1.jpg", "a/1.png", "a/2.png", "c/9.png"], "v": [3, 1, 2, 9]}
    )
    out = ix.align_to_index(index, table)
    assert out.rel_path.tolist() == index.rel_path.tolist()
    assert out.v.tolist() == [1, 2, 3]
    with pytest.raises(KeyError):
        ix.align_to_index(index, table.iloc[:2])
    loose = ix.align_to_index(index, table.iloc[:2], strict=False)
    assert loose.v.isna().tolist() == [False, True, False]
    with pytest.raises(ValueError, match="duplicate"):
        ix.align_to_index(index, pd.concat([table, table]))
