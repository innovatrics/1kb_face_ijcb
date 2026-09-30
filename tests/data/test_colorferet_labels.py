# SPDX-License-Identifier: MIT
"""Color FERET labels parser on a synthetic NIST-style ground-truth tree."""

from __future__ import annotations

import tarfile

import pandas as pd
import pytest

from face1kb.data import colorferet_labels as cl

from ._nist import write_tree as _write_tree


def test_parse_xml(tmp_path):
    _write_tree(tmp_path)
    df = cl.parse_ground_truth(tmp_path)
    assert list(df.columns) == list(cl.COLUMNS)
    assert df.image.tolist() == [
        "09001_900101_fa_a",
        "09001_900101_fb_a",
        "09002_910315_hl",
    ]
    r = df.iloc[0]
    assert r.subject == "09001" and r.rel_path == "09001/09001_900101_fa_a.png"
    assert r.pose == "frontal image" and r.gender == "male"
    assert r.race == "Black" and r.race_nist == "Black-or-African-American"
    assert r.glasses == "yes" and r.beard == 2 and r.mustache == 1
    assert r.age_from == r.age_to == 40 and r.year_of_birth == 1950
    assert df.iloc[2].race == "Asian" and df.iloc[2].age_from == 26
    assert df.iloc[2].yaw == 67.5 and df.iloc[2].pitch == 0.0
    assert df.iloc[1].mustache == 2 and df.iloc[1].pose_code == "fb"
    assert df.iloc[1].glasses == "no" and df.iloc[1].beard == 1
    assert "left_eye" not in " ".join(df.columns).lower()


def test_name_value_equals_xml(tmp_path):
    _write_tree(tmp_path)
    a = cl.parse_ground_truth(tmp_path, fmt="xml")
    b = cl.parse_ground_truth(tmp_path, fmt="name_value")
    pd.testing.assert_frame_equal(a, b)


def test_auto_falls_back_to_name_value(tmp_path):
    _write_tree(tmp_path)
    for p in (tmp_path / "colorferet").rglob("*.xml"):
        p.unlink()
    df = cl.parse_ground_truth(tmp_path)
    assert len(df) == 3


def test_tar_source_and_dvd_root(tmp_path):
    _write_tree(tmp_path / "src")
    tar_path = tmp_path / "colorferet.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(tmp_path / "src" / "colorferet", arcname="colorferet")
    a = cl.parse_ground_truth(tmp_path / "src")
    pd.testing.assert_frame_equal(a, cl.parse_ground_truth(tar_path))
    one = cl.parse_ground_truth(tmp_path / "src" / "colorferet" / "dvd2")
    assert one.image.tolist() == ["09002_910315_hl"]


def test_write_read_roundtrip(tmp_path):
    _write_tree(tmp_path)
    df = cl.parse_ground_truth(tmp_path)
    back = cl.read_labels(cl.write_labels(df, tmp_path / "labels.csv"))
    assert back.subject.tolist() == ["09001", "09001", "09002"]
    assert back.image.tolist() == df.image.tolist()


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        cl.parse_ground_truth(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        cl.parse_ground_truth(tmp_path / "empty")
    rec = {"subject": "00009", "pose_code": "fa", "capture_date": "1/1/1990"}
    with pytest.raises(KeyError):
        cl.labels_from_records({}, {"00009_x_fa": rec})


def test_race_merge_is_the_paper_seven_class_scheme():
    assert cl.RACE_MERGE == {
        "Black-or-African-American": "Black",
        "Asian-Middle-Eastern": "Asian",
        "Asian-Southern": "Asian",
    }
