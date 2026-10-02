# SPDX-License-Identifier: MIT
"""Dataset statistics, figures and the Color FERET attribute table."""

from __future__ import annotations

import random

import pandas as pd
import pytest

from face1kb.data import stats
from face1kb.data.attributes import MONK_LABELS, MONK_PALETTE, read_attributes


def test_cf_attribute_table_format():
    df = pd.DataFrame(
        {
            "gender": ["male", "male", "female", "male"],
            "race": ["White", "Asian", "White", "Black"],
            "glasses": ["no", "no", "yes", "no"],
            "pose": ["half left", "frontal image", "frontal image", "half left"],
        }
    )
    tex = stats.cf_attribute_table(df)
    lines = tex.splitlines()
    assert lines[0] == stats.CF_TABLE_HEADER
    assert lines[1] == "\\begin{tabular}{@{}lrr@{}}"
    assert "\\quad male & 3 & 75.0 \\\\" in lines
    assert "\\quad female & 1 & 25.0 \\\\" in lines
    assert lines[-2] == "\\end{tabular}" or lines[-1] == "\\end{tabular}"
    assert "\\textbf{Total} & \\textbf{4} & \\textbf{100.0} \\\\" in lines
    assert tex.endswith("\\end{tabular}\n")
    assert stats.cf_attribute_table(df, header=None).startswith("\\begin{tabular}")
    assert stats._esc("a_b & 5%") == "a\\_b \\& 5\\%"


def test_index_summary():
    idx = pd.DataFrame(
        {
            "id": range(5),
            "rel_path": [f"s/{k}.png" for k in range(5)],
            "subject": ["a", "a", "b", "b", "b"],
        }
    )
    s = stats.index_summary(idx)
    assert s["images"] == 5 and s["subjects"] == 2
    assert s["pairs"] == {"total": 10, "mated": 4, "non_mated": 6}
    assert s["images_per_subject"]["max"] == 3


def test_pick_subjects_is_seeded_sample_of_sorted():
    subjects = [f"pins_{c}" for c in "qwertyuiopasdfghjkl"]
    assert stats.pick_subjects(subjects, 5, 0) == random.Random(0).sample(
        sorted(subjects), 5
    )


def test_raw_file_stats(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    for ident, n in (("b", 3), ("a", 2)):
        (tmp_path / ident).mkdir()
        for k in range(n):
            PIL.new("RGB", (10 + k, 20)).save(tmp_path / ident / f"{k}.jpg")
    (tmp_path / "a" / "notes.txt").write_text("x")
    st = stats.raw_file_stats(tmp_path, dims_per_subject=2)
    assert st["n_identities"] == 2 and st["n_images"] == 5
    assert st["per_id"] == [2, 3]
    assert len(st["widths"]) == 4 and set(st["heights"]) == {20}
    assert st == stats.raw_file_stats(tmp_path, dims_per_subject=2)


def test_figures_write_files(tmp_path):
    pytest.importorskip("matplotlib")
    att = pd.DataFrame(
        {
            "subject": ["a", "a", "b"],
            "image": ["1.png", "2.png", "1.png"],
            "age": [25, 31, ""],
            "gender": ["F", "M", "F"],
            "mst_label": ["MST6", "", "MST7"],
            "mst_index": [6, "", 7],
            "skin_hex": ["#A07E56", "", "#825C43"],
        }
    )
    paths = stats.attribute_figures(att, tmp_path)
    paths.append(stats.fig_images_per_subject([10, 20, 30], tmp_path / "imgs.png"))
    raw = {"sizes_kb": [1.0, 5.0, 200.0], "widths": [100, 200], "heights": [90, 80]}
    paths += stats.raw_figures(raw, tmp_path)
    for p in paths:
        assert p.is_file() and p.stat().st_size > 1000


def test_attribute_constants_and_reader(tmp_path):
    assert MONK_LABELS[0] == "MST1" and len(MONK_PALETTE) == 10
    p = tmp_path / "att.csv"
    pd.DataFrame(
        {"identity": ["pins_A"], "image": ["A0_0.png"], "age": [30], "gender": ["F"]}
    ).to_csv(p, index=False)
    df = read_attributes(p)
    assert df.columns.tolist()[:3] == ["subject", "image", "rel_path"]
    assert df.rel_path.tolist() == ["pins_A/A0_0.png"]
