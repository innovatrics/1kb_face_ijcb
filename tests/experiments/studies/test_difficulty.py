# SPDX-License-Identifier: MIT
"""Sample difficulty: sampling, descriptors, attributes, aggregates, tables."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from face1kb import config

from ._scripts import load
from .conftest import make_dataset

cp = load("difficulty", "compute")
an = load("difficulty", "analyze")
mt = load("difficulty", "make_tables")


def test_sample_prefix_property(roots):
    make_dataset("colorferet", n_subjects=30, per_subject=40)
    big = cp.sample_crops("colorferet", 112, 300, 0)
    small = cp.sample_crops("colorferet", 112, 20, 0)
    assert big[:20] == small
    files = sorted(config.aligned_dir("colorferet", 112).glob("*/*.png"))
    assert small == random.Random(0).sample(files, 20)


def test_descriptors():
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    flat = cp.to01(np.full((112, 112, 3), 100, np.uint8))
    d = cp.descriptors(flat)
    assert set(d) == set(cp.DESCRIPTORS)
    assert d["lap_var"] == 0.0 and d["entropy"] == 0.0 and d["contrast"] == 0.0
    assert d["brightness"] == 100.0 and d["colorfulness"] == 0.0
    rng = np.random.default_rng(0)
    noisy = cp.descriptors(cp.to01(rng.integers(0, 256, (112, 112, 3), np.uint8)))
    assert noisy["hf_share"] > 0.5 and noisy["entropy"] > 7.0


def test_attribute_join(roots):
    config.dataset_dir("colorferet").mkdir(parents=True)
    pd.DataFrame(
        {
            "subject": ["00001", "00001"],
            "image": ["00001_a", "00001_b"],
            "rel_path": ["00001/00001_a.png", "00001/00001_b.png"],
            "pose": ["frontal image", "half left"],
            "gender": ["male", "male"],
            "race": ["White", "White"],
            "glasses": ["yes", "no"],
            "beard": [1, 0],
            "mustache": [0, 0],
            "age_from": [50, 50],
            "yaw": [0.0, -67.5],
            "pitch": [0.0, 0.0],
        }
    ).to_csv(config.labels_csv("colorferet"), index=False)
    config.dataset_dir("kk").mkdir(parents=True)
    pd.DataFrame(
        {
            "identity": ["pins_x", "pins_x"],
            "image": ["x0_1.png", "x0_2.png"],
            "age": [30, np.nan],
            "gender": ["F", ""],
            "mst_label": ["MST5", ""],
            "mst_index": [5, np.nan],
        }
    ).to_csv(config.attributes_csv("kk"), index=False)
    attrs = cp.load_attributes()
    a = attrs["colorferet"][("00001", "00001_b")]
    assert a["abs_yaw"] == 67.5 and a["glasses"] == 0 and a["beard"] == 0
    assert attrs["colorferet"][("00001", "00001_a")]["glasses"] == 1
    assert attrs["kk"][("pins_x", "x0_1")] == {"age": 30, "gender": "F", "mst_index": 5}
    assert attrs["kk"][("pins_x", "x0_2")]["mst_index"] == ""


def _per_image(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ds in ("colorferet", "kk"):
        for i in range(n):
            detail = rng.uniform(100, 3000)
            glasses = int(i % 7 == 0)
            for codec in ("webp", "avif", "ours_accurate"):
                for budget in (512, 1024):
                    rows.append(
                        {
                            "dataset": ds,
                            "subject": f"{i // 3:05d}",
                            "image": f"img{i:03d}",
                            "budget": budget,
                            "codec": codec,
                            "id_cos": 1 - detail / 1e4 - 0.01 * rng.random(),
                            "lap_var": detail,
                            "hf_share": detail / 1e5,
                            "grad_mag": rng.random(),
                            "edge_density": rng.random(),
                            "entropy": rng.random(),
                            "contrast": rng.random(),
                            "colorfulness": rng.random(),
                            "brightness": rng.random(),
                            "abs_yaw": rng.random() * 90,
                            "glasses": glasses,
                            "beard": 0,
                            "mustache": int(i % 2),
                            "mst_index": np.nan,
                            "age": 20 + i if i < 12 else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def test_aggregates_and_tables(tmp_path):
    df = _per_image()
    corr = an.correlations(df)
    by = corr.set_index("key")
    assert by.loc["lap_var", "colorferet"] > 0.95  # difficulty grows with detail
    assert by.loc["lap_var", "colorferet_n"] == 60
    assert np.isnan(by.loc["beard", "colorferet"])  # a single distinct value
    assert np.isnan(by.loc["mst_index", "kk"]) and by.loc["mst_index", "kk_n"] == 0
    assert by.loc["age", "kk_n"] == 12
    sharing = an.cross_codec_sharing(df)
    assert set(sharing) == {"colorferet", "kk"} and sharing["kk"][0] > 0.9
    assert an.sharing_text(sharing).startswith("colorferet: mean_pairwise_rho=")
    con = an.contrast(df)
    g = con.set_index("key").loc["glasses"]
    assert g["k"] == 6 and g["n"] == 60
    assert (
        con.set_index("key").loc["lap_var", "hardest"]
        > con.set_index("key").loc["lap_var", "easiest"]
    )
    # CSV round trip gives the same tables
    corr.to_csv(tmp_path / "c.csv", index=False)
    con.to_csv(tmp_path / "k.csv", index=False)
    t1 = mt.predictors_table(corr)
    assert t1 == mt.predictors_table(pd.read_csv(tmp_path / "c.csv"))
    assert "n=60/dataset" in t1.splitlines()[1]
    lap = next(ln for ln in t1.splitlines() if ln.startswith("Laplacian"))
    assert lap.split(" & ")[1].startswith("\\textbf{+") and "$^{**}$" in lap
    mst = next(ln for ln in t1.splitlines() if ln.startswith("Monk"))
    assert mst == "Monk skin-tone index & -- & -- \\\\"
    t2 = mt.contrast_table(con, mt.PAPER_HEADER)
    assert t2 == mt.contrast_table(pd.read_csv(tmp_path / "k.csv"), mt.PAPER_HEADER)
    assert (
        t2.splitlines()[1]
        == "% image-level deciles: 6 easiest vs 6 hardest images (of 60)"
    )
    assert "Fisher $p=" in t2


def test_make_tables_cli(tmp_path):
    pytest.importorskip("matplotlib")
    df = _per_image()
    an.correlations(df).to_csv(tmp_path / "difficulty_correlations.csv", index=False)
    an.contrast(df).to_csv(tmp_path / "difficulty_contrast.csv", index=False)
    rc = mt.main(
        [
            "--input-dir",
            str(tmp_path),
            "--tables-dir",
            str(tmp_path / "t"),
            "--figures-dir",
            str(tmp_path / "f"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "t" / "difficulty_predictors.tex").is_file()
    assert (tmp_path / "t" / "difficulty_contrast.tex").is_file()
    assert (tmp_path / "f" / "difficulty_predictors.png").is_file()
