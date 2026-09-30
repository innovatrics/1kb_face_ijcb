# SPDX-License-Identifier: MIT
"""Tests of the ISO/IEC 29794-5 annex-study drivers in ``experiments/annex`` (CPU)."""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from face1kb import config

pytest.importorskip("cv2")  # the annex drivers need OpenCV (eval extra)

REPO = Path(__file__).resolve().parents[3]
DRIVERS = REPO / "experiments" / "annex"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_annex_{name}", DRIVERS / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sweep():
    return _load("sweep")


@pytest.fixture(scope="module")
def stageb():
    return _load("stageb")


@pytest.fixture(scope="module")
def score():
    return _load("score")


@pytest.fixture(scope="module")
def prep():
    return _load("prep")


# ------------------------------------------------------------------ cells
def test_cell_ids(sweep):
    c = sweep.make_cell(
        "jpeg", 96, "color", "rectangle_mean", {"smooth": 30, "arithmetic": True}
    )
    assert (
        sweep.cell_id(c)
        == "jpeg_r96_color_rectangle_mean_arithmeticTrue-smooth30_b1024"
    )
    c = sweep.make_cell(
        "avif", 168, "color", "rectangle_mean", {"subsampling": "4:4:4", "speed": 1}
    )
    assert (
        sweep.cell_id(c)
        == "avif_r168_color_rectangle_mean_speed1-subsampling4:4:4_b1024"
    )
    assert sweep.cell_id(sweep.make_cell("none", 112, "color", "none")) == (
        "none_r112_color_none_def_b1024"
    )


def test_stage_cell_counts(sweep):
    assert len(sweep.cells_for("A")) == 336
    assert len({sweep.cell_id(c) for c in sweep.cells_for("A")}) == 336
    exp1 = sweep.cells_for("exp1")
    assert len(exp1) == 17
    assert sorted({c["arm"] for c in exp1}) == ["annexE", "annexF", "ours"]
    assert len(sweep.cells_for("jpegai")) == 16
    assert len(sweep.cells_for("baseline")) == 7
    assert sweep.models_for("A") == ["arcface_antelopev2", "cvlface_vit_b"]
    assert len(sweep.models_for("confirm")) == 5


def test_cells_from_json(sweep, tmp_path):
    cells = [
        sweep.make_cell(
            "heif", 96, "gray", "mean", {"chroma_downsampling": "sharp-yuv"}
        )
    ]
    p = tmp_path / "stage_b_cells.json"
    p.write_text(json.dumps(cells))
    assert sweep.cells_for("B", str(p)) == cells
    with pytest.raises(SystemExit):
        sweep.cells_for("confirm", str(tmp_path / "missing.json"))


def test_j2k_ladder(sweep):
    r = sweep.J2K_RATES
    assert r[0] == 400 and r == sorted(r, reverse=True)
    assert 1.9 < r[-1] < 2.0  # down to a compression ratio of about 2
    assert sweep.QUALITIES[0] == 2 and sweep.QUALITIES[-1] == 94


# ------------------------------------------------------------------ manipulations
def test_mask_rect_covers_template(sweep):
    for res in (56, 112, 224):
        m = sweep.mask_rect(res)
        pts = (sweep.ARCFACE_DST_112 * res / 112).astype(int)
        assert m[pts[:, 1], pts[:, 0]].all()
        assert m[0, 0] == 0 and m.sum() < res * res


def test_manipulate(sweep):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    lmk = np.array(
        [[112 + 80 * np.cos(t), 112 + 80 * np.sin(t)] for t in np.linspace(0, 6, 106)],
        np.float32,
    )
    assert sweep.manipulate(img, "none", lmk) is img
    blur = sweep.manipulate(img, "mean", lmk)
    assert blur.shape == img.shape and not np.array_equal(blur, img)
    for manip in ("rectangle_mean", "ofiq_landmarks"):
        out = sweep.manipulate(img, manip, lmk)
        keep = (out == img).all(2)
        assert keep.any() and (~keep).any()
        assert np.array_equal(out[~keep], blur[~keep])


# ------------------------------------------------------------------ encoders
def _face(res=64, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:res, :res]
    base = (128 + 60 * np.sin(x / 5.0) * np.cos(y / 7.0))[..., None]
    img = base + rng.normal(0, 20, (res, res, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


@pytest.mark.parametrize(
    ("codec", "flags"),
    [
        ("jpeg", {"smooth": 30}),
        ("jpeg2000", {"num_resolutions": 3}),
        ("webp", {"method": 6}),
        ("avif", {"subsampling": "4:2:0", "speed": 6}),
    ],
)
def test_encoder_fits_budget(sweep, codec, flags):
    img = _face()
    enc, settings = sweep.encoder(codec, img, flags)
    idx, fit = sweep.hinted_binary_search_fit(enc, settings, 1024)
    assert fit.fitted and fit.size <= 1024 and fit.size == len(fit.payload)
    dec = sweep.decode_image(fit.payload)
    assert dec.shape == (64, 64, 3)


def test_jpeg_xl_effort_clamped(sweep, monkeypatch):
    seen = {}

    def fake_save(img, fmt, **kw):
        seen.update(kw)
        return 1, b"x"

    monkeypatch.setattr(sweep, "_save", fake_save)
    enc, _ = sweep.encoder("jpeg_xl", _face(), {"effort": 10})
    enc(50)
    assert seen["effort"] == 9


@pytest.mark.skipif(shutil.which("jpegtran") is None, reason="jpegtran not installed")
def test_jpeg_arithmetic(sweep):
    img = _face().convert("L")
    enc, _ = sweep.encoder("jpeg", img, {"arithmetic": True})
    n, data = enc(50)
    assert n == len(data) and data[:2] == b"\xff\xd8"
    with Image.open(io.BytesIO(data)) as im:
        assert im.size == (64, 64)


@pytest.mark.skipif(shutil.which("cwebp") is None, reason="cwebp not installed")
def test_cwebp(sweep):
    size, data = sweep.cwebp(_face(), 1024, {"method": 6, "sns": 40})
    assert 0 < size <= 1024 and data[:4] == b"RIFF"


def test_check_tools(sweep, monkeypatch):
    monkeypatch.setattr(sweep.shutil, "which", lambda name: None)
    cells = [sweep.make_cell("webp", 64, "color", "none", {"method": 6, "sns": 40})]
    with pytest.raises(SystemExit):
        sweep.check_tools(cells)
    sweep.check_tools([sweep.make_cell("webp", 64, "color", "none")])


# ------------------------------------------------------------------ stage B
def _scores(stageb):
    rows = []
    eer = 0.0
    for codec in ("jpeg", "webp", "jpeg_ai"):
        for res in (96, 112, 168):
            eer += 0.001
            for model in stageb.MODELS:
                rows.append(
                    {
                        "dataset": "colorferet",
                        "model": model,
                        "codec": codec,
                        "res": res,
                        "color": "color",
                        "manip": "none",
                        "flags": "{}",
                        "arm": "grid",
                        "all_sym_eer": eer,
                    }
                )
    return pd.DataFrame(rows)


def test_stage_b_ignores_jpegai_grid_cells(stageb):
    cells = stageb.stage_b(_scores(stageb))
    per_codec = {}
    for c in cells:
        per_codec.setdefault(c["codec"], set()).add(c["res"])
    assert set(per_codec) == {"jpeg", "webp"}
    assert per_codec["jpeg"] == {96, 112}
    n = stageb.TOP_K * (len(stageb.FLAG_SETS["jpeg"]) + len(stageb.FLAG_SETS["webp"]))
    assert len(cells) == n and {c["arm"] for c in cells} == {"stageB"}


def test_winners(stageb):
    df = _scores(stageb)
    df.loc[(df.codec == "webp") & (df.res == 168), "all_sym_eer"] = 0.0
    w = {c["codec"]: c for c in stageb.winners(df)}
    assert set(w) == {"jpeg", "webp"}
    assert w["webp"]["res"] == 168 and w["webp"]["arm"] == "winner"
    assert w["jpeg"]["flags"] == {}


def test_flag_set_sizes(stageb):
    # the paper's Stage B: two cells per codec -> 62 cells
    assert 2 * sum(len(v) for v in stageb.FLAG_SETS.values()) == 62


# ------------------------------------------------------------------ scoring
def test_metrics(score):
    pos = np.array([0.9, 0.8, 0.7, 0.2])
    neg = np.array([0.1, 0.3, 0.25, 0.75, 0.05])
    m = score.metrics(pos, neg)
    # at the mated score 0.7: FNMR 1/4, FMR 1/5
    assert m["eer"] == pytest.approx((0.25 + 0.2) / 2)
    assert set(m) == {"eer", "fnmr_0.001", "fnmr_0.0001"}
    assert score.metrics(np.array([0.9, 0.8]), np.array([0.1, 0.2]))["eer"] == 0.0


def test_make_pairs(score):
    meta = pd.DataFrame({"sid": ["b", "a", "a", "b", "c"]})
    pos, neg = score.make_pairs(meta, np.ones(5, bool))
    assert sorted(map(tuple, pos.tolist())) == [(0, 3), (1, 2)]
    s = meta.sid.to_numpy()
    assert (s[neg[:, 0]] != s[neg[:, 1]]).all()
    pos, _ = score.make_pairs(meta, np.array([True, True, False, True, True]))
    assert pos.tolist() == [[0, 3]]


def test_read_cells_latest_file_wins(score, tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(config, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(config, "LAYOUT", "public")
    d = score.annex_dir("colorferet")
    d.mkdir(parents=True)
    pd.DataFrame({"cell_id": ["x", "y"], "arm": ["grid", "grid"]}).to_parquet(
        d / "cells_b.parquet"
    )
    pd.DataFrame({"cell_id": ["x"], "arm": ["winner"]}).to_parquet(
        d / "cells_a.parquet"
    )
    os.utime(d / "cells_b.parquet", (1, 1))
    cells = score.read_cells("colorferet").set_index("cell_id")
    assert cells.loc["x", "arm"] == "winner" and cells.loc["y", "arm"] == "grid"


def test_ci_matches_score_eer(score):
    torch = pytest.importorskip("torch")
    ci = _load("ci")
    rng = np.random.default_rng(3)
    pos = rng.normal(0.6, 0.1, 500).astype(np.float32)
    neg = rng.normal(0.1, 0.1, 5000).astype(np.float32)
    assert ci.eer(torch.as_tensor(pos), torch.as_tensor(neg)) == pytest.approx(
        score.metrics(pos, neg)["eer"], abs=1e-7
    )


def test_ci_arm_kind():
    pytest.importorskip("torch")
    ci = _load("ci")
    row = pd.Series(
        {
            "flags": "{}",
            "color": "color",
            "manip": "none",
            "codec": "jpeg",
            "res": 96,
            "arm": "grid",
        }
    )
    assert ci.arm_kind(row) == "ours"
    assert ci.arm_kind(row.replace(96, 112)) == "grid"


# ------------------------------------------------------------------ prep
def _index(rels):
    return pd.DataFrame({"id": range(len(rels)), "rel_path": rels})


def test_build_meta_colorferet(prep):
    rels = [
        "00001/00001_930831_hl_a.png",
        "00001/00001_930831_fa_a.png",
        "00001/00001_930831_pr_a.png",
        "00001/00001_930831_fb_a.png",
        "00001/00001_930831_hr_a.png",
        "00001/00001_931230_fa.png",
        "00002/00002_930831_hl.png",
    ]
    meta = prep.build_meta("colorferet", _index(rels))
    assert list(meta.columns) == list(prep.META_COLUMNS)
    assert meta.image.tolist() == [
        "00001_930831_fa_a",
        "00001_930831_fb_a",
        "00001_931230_fa",
        "00001_930831_hl_a",
        "00002_930831_hl",
    ]
    assert meta.frontal.tolist() == [True, True, True, False, False]
    assert meta.id.tolist() == [1, 3, 5, 0, 6]


def test_build_meta_kk_caps_per_identity(prep):
    rels = [f"pins_A/A{i}.png" for i in range(35)] + ["pins_B/B0.png"]
    meta = prep.build_meta("kk", _index(rels))
    assert (meta.sid == "pins_A").sum() == 30 and meta.frontal.all()
    assert meta.image.tolist()[:3] == ["A0", "A1", "A10"]  # string order


def test_crop_source(prep, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(config, "LAYOUT", "public")
    for r in (112, 224):
        config.aligned_dir("kk", r).mkdir(parents=True)
    assert prep.crop_source("kk", 112) == (112, True)
    assert prep.crop_source("kk", 56) == (112, True)
    assert prep.crop_source("kk", 80) == (224, False)
    shutil.rmtree(config.aligned_dir("kk", 224))
    with pytest.raises(SystemExit):
        prep.crop_source("kk", 80)
