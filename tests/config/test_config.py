# SPDX-License-Identifier: MIT
"""Path configuration (face1kb.config)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import face1kb.config as config


@pytest.fixture()
def cfg(monkeypatch, tmp_path):
    for name in ("DATA", "WORK", "OUTPUT", "MODELS"):
        monkeypatch.setenv(f"FACE1KB_{name}_ROOT", str(tmp_path / name.lower()))
    monkeypatch.setenv("FACE1KB_WEIGHTS_DIR", str(tmp_path / "w"))
    monkeypatch.delenv("FACE1KB_LAYOUT", raising=False)
    mod = importlib.reload(config)
    yield mod, tmp_path
    monkeypatch.undo()
    importlib.reload(config)


def test_constants():
    assert config.DATASETS == ("colorferet", "kk")
    assert config.RESOLUTIONS == (64, 96, 112, 168, 224)
    assert config.BUDGETS == (1024, 512)
    assert "cheng" + "2020" not in str(vars(config))


def test_defaults_are_inside_the_repo(monkeypatch):
    for name in ("DATA", "WORK", "OUTPUT", "MODELS"):
        monkeypatch.delenv(f"FACE1KB_{name}_ROOT", raising=False)
    monkeypatch.delenv("FACE1KB_WEIGHTS_DIR", raising=False)
    mod = importlib.reload(config)
    try:
        repo = Path(mod.__file__).resolve().parents[1]
        assert mod.DATA_ROOT == repo / "data"
        assert mod.WORK_ROOT == repo / "work"
        assert mod.OUTPUT_ROOT == repo / "outputs"
        assert mod.WEIGHTS_DIR == repo / "weights"
        assert mod.RESULTS_ROOT == repo / "results"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_public_layout(cfg):
    c, tmp = cfg
    d, w = tmp / "data", tmp / "work"
    assert c.aligned_dir("kk", 112) == d / "kk" / "aligned_112"
    assert c.aligned_dir("colorferet", 112, "_adv_hfc_003") == (
        d / "colorferet" / "aligned_112_adv_hfc_003"
    )
    assert c.index_csv("kk") == d / "kk" / "index.csv"
    assert c.pairs_parquet("colorferet") == d / "colorferet" / "pairs.parquet"
    assert c.labels_csv() == d / "colorferet" / "labels.csv"
    assert c.attributes_csv() == d / "kk" / "attributes.csv"
    assert c.compressed_dir("kk", 224, 512, "ours_fast") == (
        w / "kk" / "compressed" / "224px_512B" / "ours_fast"
    )
    assert c.compressed_dir("kk", 112, 1024, "webp", "_A1") == (
        w / "kk" / "compressed" / "112_A1px_1024B" / "webp"
    )
    assert c.decoded_dir("kk", 224, 512, "jpeg_ai") == (
        w / "kk" / "decoded" / "224px_512B" / "jpeg_ai"
    )
    assert c.embeddings_path("kk", "lvface_l", "aligned_112") == (
        w / "kk" / "embeddings" / "lvface_l" / "aligned_112.npy"
    )
    assert c.embeddings_manifest("kk") == w / "kk" / "embeddings" / "manifest.csv"
    assert c.weights_path("fast") == tmp / "w" / "face1kb_fast.safetensors"
    assert c.output_dir("accuracy") == tmp / "output" / "accuracy"
    with pytest.raises(ValueError):
        c.aligned_dir("lfw", 112)
    with pytest.raises(ValueError):
        c.weights_path("medium")


def test_embedding_tag():
    assert config.embedding_tag(112) == "aligned_112"
    assert config.embedding_tag(112, "_A1") == "aligned_112_A1"
    assert config.embedding_tag(224, codec="ours_fast", budget=512) == (
        "ours_fast_224_512"
    )
    assert config.embedding_tag(112, "_adv_clip_003", "webp", 1024) == (
        "webp_112_adv_clip_003_1024"
    )
    with pytest.raises(ValueError):
        config.embedding_tag(112, codec="webp")


def test_legacy_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("FACE1KB_DATA_ROOT", str(tmp_path / "datasets"))
    monkeypatch.setenv("FACE1KB_WORK_ROOT", str(tmp_path / "datasets"))
    monkeypatch.setenv("FACE1KB_LAYOUT", "legacy")
    c = importlib.reload(config)
    try:
        d = tmp_path / "datasets"
        assert c.aligned_dir("kk", 112) == d / "ai_solutions_kk_aligned" / (
            "dat_aligned_112"
        )
        assert c.index_csv("colorferet") == (
            d / "colorferet" / "alignment" / "pairs_images.csv"
        )
        assert c.decoded_dir("kk", 224, 1024, "ours_accurate") == (
            d
            / "ai_solutions_kk_aligned"
            / "compressed"
            / "224px_1024B"
            / "ours_accurate_png"
        )
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_read_index_normalises_the_legacy_kk_schema(monkeypatch, tmp_path):
    pytest.importorskip("pandas")
    monkeypatch.setenv("FACE1KB_DATA_ROOT", str(tmp_path / "datasets"))
    monkeypatch.setenv("FACE1KB_LAYOUT", "legacy")
    c = importlib.reload(config)
    try:
        kk = c.index_csv("kk")
        kk.parent.mkdir(parents=True)
        kk.write_text(
            "id,rel_path,identity,det_score,residual_px\n"
            "0,pins_A B/A B0_0.jpg,pins_A B,0.97,3.3\n"
            "1,pins_A B/A B1_2.jpg,pins_A B,0.95,4.0\n"
        )
        cf = c.index_csv("colorferet")
        cf.parent.mkdir(parents=True)
        cf.write_text("id,rel_path,subject,pose\n0,00001/00001_x_fa.png,1,frontal\n")
        df = c.read_index("kk")
        assert list(df.columns[:3]) == ["id", "rel_path", "subject"]
        assert df.rel_path.tolist() == ["pins_A B/A B0_0.png", "pins_A B/A B1_2.png"]
        assert df.subject.tolist() == ["pins_A B", "pins_A B"]
        df = c.read_index("colorferet")
        assert df.rel_path.tolist() == ["00001/00001_x_fa.png"]
        assert df.subject.tolist() == ["00001"] and "pose" in df.columns
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_weights_dir_hint(monkeypatch):
    monkeypatch.setattr(config, "INSTALLED_COPY", False)
    assert config.weights_dir_hint() == ""
    monkeypatch.setattr(config, "INSTALLED_COPY", True)
    monkeypatch.delenv("FACE1KB_WEIGHTS_DIR", raising=False)
    assert "FACE1KB_WEIGHTS_DIR" in config.weights_dir_hint()
    monkeypatch.setenv("FACE1KB_WEIGHTS_DIR", "/somewhere")
    assert config.weights_dir_hint() == ""


def test_bad_layout(monkeypatch):
    monkeypatch.setenv("FACE1KB_LAYOUT", "weird")
    with pytest.raises(ValueError):
        importlib.reload(config)
    monkeypatch.undo()
    importlib.reload(config)
