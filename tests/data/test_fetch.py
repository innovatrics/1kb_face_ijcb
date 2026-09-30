# SPDX-License-Identifier: MIT
"""Checksum guards of the model download helpers (local file:// URLs, no network)."""

from __future__ import annotations

import hashlib

import pytest

from face1kb import config
from face1kb.data import fetch


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_download_verifies_and_is_atomic(tmp_path):
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights" * 100)
    url = src.as_uri()
    dst = tmp_path / "models" / "m.bin"
    assert fetch.download(url, dst, _sha(src.read_bytes())) == dst
    assert dst.read_bytes() == src.read_bytes()
    assert not list(dst.parent.glob("*.part"))

    bad = tmp_path / "models" / "bad.bin"
    with pytest.raises(OSError, match="checksum mismatch"):
        fetch.download(url, bad, "0" * 64)
    assert not bad.exists()
    assert not list(bad.parent.glob("*.part"))


def test_verified_rejects_corrupt_cache(tmp_path):
    p = tmp_path / "x.onnx"
    assert fetch._verified(p, _sha(b"ok")) is False
    p.write_bytes(b"ok")
    assert fetch._verified(p, _sha(b"ok")) is True
    p.write_bytes(b"corrupt")
    with pytest.raises(OSError, match="delete it to re-download"):
        fetch._verified(p, _sha(b"ok"))


def test_model_paths_without_download(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        fetch.selfie_segmenter_path(download_missing=False)
    with pytest.raises(FileNotFoundError):
        fetch.genderage_path(download_missing=False)
    corrupt = tmp_path / "insightface" / "buffalo_l" / "genderage.onnx"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not the model")
    with pytest.raises(OSError, match="expected"):
        fetch.genderage_path(download_missing=False)
    assert fetch.insightface_pack_dir() == corrupt.parent


def _fake_pack(tmp_path, monkeypatch):
    """Point the buffalo_l pins at a local zip with two small models."""
    import zipfile

    models = {"genderage.onnx": b"ga model", "2d106det.onnx": b"landmark model"}
    zpath = tmp_path / "src" / "buffalo_l.zip"
    zpath.parent.mkdir()
    with zipfile.ZipFile(zpath, "w") as z:
        for name, data in models.items():
            z.writestr(name, data)
    monkeypatch.setattr(config, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(fetch, "BUFFALO_L_URL", zpath.as_uri())
    monkeypatch.setattr(fetch, "BUFFALO_L_SHA256", _sha(zpath.read_bytes()))
    monkeypatch.setattr(
        fetch, "BUFFALO_L_MODELS", {k: _sha(v) for k, v in models.items()}
    )
    return models


def test_insightface_model_path_extracts_and_repairs(tmp_path, monkeypatch):
    models = _fake_pack(tmp_path, monkeypatch)
    pack = fetch.insightface_pack_dir()
    pack.mkdir(parents=True)
    # only genderage present: the landmark model is still fetched on request
    (pack / "genderage.onnx").write_bytes(models["genderage.onnx"])
    with pytest.raises(FileNotFoundError):
        fetch.insightface_model_path("2d106det.onnx", download_missing=False)
    p = fetch.insightface_model_path("2d106det.onnx")
    assert p.read_bytes() == models["2d106det.onnx"]
    # a truncated model (interrupted extraction) is re-extracted
    p.write_bytes(b"land")
    with pytest.raises(OSError, match="expected"):
        fetch.insightface_model_path("2d106det.onnx", download_missing=False)
    assert (
        fetch.insightface_model_path("2d106det.onnx").read_bytes()
        == (models["2d106det.onnx"])
    )
    assert not list(pack.glob("*.part"))
    assert not (pack.parent / "buffalo_l.zip").exists()
    with pytest.raises(ValueError, match="no pinned"):
        fetch.insightface_model_path("other.onnx")
