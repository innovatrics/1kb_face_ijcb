# SPDX-License-Identifier: MIT
"""Registry and decode dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from face1kb import baselines, config
from face1kb.baselines import decode as decode_mod
from face1kb.baselines.verify import scan_jpegai, synthetic_image

EXPECTED = {
    "jpeg": (".jpg", "cpu"),
    "jpeg2000": (".jp2", "cpu"),
    "webp": (".webp", "cpu"),
    "jpeg_xl": (".jxl", "cpu"),
    "avif": (".avif", "cpu"),
    "heif": (".heic", "cpu"),
    "jpeg_fzt": (".fzt", "cpu"),
    "jpeg_ai": (".jpegai", "cuda"),
    "neural_bmshj2018": (".ptci", "cuda"),
    "neural_mbt2018_mean": (".ptci", "cuda"),
}


def test_ten_baselines():
    assert baselines.CODECS == tuple(EXPECTED)
    for name, (ext, device) in EXPECTED.items():
        c = baselines.get_codec(name)
        assert (c.name, c.ext, c.device) == (name, ext, device)
    assert baselines.CPU_CODECS == tuple(list(EXPECTED)[:7])
    assert baselines.GPU_CODECS == tuple(list(EXPECTED)[7:])
    with pytest.raises(ValueError):
        baselines.get_codec("neural_unknown")
    # swapped arguments of the package-level encode_to_budget(codec, img, budget)
    with pytest.raises(TypeError, match="codec name"):
        baselines.encode_to_budget(np.zeros((8, 8, 3), np.uint8), "webp", 1024)


def test_extensions_include_ours():
    assert baselines.extension("ours_fast") == config.OURS_EXT
    assert baselines.extension("ours_accurate") == config.OURS_EXT
    assert baselines.codec_for_extension(".ptci") == (
        "neural_bmshj2018",
        "neural_mbt2018_mean",
    )
    assert baselines.codec_for_extension("bin") == ("ours_fast", "ours_accurate")
    with pytest.raises(ValueError):
        baselines.extension("png")


def test_labels_match_report_style():
    from face1kb.report.style import CODEC_LABELS

    for name in baselines.CODECS:
        assert baselines.get_codec(name).label == CODEC_LABELS[name]


@pytest.mark.parametrize("name", ["jpeg", "webp", "jpeg_fzt"])
def test_uniform_interface_and_decode_file(name, tmp_path):
    codec = baselines.get_codec(name)
    img = synthetic_image(96)
    data, info = baselines.encode_to_budget(name, img, 1024)
    assert info["fitted"] and set(info) == {"fitted", "setting", "size"}
    out = codec.decode(data, res=96, device="cpu")
    path = tmp_path / f"crop{codec.ext}"
    path.write_bytes(data)
    assert np.array_equal(baselines.decode_file(path, res=96), out)
    assert np.array_equal(baselines.decode_bytes(data, codec.ext), out)
    assert out.shape == (96, 96, 3)


def test_decode_unknown_extension():
    with pytest.raises(ValueError):
        baselines.decode_bytes(b"", ".xyz")


def test_decoded_cache_path(monkeypatch):
    stream = Path("/w/colorferet/compressed/112px_1024B/jpeg_ai/00001/a.jpegai")
    monkeypatch.setattr(config, "LAYOUT", "public")
    assert decode_mod.decoded_cache_path(stream) == Path(
        "/w/colorferet/decoded/112px_1024B/jpeg_ai/00001/a.png"
    )
    monkeypatch.setattr(config, "LAYOUT", "legacy")
    assert decode_mod.decoded_cache_path(stream) == Path(
        "/w/colorferet/compressed/112px_1024B/jpeg_ai_png/00001/a.png"
    )
    assert decode_mod.decoded_cache_path("/x/y.jpegai") is None


def test_decoded_cache_path_agrees_with_config(monkeypatch, tmp_path):
    for layout in ("public", "legacy"):
        monkeypatch.setattr(config, "LAYOUT", layout)
        monkeypatch.setattr(config, "WORK_ROOT", tmp_path)
        rel = Path("00001/00001_930831_fa_a.png")
        bits = config.compressed_dir(
            "kk", 224, 512, "webp", "_tight"
        ) / rel.with_suffix(".webp")
        assert decode_mod.decoded_cache_path(bits) == (
            config.decoded_dir("kk", 224, 512, "webp", "_tight") / rel
        )


def test_scan_jpegai(tmp_path):
    cell = tmp_path / "compressed" / "112px_1024B" / "jpeg_ai" / "s"
    cell.mkdir(parents=True)
    (cell / "ok.jpegai").write_bytes(b"x" * 600)
    (cell / "cut.jpegai").write_bytes(b"x" * 100)
    bad = scan_jpegai(tmp_path)
    assert [(p.name, n) for p, n in bad] == [("cut.jpegai", 100)]


def test_decode_file_prefers_cache_when_asked(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LAYOUT", "public")
    monkeypatch.setattr(config, "WORK_ROOT", tmp_path)
    img = synthetic_image(64)
    data, _ = baselines.encode_to_budget("webp", img, 1024)
    stream = config.compressed_dir("kk", 64, 1024, "webp") / "s" / "a.webp"
    stream.parent.mkdir(parents=True)
    stream.write_bytes(data)
    live = baselines.decode_file(stream)
    assert np.array_equal(baselines.decode_file(stream, cache=True), live)  # no cache
    cached = decode_mod.decoded_cache_path(stream)
    cached.parent.mkdir(parents=True)
    marker = np.zeros_like(img)
    from PIL import Image

    Image.fromarray(marker).save(cached)
    assert np.array_equal(baselines.decode_file(stream, cache=True), marker)
    assert np.array_equal(baselines.decode_file(stream), live)


def test_decode_ignores_options_of_other_decoders():
    img = synthetic_image(64)
    data, _ = baselines.encode_to_budget("jpeg", img, 1024)
    out = baselines.decode_bytes(data, ".jpg", weights_dir="w", profile="sop")
    assert out.shape == img.shape
