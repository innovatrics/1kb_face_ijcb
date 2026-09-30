# SPDX-License-Identifier: MIT
"""Classical codecs: grids, knobs, budget fit and decode on synthetic images."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from face1kb.baselines import classical
from face1kb.baselines.verify import synthetic_image


def _available(codec):
    classical._ensure_plugins()
    Image.init()
    return classical.PIL_FORMATS[codec] in Image.SAVE


def test_benchmark_grids():
    # quality 2, 4, ..., 94 and JPEG 2000 ratio 400, 396, ..., 12
    q = classical.settings("jpeg")
    assert q == list(range(2, 96, 2)) and q[0] == 2 and q[-1] == 94 and len(q) == 47
    for codec in ("webp", "jpeg_xl", "avif", "heif"):
        assert classical.settings(codec) == q
    r = classical.settings("jpeg2000")
    assert r == list(range(400, 8, -4)) and r[0] == 400 and r[-1] == 12
    # coarser grids of the sample-difficulty study
    assert classical.settings("jpeg", quality_step=4) == list(range(2, 96, 4))
    assert classical.settings("jpeg2000", ratio_step=12) == list(range(400, 8, -12))


def test_save_kwargs():
    assert classical.save_kwargs("jpeg", 40) == {"quality": 40, "optimize": True}
    assert classical.save_kwargs("webp", 40) == {"quality": 40, "method": 6}
    assert classical.save_kwargs("jpeg2000", 36) == {
        "quality_mode": "rates",
        "quality_layers": [36],
    }
    for codec in ("jpeg_xl", "avif", "heif"):
        assert classical.save_kwargs(codec, 12) == {"quality": 12}
    with pytest.raises(ValueError):
        classical.save_kwargs("png", 1)


def test_extensions():
    assert classical.EXTENSIONS == {
        "jpeg": ".jpg",
        "jpeg2000": ".jp2",
        "webp": ".webp",
        "jpeg_xl": ".jxl",
        "avif": ".avif",
        "heif": ".heic",
    }


@pytest.mark.parametrize("codec", classical.CODECS)
@pytest.mark.parametrize("budget", [1024, 512])
def test_encode_to_budget_round_trip(codec, budget):
    if not _available(codec):
        pytest.skip(f"Pillow cannot write {codec} here")
    img = synthetic_image(112)
    data, info = classical.encode_to_budget(img, codec, budget)
    assert set(info) == {"fitted", "setting", "size"}
    assert info["size"] == len(data)
    assert info["fitted"] == (len(data) <= budget)
    assert info["fitted"]  # a smooth 112 px image fits both budgets
    assert info["setting"] in classical.settings(codec)
    assert data == classical.encode_setting(img, codec, info["setting"])
    out = classical.decode(data)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_unreachable_budget_returns_smallest_setting():
    img = synthetic_image(224)
    data, info = classical.encode_to_budget(img, "jpeg", 64)
    assert not info["fitted"] and info["setting"] == 2 and len(data) > 64


def test_metadata_of_pil_inputs():
    # A PIL input is encoded as Pillow saves it (the benchmark behaviour): the AVIF
    # encoder embeds an ICC profile of img.info. strip_metadata encodes pixels only.
    if not _available("avif"):
        pytest.skip("Pillow cannot write AVIF here")
    img = synthetic_image(64)
    pil = Image.fromarray(img)
    pil.info["icc_profile"] = _srgb_profile()
    plain = classical.encode_setting(img, "avif", 50)
    with_icc = classical.encode_setting(pil, "avif", 50)
    assert len(with_icc) > len(plain)
    assert classical.encode_setting(pil, "avif", 50, strip_metadata=True) == plain
    assert classical.to_pil(pil).info == pil.info
    assert classical.to_pil(pil, strip_metadata=True).info == {}


def _srgb_profile() -> bytes:
    from PIL import ImageCms

    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def test_decode_pil_keeps_info():
    if not _available("jpeg2000"):
        pytest.skip("Pillow cannot write JPEG 2000 here")
    data = classical.encode_setting(synthetic_image(64), "jpeg2000", 40)
    pil = classical.decode_pil(data)
    assert pil.mode == "RGB" and "comment" in pil.info  # the OpenJPEG comment
    assert np.array_equal(np.asarray(pil), classical.decode(data))


def test_scan_search_variant():
    img = synthetic_image(112)
    data, info = classical.encode_to_budget(
        img,
        "jpeg",
        1024,
        grid=classical.settings("jpeg", quality_step=4),
        search="scan",
        skip_errors=True,
    )
    assert info["fitted"] and info["setting"] % 4 == 2 and len(data) <= 1024
    with pytest.raises(TypeError):
        classical.encode_to_budget(img, "jpeg", 1024, keep="last")


def test_input_validation():
    with pytest.raises(ValueError):
        classical.to_pil(np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError):
        classical.to_pil(np.zeros((8, 8, 3), np.float32))
    buf = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).convert("L").save(buf, "PNG")
    assert classical.decode(buf.getvalue()).shape == (8, 8, 3)
