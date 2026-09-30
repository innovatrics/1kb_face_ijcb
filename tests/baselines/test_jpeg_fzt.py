# SPDX-License-Identifier: MIT
"""JPEG-FzT: the transform stages against golden values of the original code."""

from __future__ import annotations

import hashlib
from importlib import resources

import numpy as np
import pytest

from face1kb.baselines import jpeg_fzt
from face1kb.baselines.verify import synthetic_image

# SHA-256 of the stage outputs of the original implementation on
# synthetic_image(res, seed=1): (low-res image, rad=1 reconstruction,
# rad=2 reconstruction).
GOLDEN = {
    112: (
        "9e3f9e7e6ab42a4364e21953d46dd41898baa088f5601b3e50e310ed29e6d242",
        "a6ec6bc2dba8e2889524e93100a3c59fbd21048644efbeb07bd868b9efd4e9d3",
        "5233fe70899b011a2f12bf563f881111fa9b2937e9eaaa33f68469600684d981",
    ),
    57: (  # odd size: exercises the boundary rules
        "7badb7927f0888276eadcbebc009b86701af1c971cc1fe014ac5d6670fa11f73",
        "787c03dd49414d3ca6c113a1c011fdac1efd6b79258f17e05e04fb1f1670aa4a",
        "299dd1c6b3344db7373ab1fad30171fb3d2907c71b23baa3833b0c905fd788a4",
    ),
}
IBF_SHA256 = "d191b10d215927cd62d378b809584269f9ab4725e9deaee0823cead1a22c1b32"


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_packaged_table():
    ref = resources.files("face1kb.baselines").joinpath("data", "jpeg_fzt_ibf.txt")
    assert hashlib.sha256(ref.read_bytes()).hexdigest() == IBF_SHA256
    ibf = jpeg_fzt.load_ibf()
    assert ibf.dtype == np.float32 and ibf.shape == (4990,)
    assert ibf[0] == np.float32(1.27142)


def test_table_is_package_data():
    # The table must be declared as package data, or wheels ship without it.
    import fnmatch
    from pathlib import Path

    tomllib = pytest.importorskip("tomllib")
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("no pyproject.toml (installed package)")
    cfg = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    setuptools = cfg.get("tool", {}).get("setuptools", {})
    patterns = setuptools.get("package-data", {}).get("face1kb.baselines", [])
    patterns += setuptools.get("package-data", {}).get("*", [])
    assert any(fnmatch.fnmatch("data/jpeg_fzt_ibf.txt", p) for p in patterns), (
        "pyproject.toml must list data/*.txt as package data of face1kb.baselines"
    )


def test_constants():
    assert (jpeg_fzt.H, jpeg_fzt.HH, jpeg_fzt.RAD) == (2, 0, 1)
    assert jpeg_fzt.QUALITIES == tuple(range(1, 96, 2))
    assert jpeg_fzt.QUALITIES_EVEN == tuple(range(2, 96, 2))
    assert jpeg_fzt.EXTENSION == ".fzt"


@pytest.mark.parametrize("res", sorted(GOLDEN))
def test_stages_match_original(res):
    img = synthetic_image(res, seed=1)
    low = jpeg_fzt.encode_stage(img)
    assert low.shape == (res // 2, res // 2, 3) and low.dtype == np.uint8
    ibf = jpeg_fzt.load_ibf()
    rec1 = jpeg_fzt.decode_stage(low, (res, res), ibf, h=2, rad=1)
    rec2 = jpeg_fzt.decode_stage(low, (res, res), ibf, h=2, rad=2)
    assert (_sha(low), _sha(rec1), _sha(rec2)) == GOLDEN[res]


def test_encode_decode_round_trip():
    img = synthetic_image(112)
    data, info = jpeg_fzt.encode_to_budget(img, 1024)
    assert info["fitted"] and len(data) == info["size"] <= 1024
    assert info["setting"] in jpeg_fzt.QUALITIES
    # the .fzt file is a JPEG of the half-resolution image
    assert data[:2] == b"\xff\xd8"
    assert data == jpeg_fzt.encode_quality(jpeg_fzt.downsample(img), info["setting"])
    out = jpeg_fzt.decode(data)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(out, jpeg_fzt.decode(data, 112))
    # the legacy helper reconstructs the same image
    low, rec, size = jpeg_fzt.compress_with_quality(img, info["setting"])
    assert size == len(data) and np.array_equal(np.asarray(rec), out)
    assert jpeg_fzt.compress_only_with_quality(img, info["setting"])[1] == size


def test_even_grid_and_fallback():
    img = synthetic_image(112)
    _, info = jpeg_fzt.encode_to_budget(img, 1024, qualities=jpeg_fzt.QUALITIES_EVEN)
    assert info["setting"] % 2 == 0
    data, info = jpeg_fzt.encode_to_budget(synthetic_image(224), 200)
    assert not info["fitted"] and info["setting"] == 1 and len(data) > 200


def test_original_names_are_kept():
    assert jpeg_fzt._encode_stage is jpeg_fzt.encode_stage
    assert jpeg_fzt._decode_stage is jpeg_fzt.decode_stage
    assert jpeg_fzt._load_weights is jpeg_fzt.load_ibf
