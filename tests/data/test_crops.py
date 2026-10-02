# SPDX-License-Identifier: MIT
"""Derived crop folders and the subsampling consistency check."""

from __future__ import annotations

import numpy as np
import pytest

from face1kb.data import alignment as al
from face1kb.data import crops

cv2 = pytest.importorskip("cv2")


def _write(root, rel, img):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)


def _crop(seed, size=224):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (size // 16, size // 16, 3), dtype=np.uint8)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)


def test_make_resolution_and_check_subsampling(tmp_path):
    rel = [f"s{k}/x{k}.png" for k in range(5)]
    for k, rp in enumerate(rel):
        _write(tmp_path / "224", rp, _crop(k))
    assert crops.make_resolution(tmp_path / "224", tmp_path / "112", 112, rel, 1) == (
        5,
        5,
    )
    assert crops.check_subsampling(tmp_path / "224", tmp_path / "112", rel) == (5, 5)
    # a crop of another alignment is detected
    _write(tmp_path / "112", rel[2], _crop(99, 112))
    assert crops.check_subsampling(tmp_path / "224", tmp_path / "112", rel) == (5, 4)
    assert crops.check_subsampling(tmp_path / "224", tmp_path / "112", []) == (0, 0)
    # non-integer ratio: written, but not exact
    assert crops.make_resolution(tmp_path / "224", tmp_path / "80", 80, rel, 1) == (
        5,
        0,
    )


def test_make_variants_equals_render_variant(tmp_path):
    rel = ["a/1.png", "b/2.png"]
    for k, rp in enumerate(rel):
        _write(tmp_path / "224", rp, _crop(k))
    dst = {v: tmp_path / f"112_{v}" for v in al.CROP_VARIANTS}
    assert crops.make_variants(tmp_path / "224", dst, rel, workers=1) == 6
    for v, d in dst.items():
        src = cv2.imread(str(tmp_path / "224" / rel[1]))
        assert np.array_equal(cv2.imread(str(d / rel[1])), al.render_variant(src, v))
    assert crops.make_variants(tmp_path / "224", dst, rel, workers=1) == 0
    with pytest.raises(ValueError):
        crops.make_variants(tmp_path / "224", {"loose": tmp_path / "x"}, rel)
