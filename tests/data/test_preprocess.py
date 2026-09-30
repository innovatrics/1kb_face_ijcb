# SPDX-License-Identifier: MIT
"""Preprocessing operators (OpenCV part; segmenters are exercised with fake masks)."""

from __future__ import annotations

import numpy as np
import pytest

from face1kb.data import preprocess as pp

cv2 = pytest.importorskip("cv2")


def _crop(seed=0, size=112):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (14, 14, 3), dtype=np.uint8)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)


def test_operator_lists():
    assert pp.OPERATORS == ("std", "A1", "A2", "A3", "A4", "B1", "B2", "C1", "C2")
    assert set(pp.OPERATOR_DESCRIPTIONS) == set(pp.OPERATORS)


@pytest.mark.parametrize("op", ["A1", "A2", "A3", "A4"])
def test_cpu_ops_shape_and_determinism(op):
    bgr = _crop()
    a = pp.CPU_FUNCS[op](bgr.copy())
    b = pp.CPU_FUNCS[op](bgr.copy())
    assert a.shape == bgr.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, bgr)


@pytest.mark.parametrize("op", ["A1", "A2", "A3", "A4"])
def test_apply_operator_rgb_equals_bgr(op):
    bgr = _crop(1)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    out_rgb = pp.apply_operator(rgb, op)
    out_bgr = pp.apply_operator(bgr, op, bgr=True)
    assert np.array_equal(out_rgb[:, :, ::-1], out_bgr)


def test_std_is_identity_and_unknown_op():
    x = _crop()
    assert pp.apply_operator(x, "std") is x
    with pytest.raises(ValueError):
        pp.apply_operator(x, "Z9")


def test_composite():
    bgr = np.full((4, 4, 3), 200, np.uint8)
    alpha = np.zeros((4, 4), np.float32)
    alpha[:2] = 1.0
    alpha[2, 0] = 0.5
    out = pp.composite(bgr, alpha)
    assert (out[:2] == 200).all() and (out[3] == 128).all()
    assert out[2, 0, 0] == 164


def test_fovea_mask_matches_definition():
    m = np.zeros((112, 112), np.float32)
    cv2.ellipse(m, (56, 72), (36, 46), 0, 0, 360, 1.0, -1)
    ref = cv2.GaussianBlur(m, (0, 0), 6.0)[..., None]
    assert np.array_equal(pp.fovea_mask(112), ref)
    assert pp.fovea_mask(224).shape == (224, 224, 1)


def test_mask_operators_with_given_mask():
    bgr = _crop(2)
    alpha = np.ones((112, 112), np.float32)
    alpha[:, :30] = 0.0
    b1 = pp.apply_with_mask(bgr, "B1", alpha)
    assert (b1[:, :30] == 128).all() and np.array_equal(b1[:, 30:], bgr[:, 30:])
    assert np.array_equal(pp.apply_with_mask(bgr, "C1", alpha), pp.op_a1(b1))
    c2 = pp.apply_with_mask(bgr, "C2", alpha)
    fov = pp.fovea_mask(112)
    smooth = cv2.bilateralFilter(b1, 9, 100, 100).astype(np.float32)
    ref = (
        (b1.astype(np.float32) * fov + smooth * (1 - fov)).clip(0, 255).astype(np.uint8)
    )
    assert np.array_equal(c2, ref)
    with pytest.raises(ValueError):
        pp.apply_with_mask(bgr, "A1", alpha)


def test_preprocess_folder_cpu(tmp_path):
    src = tmp_path / "aligned_112"
    rels = ["s1/a.png", "s2/b.png"]
    for k, rp in enumerate(rels):
        (src / rp).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(src / rp), _crop(k))
    dst = {"A2": tmp_path / "aligned_112_A2", "A3": tmp_path / "aligned_112_A3"}
    n = pp.preprocess_folder(src, dst, rels, workers=1)
    assert n == {"A2": 2, "A3": 2}
    got = cv2.imread(str(dst["A3"] / rels[1]))
    assert np.array_equal(got, pp.op_a3(cv2.imread(str(src / rels[1]))))
    assert pp.preprocess_folder(src, dst, rels, workers=1) == {"A2": 0, "A3": 0}
    with pytest.raises(ValueError):
        pp.preprocess_folder(src, {"X": tmp_path}, rels)


class _FakeBiRefNet:
    calls: list[int] = []

    def __init__(self, device=None):
        pass

    def masks(self, bgrs):
        _FakeBiRefNet.calls.append(len(bgrs))
        return [np.full(b.shape[:2], 0.5, np.float32) for b in bgrs]


class _FakeMediaPipe:
    def mask(self, bgr):
        return np.full(bgr.shape[:2], 0.25, np.float32)

    def close(self):
        pass


def test_preprocess_folder_resume_writes_only_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "BiRefNetSegmenter", _FakeBiRefNet)
    monkeypatch.setattr(pp, "MediaPipeSegmenter", _FakeMediaPipe)
    _FakeBiRefNet.calls = []
    src = tmp_path / "aligned_112"
    rels = ["s1/a.png", "s1/b.png", "s2/c.png"]
    for k, rp in enumerate(rels):
        (src / rp).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(src / rp), _crop(k))
    dst = {op: tmp_path / f"aligned_112_{op}" for op in ("B1", "B2")}
    assert pp.preprocess_folder(src, {"B2": dst["B2"]}, rels) == {"B2": 3}
    assert _FakeBiRefNet.calls == []  # B2 alone needs no BiRefNet
    b2 = dst["B2"] / rels[0]
    b2.write_bytes(b"sentinel")  # an existing output must not be rewritten
    (dst["B2"] / rels[2]).unlink()
    assert pp.preprocess_folder(src, dst, rels) == {"B1": 3, "B2": 1}
    assert b2.read_bytes() == b"sentinel"
    assert sum(_FakeBiRefNet.calls) == 3
    _FakeBiRefNet.calls = []
    assert pp.preprocess_folder(src, dst, rels) == {"B1": 0, "B2": 0}
    assert _FakeBiRefNet.calls == []
    assert pp.preprocess_folder(src, dst, rels, overwrite=True) == {"B1": 3, "B2": 3}
    assert b2.read_bytes() != b"sentinel"
