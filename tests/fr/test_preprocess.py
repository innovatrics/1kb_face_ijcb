# SPDX-License-Identifier: MIT
"""Input handling and preprocessing of face1kb.fr (pure numpy)."""

from __future__ import annotations

import numpy as np
import pytest

from face1kb.fr.embedders import (
    Embedder,
    as_batch,
    normalize,
    ort_providers,
    resize_batch,
    resolve_device,
)

RNG = np.random.default_rng(0)


def crops(n=3):
    return RNG.integers(0, 256, (n, 112, 112, 3), dtype=np.uint8)


def test_as_batch_accepts_list_and_array():
    x = crops()
    np.testing.assert_array_equal(as_batch(list(x)), x)
    assert as_batch(x) is x
    assert as_batch([]).shape == (0, 112, 112, 3)


def test_as_batch_rejects_wrong_shape_and_dtype():
    with pytest.raises(ValueError, match="112"):
        as_batch([np.zeros((96, 96, 3), np.uint8)])
    with pytest.raises(ValueError):
        as_batch([np.zeros((112, 112), np.uint8)])
    with pytest.raises(TypeError, match="uint8"):
        as_batch([np.zeros((112, 112, 3), np.float32)])


def test_minus_one_one_matches_the_paper_formula_bitwise():
    x = crops().astype(np.float32).transpose(0, 3, 1, 2)
    y = normalize(x, "[-1,1]")
    assert y.dtype == np.float32
    np.testing.assert_array_equal(y, ((x / 255.0) - 0.5) / 0.5)
    # ((x / 255) - 0.5) * 2 is the same float32 value (division by 0.5 is exact)
    np.testing.assert_array_equal(y, ((x / 255.0) - 0.5) * 2.0)
    assert y.min() >= -1.0 and y.max() <= 1.0


def test_other_normalisations():
    x = crops().astype(np.float32).transpose(0, 3, 1, 2)
    np.testing.assert_array_equal(normalize(x, "[0,1]"), x / 255.0)
    assert normalize(x, "none") is x and normalize(x, None) is x
    np.testing.assert_array_equal(
        normalize(x, (127.5, 128.0)), (x - np.float32(127.5)) / np.float32(128.0)
    )
    mean, std = (10.0, 20.0, 30.0), (1.0, 2.0, 4.0)
    y = normalize(x, (mean, std))
    for c in range(3):
        np.testing.assert_allclose(y[:, c], (x[:, c] - mean[c]) / std[c])
    nhwc = x.transpose(0, 2, 3, 1)
    y2 = normalize(nhwc, (mean, std), channel_axis=3)
    np.testing.assert_array_equal(y2, y.transpose(0, 2, 3, 1))
    np.testing.assert_array_equal(normalize(x, lambda a: a * 0), np.zeros_like(x))
    with pytest.raises(ValueError):
        normalize(x, "[-2,2]")


def test_resize_batch():
    x = crops(2)
    assert resize_batch(x, 112) is x
    pytest.importorskip("cv2")
    assert resize_batch(x, 128).shape == (2, 128, 128, 3)


def test_devices_and_providers():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"
    assert resolve_device(None) in ("cpu", "cuda")
    with pytest.raises(ValueError):
        resolve_device("mps")
    assert ort_providers("cpu") == ["CPUExecutionProvider"]
    assert ort_providers("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert ort_providers("cuda:2")[0] == ("CUDAExecutionProvider", {"device_id": 2})


def test_embedder_base_handles_empty_input():
    class E(Embedder):
        def _embed(self, x):
            raise AssertionError("not called for an empty batch")

    out = E().embed([])
    assert out.shape == (0, 512) and out.dtype == np.float32
