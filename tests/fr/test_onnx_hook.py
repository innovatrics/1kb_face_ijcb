# SPDX-License-Identifier: MIT
"""register_onnx end to end with tiny ONNX graphs built on the fly (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from face1kb import fr

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper  # noqa: E402


def _mean_model(path, batch, layout="NCHW"):
    """Graph: per-channel spatial mean -> (N, 3); input name 'images'."""
    if layout == "NCHW":
        shape, axes = [batch, 3, 112, 112], [2, 3]
    else:
        shape, axes = [batch, 112, 112, 3], [1, 2]
    axes_init = helper.make_tensor("axes", TensorProto.INT64, [2], axes)
    node = helper.make_node("ReduceMean", ["images", "axes"], ["emb"], keepdims=0)
    graph = helper.make_graph(
        [node],
        "mean",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("emb", TensorProto.FLOAT, [batch, 3])],
        [axes_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


@pytest.fixture()
def clean_registry():
    before = dict(fr.REGISTRY)
    yield
    fr.REGISTRY.clear()
    fr.REGISTRY.update(before)


def _crops(n):
    return np.random.default_rng(1).integers(0, 256, (n, 112, 112, 3), dtype=np.uint8)


def _ref(crops, bgr=False, norm="[-1,1]"):
    x = crops[..., ::-1] if bgr else crops
    x = x.astype(np.float32)
    if norm == "[-1,1]":
        x = ((x / 255.0) - 0.5) / 0.5
    return x.mean(axis=(1, 2))


@pytest.mark.parametrize("bgr", [False, True])
def test_dynamic_batch(clean_registry, tmp_path, bgr):
    path = _mean_model(tmp_path / "m.onnx", "N")
    fr.register_onnx("toy", path, color="BGR" if bgr else "RGB", normalize="[-1,1]")
    emb = fr.load("toy", device="cpu")
    assert emb.fixed_batch is None and emb.input_name == "images" and emb.dim == 3
    x = _crops(5)
    np.testing.assert_allclose(emb.embed(x), _ref(x, bgr), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("batch,n", [(1, 4), (3, 4), (3, 6), (4, 2)])
def test_fixed_batch_is_chunked_and_padded(clean_registry, tmp_path, batch, n):
    path = _mean_model(tmp_path / f"b{batch}.onnx", batch)
    fr.register_onnx("toyb", path, input_name="images", color="BGR")
    emb = fr.load("toyb", device="cpu")
    assert emb.fixed_batch == batch
    x = _crops(n)
    out = emb.embed(x)
    assert out.shape == (n, 3) and out.dtype == np.float32
    np.testing.assert_allclose(out, _ref(x, True), rtol=1e-5, atol=1e-6)


def test_nhwc_and_raw_pixels(clean_registry, tmp_path):
    path = _mean_model(tmp_path / "nhwc.onnx", "N", layout="NHWC")
    fr.register_onnx("toyh", path, layout="NHWC", normalize="none")
    x = _crops(2)
    np.testing.assert_allclose(
        fr.load("toyh", device="cpu").embed(x), _ref(x, norm="none"), rtol=1e-5
    )


def test_sha256_is_checked(clean_registry, tmp_path):
    path = _mean_model(tmp_path / "m.onnx", "N")
    fr.register_onnx("toys", path, sha256="0" * 64)
    with pytest.raises(RuntimeError, match="sha256"):
        fr.load("toys", device="cpu")
    fr.load("toys", device="cpu", verify=False)


def test_unknown_input_name(clean_registry, tmp_path):
    path = _mean_model(tmp_path / "m.onnx", "N")
    fr.register_onnx("toyx", path, input_name="data")
    with pytest.raises(ValueError, match="input"):
        fr.load("toyx", device="cpu")
