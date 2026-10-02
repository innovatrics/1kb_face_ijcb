# SPDX-License-Identifier: MIT
"""The .ptci container: format, restricted unpickling, and a GPU round trip."""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from face1kb.baselines import compressai_codecs as ca
from face1kb.baselines.verify import synthetic_image


def _payload(shape=(2, 2)):
    return {
        "strings": [[b"\x01\x02\x03"], [b"\x04"]],
        "shape": shape,
        "size": (112, 112),
        "codec": "neural_mbt2018_mean",
        "quality": 3,
    }


def test_constants():
    assert ca.MODELS == {
        "neural_bmshj2018": "bmshj2018-factorized",
        "neural_mbt2018_mean": "mbt2018-mean",
    }
    assert ca.QUALITIES == tuple(range(1, 9))
    assert ca.EXTENSION == ".ptci" and ca.PAD_MULTIPLE == 64


def test_pack_format_and_entropy_size():
    d = _payload()
    data = ca.pack(d["strings"], d["shape"], d["size"], d["codec"], d["quality"])
    assert data[:2] == b"\x80\x04"  # pickle protocol 4
    assert pickle.loads(data) == d
    assert list(pickle.loads(data)) == ["strings", "shape", "size", "codec", "quality"]
    assert ca.unpack(data) == d
    assert ca.entropy_size(d["strings"]) == 4


def test_unpack_accepts_torch_size():
    torch = pytest.importorskip("torch")
    d = _payload(torch.Size([2, 2]))
    data = ca.pack(d["strings"], d["shape"], d["size"], d["codec"], d["quality"])
    out = ca.unpack(data)
    assert isinstance(out["shape"], torch.Size) and out["shape"] == (2, 2)


class _Exploit:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


@pytest.mark.parametrize(
    "obj",
    [
        {**_payload(), "shape": _Exploit()},
        {**_payload(), "extra": np.zeros(2)},
    ],
    ids=["os.system", "numpy"],
)
def test_unpack_rejects_globals(obj):
    with pytest.raises(pickle.UnpicklingError):
        ca.unpack(pickle.dumps(obj, protocol=4))


def test_unpack_validates_structure():
    with pytest.raises(ValueError):
        ca.unpack(pickle.dumps([1, 2]))
    with pytest.raises(ValueError):
        ca.unpack(pickle.dumps({k: v for k, v in _payload().items() if k != "size"}))
    with pytest.raises(ValueError):
        ca.unpack(pickle.dumps({**_payload(), "strings": ["abc"]}))
    with pytest.raises(ValueError):
        ca.unpack(pickle.dumps({**_payload(), "codec": "neural_unknown"}))


def test_padding():
    torch = pytest.importorskip("torch")
    x, size = ca.to_tensor(synthetic_image(112), device="cpu")
    assert size == (112, 112) and tuple(x.shape) == (1, 3, 128, 128)
    assert x.dtype == torch.float32
    # edge padding repeats the last row / column
    assert torch.equal(x[..., 111, :112], x[..., 127, :112])


@pytest.mark.gpu
@pytest.mark.parametrize("codec", ca.CODECS)
def test_round_trip_gpu(codec):
    pytest.importorskip("compressai")
    img = synthetic_image(112)
    try:
        data, info = ca.encode_to_budget(img, codec, 1024, device="cuda")
    except (OSError, RuntimeError) as exc:  # pretrained weights need a download
        pytest.skip(f"CompressAI weights not available: {exc}")
    assert info["fitted"] and info["size"] <= 1024 and info["setting"] in ca.QUALITIES
    assert len(data) > info["size"]  # the pickle container adds its overhead
    out = ca.decode(data, device="cuda")
    assert out.shape == img.shape and out.dtype == np.uint8
    d = ca.unpack(data)
    assert d["quality"] == info["setting"] and d["codec"] == codec


@pytest.mark.gpu
def test_decode_follows_the_encoder_under_any_global_cudnn_setting():
    # mbt2018-mean decodes with hyperprior scales that must match the encoder's; a
    # global deterministic / autotuning / TF32 setting must not leak into the call.
    torch = pytest.importorskip("torch")
    pytest.importorskip("compressai")
    img = synthetic_image(224, seed=2)
    codec = "neural_mbt2018_mean"
    try:
        data, info = ca.encode_to_budget(img, codec, 1024, device="cuda")
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"CompressAI weights not available: {exc}")
    x, _ = ca.to_tensor(img, "cuda")
    net = ca.get_net(codec, info["setting"], "cuda")
    with torch.no_grad(), ca.paper_numerics():
        ref = net(x)["x_hat"].clamp(0, 1)[0, :, :224, :224]
    ref = (ref.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    cudnn = torch.backends.cudnn
    saved = (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32)
    try:
        cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32 = True, True, True
        out = ca.decode(data, device="cuda")
        assert (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32) == (
            True,
            True,
            True,
        )
    finally:
        cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32 = saved
    assert np.abs(out.astype(int) - ref.astype(int)).max() <= 1
    again = ca.decode(data, device="cuda")
    assert np.abs(again.astype(int) - out.astype(int)).max() <= 1


def _mbt_stream(device="cpu"):
    pytest.importorskip("compressai")
    img = synthetic_image(112, seed=4)
    try:
        data, info = ca.encode_to_budget(
            img, "neural_mbt2018_mean", 1024, device=device
        )
    except (OSError, RuntimeError) as exc:  # pretrained weights need a download
        pytest.skip(f"CompressAI weights not available: {exc}")
    return data, ca.get_net("neural_mbt2018_mean", info["setting"], device)


def _glitch_hyperprior(monkeypatch, net, bad_calls):
    """Make the first ``bad_calls`` hyperprior runs return shifted scales."""
    import torch

    inner, calls = net.h_s, []

    class Glitch(torch.nn.Module):
        def forward(self, z):
            out = inner(z)
            calls.append(1)
            if len(calls) <= bad_calls:
                scales, means = out.chunk(2, 1)
                out = torch.cat([scales * 1.5, means], 1)
            return out

    monkeypatch.setattr(net, "h_s", Glitch())
    return calls


def test_verified_decode_recovers_from_a_hyperprior_glitch(monkeypatch):
    data, net = _mbt_stream()
    clean = ca.decode(data, device="cpu", verify=False)
    assert np.array_equal(ca.decode(data, device="cpu"), clean)
    calls = _glitch_hyperprior(monkeypatch, net, bad_calls=1)
    out = ca.decode(data, device="cpu")
    assert len(calls) == 2 and np.array_equal(out, clean)
    # without verification the glitch corrupts the reconstruction
    calls.clear()
    bad = ca.decode(data, device="cpu", verify=False)
    assert np.abs(bad.astype(int) - clean.astype(int)).max() > 1


def test_verified_decode_warns_when_nothing_verifies(monkeypatch, caplog):
    data, net = _mbt_stream()
    calls = _glitch_hyperprior(monkeypatch, net, bad_calls=100)
    with caplog.at_level("WARNING", logger=ca.__name__):
        out = ca.decode(data, device="cpu", attempts=3)
    # three runs with the default kernels, one with the deterministic kernels
    assert len(calls) == 4 and out.shape == (112, 112, 3)
    assert "did not verify" in caplog.text


def test_hyperprior_fallback_to_deterministic_kernels(monkeypatch):
    # a stream whose indexes only the deterministic kernels reproduce verifies at the
    # last attempt
    torch = pytest.importorskip("torch")
    data, net = _mbt_stream()
    clean = ca.decode(data, device="cpu", verify=False)
    calls = _glitch_hyperprior(monkeypatch, net, bad_calls=2)
    seen = []
    inner = net.h_s

    class Spy(torch.nn.Module):
        def forward(self, z):
            seen.append(torch.backends.cudnn.deterministic)
            return inner(z)

    monkeypatch.setattr(net, "h_s", Spy())
    out = ca.decode(data, device="cpu", attempts=2)
    assert len(calls) == 3 and seen == [False, False, True]
    assert np.array_equal(out, clean)


def test_paper_numerics_restores_flags():
    torch = pytest.importorskip("torch")
    cudnn = torch.backends.cudnn
    before = (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32)
    with ca.paper_numerics(True):
        assert cudnn.deterministic and not cudnn.benchmark and not cudnn.allow_tf32
    with ca.paper_numerics():
        assert not cudnn.deterministic
    assert (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32) == before
