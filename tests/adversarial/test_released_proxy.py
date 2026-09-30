# SPDX-License-Identifier: MIT
"""The released Li-AE proxy weights (weights/liae_proxy.safetensors)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from face1kb.adversarial import proxy as P  # noqa: E402

from ._resources import released_proxy_available  # noqa: E402
from ._synthetic import crops  # noqa: E402

RELEASED = {
    "sha256": "eac885eaf6088a5352f8a8a4350a9bbe66e15fb41666fa1d46d97a20943e7afa",
    "bytes": 5_536_012,
    "params": 1_381_443,
    "tensors": 51,
    "metadata": {
        "format": "pt",
        "face1kb.format_version": "1",
        "face1kb.architecture": "ProxyAE",
        "face1kb.width": "32",
        "face1kb.parameters": "1381443",
        "face1kb.n_ids": "400",
        "face1kb.n_imgs": "3854",
        "face1kb.epochs": "60",
        "face1kb.license": "CC-BY-NC-SA-4.0",
    },
    # CPU fp32 forward of crops(2, 112) / 255
    "forward": {
        "mid_mean": 0.2497460327814821,
        "mid_abs_mean": 0.504631952462055,
        "code_mean": 0.15472590646259007,
        "code_abs_mean": 0.3772166949867944,
        "recon_mean": 0.45682304514751004,
        "recon_std": 0.12200518567507614,
    },
}


pytestmark = [
    pytest.mark.weights,
    pytest.mark.skipif(
        not released_proxy_available(), reason="liae_proxy.safetensors missing"
    ),
]


def test_file_and_metadata():
    path = P.default_proxy_path()
    assert path.stat().st_size == RELEASED["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == RELEASED["sha256"]
    state, meta = P.read_proxy_file(path)
    for k, v in RELEASED["metadata"].items():
        assert meta[k] == v, k
    assert "WebFace42M" in meta["face1kb.training_data"]
    assert len(state) == RELEASED["tensors"]
    for k, v in state.items():
        want = torch.int64 if k.endswith("num_batches_tracked") else torch.float32
        assert v.dtype == want, k


def test_strict_load_and_forward():
    net = P.load_proxy()
    assert P.count_parameters(net) == RELEASED["params"]
    x = torch.from_numpy(crops(2, 112).astype(np.float32) / 255).permute(0, 3, 1, 2)
    with torch.no_grad():
        recon, f = net(x)
    got = {
        "mid_mean": f["mid"].double().mean(),
        "mid_abs_mean": f["mid"].double().abs().mean(),
        "code_mean": f["code"].double().mean(),
        "code_abs_mean": f["code"].double().abs().mean(),
        "recon_mean": recon.double().mean(),
        "recon_std": recon.double().std(),
    }
    for k, v in RELEASED["forward"].items():
        assert float(got[k]) == pytest.approx(v, rel=1e-5, abs=1e-6), k
