# SPDX-License-Identifier: MIT
"""JPEG-AI with the real reference software (needs its setup and a CUDA GPU)."""

from __future__ import annotations

import hashlib
import subprocess

import numpy as np
import pytest

from face1kb.baselines import decode_bytes, jpeg_ai
from face1kb.baselines.verify import synthetic_image

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def ref():
    if not jpeg_ai.available():
        pytest.skip("JPEG-AI reference software not set up (scripts/setup_jpegai.sh)")
    return jpeg_ai.get_reference()


def _checkout_state(repo) -> tuple[str, str]:
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--", "cfg"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    betas = sorted((repo / "cfg" / "betas").rglob("*.txt"))
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in betas)).hexdigest()
    return status, digest


def test_encode_decode_round_trip(ref):
    before = _checkout_state(ref.repo_dir)
    img = synthetic_image(112)
    data, info = jpeg_ai.encode_to_budget(img, 1024)
    assert info["fitted"] and len(data) == info["size"] <= 1024
    assert jpeg_ai.BPP_RANGE[0] <= info["setting"] <= jpeg_ai.BPP_RANGE[1]
    assert len(data) >= jpeg_ai.MIN_PLAUSIBLE_FRAC * 1024
    # the stream at the chosen target is reproducible
    assert jpeg_ai.encode_bpp(img, info["setting"]) == data
    out = decode_bytes(data, ".jpegai")
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(out, jpeg_ai.decode(data))
    # the bitrate log went to a private folder: the checkout is unchanged
    assert _checkout_state(ref.repo_dir) == before


def test_torch_settings_do_not_leak(ref):
    import torch

    cudnn = torch.backends.cudnn
    before = (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32)
    data = jpeg_ai.encode_bpp(synthetic_image(64), 20)
    jpeg_ai.decode(data)
    assert (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32) == before


def test_encoder_reconstruction_equals_decode(ref):
    img = synthetic_image(112, seed=3)
    data, recon = jpeg_ai.encode_bpp(img, 30, return_recon=True)
    assert np.array_equal(recon, jpeg_ai.decode(data))
