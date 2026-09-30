# SPDX-License-Identifier: MIT
"""Crafted crops against stored adversarial crop sets (needs the datasets).

The HFC attack is deterministic on the CPU and must reproduce stored crops bit for
bit. The Li-AE and CLIP attacks use the paper's random-start stream
(``cuda_blocks=paper_cuda_blocks(...)``), so they can be checked on any device, but
their gradient kernels are not bit-deterministic (see ``docs/adversarial.md``):
Li-AE reproduces most crops exactly, CLIP only a fraction of the pixels.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from face1kb import adversarial as adv
from face1kb import config

from ._resources import released_proxy_available

pytestmark = pytest.mark.data


def _load(dataset: str, n: int, suffix: str = ""):
    from PIL import Image

    root = config.aligned_dir(dataset, adv.ATTACK_RES, suffix)
    if not root.is_dir() or not config.index_csv(dataset).is_file():
        pytest.skip(f"{root} not available")
    index = config.read_index(dataset).sort_values("id")
    rels = index["rel_path"].to_numpy()[:n]
    return np.stack([np.asarray(Image.open(root / r).convert("RGB")) for r in rels])


def _pixels_equal(a: np.ndarray, b: np.ndarray) -> float:
    return float((a == b).mean())


def _crops_equal(a: np.ndarray, b: np.ndarray) -> float:
    return float((a == b).reshape(len(a), -1).all(1).mean())


@pytest.mark.parametrize("dataset", config.DATASETS)
@pytest.mark.parametrize("eps", adv.DEFAULT_EPS)
def test_hfc_reproduces_stored_crops(dataset, eps):
    pytest.importorskip("cv2")
    stored = _load(dataset, 16, adv.adv_suffix("hfc", eps))
    clean = _load(dataset, 16)
    assert np.array_equal(adv.hfc_attack(clean, eps, seed=0), stored)


def _liae(dataset, eps, device):
    if not released_proxy_available():
        pytest.skip("liae_proxy.safetensors missing (git lfs pull)")
    stored = _load(dataset, 64, adv.adv_suffix("liae", eps))
    clean = _load(dataset, 64)
    blocks = adv.paper_cuda_blocks(dataset, "liae")
    got = adv.li_ae_attack(clean, eps, device=device, cuda_blocks=blocks)
    assert np.abs(got.astype(int) - clean.astype(int)).max() <= math.ceil(eps * 255)
    return got, stored


@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.parametrize("dataset", config.DATASETS)
@pytest.mark.parametrize("eps", adv.DEFAULT_EPS)
def test_liae_close_to_stored_crops_cuda(dataset, eps):
    pytest.importorskip("torch")
    got, stored = _liae(dataset, eps, "cuda")
    if adv.cuda_launch_blocks() == adv.paper_cuda_blocks(dataset, "liae"):
        # the paper GPU model: only the run-to-run variation of cuDNN remains
        assert _crops_equal(got, stored) >= 0.9
    assert _pixels_equal(got, stored) >= 0.95


@pytest.mark.weights
def test_liae_close_to_stored_crops_cpu():
    pytest.importorskip("torch")
    got, stored = _liae("colorferet", 0.06, "cpu")
    assert _pixels_equal(got, stored) >= 0.95


def _cached_clip():
    try:
        import clip  # noqa: F401
    except ImportError:
        return None
    root = config.models_dir("clip")
    return root if (root / "ViT-B-32.pt").is_file() else None


@pytest.mark.gpu
@pytest.mark.skipif(_cached_clip() is None, reason="CLIP ViT-B/32 not cached")
@pytest.mark.parametrize("dataset", config.DATASETS)
def test_clip_random_start_matches_stored_crops(dataset):
    """Same random start: ~25-40 % equal pixels; a different one gives ~10 %."""
    eps = 0.06
    stored = _load(dataset, 64, adv.adv_suffix("clip", eps))
    clean = _load(dataset, 64)
    net = adv.load_clip(device="cuda", download_root=_cached_clip())
    blocks = adv.paper_cuda_blocks(dataset, "clip")
    got = adv.clip_surrogate_attack(clean, eps, model=net, cuda_blocks=blocks)
    assert _pixels_equal(got, stored) >= 0.2
