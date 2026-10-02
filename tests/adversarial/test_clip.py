# SPDX-License-Identifier: MIT
"""CLIP-surrogate I-FGSM, with a small stand-in image encoder (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
nnf = torch.nn.functional

from face1kb import adversarial as adv  # noqa: E402
from face1kb.adversarial.attacks import CLIP_MEAN, CLIP_STD  # noqa: E402

from ._synthetic import crops  # noqa: E402


class _Visual:
    input_resolution = 32


class FakeCLIP(nn.Module):
    """Stand-in with the two members the attack uses (``visual``, ``encode_image``)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.conv = nn.Conv2d(3, 8, 5, stride=2)
        self.visual = _Visual()

    def encode_image(self, x):
        return torch.tanh(self.conv(x)).flatten(1)


@pytest.fixture(scope="module")
def fake():
    net = FakeCLIP().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def linf_uint8(a, b):
    return int(np.abs(a.astype(int) - b.astype(int)).max())


def reference_attack(net, imgs, eps, steps, seed, batch=64):
    """The attack written with the global torch seed (paper formulation)."""
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    res = net.visual.input_resolution
    step = max(eps / 4.0, 1.0 / 255.0)

    def feats(x01):
        x = nnf.interpolate(x01, size=res, mode="bilinear", align_corners=False)
        return net.encode_image((x - mean) / std).float()

    torch.manual_seed(seed)
    out = np.empty_like(imgs)
    for s in range(0, len(imgs), batch):
        b = imgs[s : s + batch]
        clean = torch.from_numpy(b.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        with torch.no_grad():
            f0 = feats(clean)
        a = (clean + torch.empty_like(clean).uniform_(-eps, eps)).clamp_(0, 1)
        for _ in range(steps):
            a.requires_grad_(True)
            loss = (1 - nnf.cosine_similarity(feats(a), f0)).mean()
            (g,) = torch.autograd.grad(loss, a)
            a = a.detach() + step * g.sign()
            a = torch.min(torch.max(a, clean - eps), clean + eps).clamp_(0, 1)
        out[s : s + batch] = (
            (a.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
        )
    return out


@pytest.mark.parametrize("eps", adv.DEFAULT_EPS)
def test_shape_and_budget(fake, eps):
    x = crops(2, 48)
    y = adv.clip_surrogate_attack(x, eps, steps=3, model=fake)
    assert y.shape == x.shape and y.dtype == np.uint8
    assert linf_uint8(y, x) <= math.ceil(eps * 255)


def test_matches_paper_formulation(fake):
    x = crops(5, 48)
    got = adv.clip_surrogate_attack(x, 0.06, steps=4, model=fake, batch_size=2)
    assert np.array_equal(got, reference_attack(fake, x, 0.06, 4, seed=0, batch=2))


def test_steps_zero_is_the_random_start(fake):
    x = crops(2, 48)
    y = adv.clip_surrogate_attack(x, 0.1, steps=0, model=fake, seed=7)
    gen = torch.Generator().manual_seed(7)
    clean = torch.from_numpy(x.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    start = (clean + torch.empty_like(clean).uniform_(-0.1, 0.1, generator=gen)).clamp(
        0, 1
    )
    want = (start.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    assert np.array_equal(y, want)


def test_seeded_and_global_rng(fake):
    x = crops(3, 48)
    state = torch.random.get_rng_state()
    a = adv.clip_surrogate_attack(x, 0.03, steps=2, model=fake)
    assert torch.equal(state, torch.random.get_rng_state())
    assert np.array_equal(a, adv.clip_surrogate_attack(x, 0.03, steps=2, model=fake))
    b = adv.clip_surrogate_attack(x, 0.03, steps=2, model=fake, seed=1)
    assert not np.array_equal(a, b)


def test_registry_dispatch(fake):
    x = crops(1, 48)
    kw = dict(steps=1, model=fake)
    assert np.array_equal(
        adv.craft("clip", x, 0.06, **kw), adv.clip_surrogate_attack(x, 0.06, **kw)
    )


def _cached_clip():
    try:
        import clip  # noqa: F401
    except ImportError:
        return None
    from face1kb import config

    root = config.models_dir("clip")
    return root if (root / "ViT-B-32.pt").is_file() else None


@pytest.mark.skipif(_cached_clip() is None, reason="CLIP ViT-B/32 not cached")
def test_real_clip_budget():
    net = adv.load_clip(download_root=_cached_clip())
    x = crops(2, 112)
    y = adv.clip_surrogate_attack(x, 0.03, steps=2, model=net)
    assert y.shape == x.shape and linf_uint8(y, x) <= math.ceil(0.03 * 255)
    assert not np.array_equal(y, x)


def test_cuda_blocks_random_start(fake):
    from face1kb.adversarial import _philox

    x = crops(2, 48)
    y = adv.clip_surrogate_attack(x, 0.1, steps=0, model=fake, cuda_blocks=288)
    v = _philox.cuda_uniform(x.size, -0.1, 0.1, 0, 0, 288).reshape(x.shape)
    start = (
        torch.from_numpy(x.astype(np.float32) / 255.0) + torch.from_numpy(v)
    ).clamp(0, 1)
    assert np.array_equal(y, (start.numpy() * 255.0).round().astype(np.uint8))
