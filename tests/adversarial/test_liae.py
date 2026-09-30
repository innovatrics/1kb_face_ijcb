# SPDX-License-Identifier: MIT
"""Li-AE attack (I-FGSM on the proxy code + ILA) with a small random proxy (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from face1kb import adversarial as adv  # noqa: E402
from face1kb.adversarial.proxy import ProxyAE  # noqa: E402

from ._synthetic import crops  # noqa: E402


@pytest.fixture(scope="module")
def proxy():
    torch.manual_seed(1)
    return ProxyAE(width=4).eval()


def linf_uint8(a, b):
    return int(np.abs(a.astype(int) - b.astype(int)).max())


def reference_craft(net, imgs, eps, steps, ila_steps, seed, batch):
    """The crafting loop written with the global torch seed (paper formulation)."""
    step = max(eps / 4.0, 1.0 / 255.0)
    torch.manual_seed(seed)
    out = np.empty_like(imgs)

    def project(a, c):
        return torch.min(torch.max(a, c - eps), c + eps).clamp_(0, 1)

    for s in range(0, len(imgs), batch):
        chunk = imgs[s : s + batch]
        clean = torch.from_numpy(chunk.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        with torch.no_grad():
            f0 = net.encode(clean)
        a = project(clean + torch.empty_like(clean).uniform_(-eps, eps), clean)
        for _ in range(steps):
            a.requires_grad_(True)
            loss = ((net.encode(a)["code"] - f0["code"]) ** 2).flatten(1).sum(1).mean()
            (g,) = torch.autograd.grad(loss, a)
            a = project(a.detach() + step * g.sign(), clean)
        with torch.no_grad():
            ref = (net.encode(a)["mid"] - f0["mid"]).flatten(1)
            ref = ref / (ref.norm(dim=1, keepdim=True) + 1e-12)
        a = project(clean + torch.empty_like(clean).uniform_(-eps, eps), clean)
        for _ in range(ila_steps):
            a.requires_grad_(True)
            proj = ((net.encode(a)["mid"] - f0["mid"]).flatten(1) * ref).sum(1)
            (g,) = torch.autograd.grad(proj.mean(), a)
            a = project(a.detach() + step * g.sign(), clean)
        out[s : s + batch] = (
            (a.detach().permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
        )
    return out


@pytest.mark.parametrize("eps", adv.DEFAULT_EPS)
def test_shape_and_budget(proxy, eps):
    x = crops(3, 112)
    y = adv.LiAEAttack(proxy, device="cpu").craft(x, eps, steps=2, ila_steps=2)
    assert y.shape == x.shape and y.dtype == np.uint8
    assert linf_uint8(y, x) <= math.ceil(eps * 255)
    assert not np.array_equal(y, x)


def test_matches_paper_formulation(proxy):
    x = crops(5, 112)
    got = adv.li_ae_attack(
        x, 0.06, proxy=proxy, device="cpu", steps=3, ila_steps=3, batch_size=2
    )
    want = reference_craft(proxy, x, 0.06, steps=3, ila_steps=3, seed=0, batch=2)
    assert np.array_equal(got, want)


def test_seeded_and_batch_prefix(proxy):
    att = adv.LiAEAttack(proxy, device="cpu")
    x = crops(4, 112)
    a = att.craft(x, 0.03, steps=2, ila_steps=2, batch_size=2, seed=0)
    assert np.array_equal(a, att.craft(x, 0.03, steps=2, ila_steps=2, batch_size=2))
    assert not np.array_equal(
        a, att.craft(x, 0.03, steps=2, ila_steps=2, batch_size=2, seed=1)
    )
    # random starts are drawn batch by batch: a prefix of whole batches is stable
    head = att.craft(x[:2], 0.03, steps=2, ila_steps=2, batch_size=2)
    assert np.array_equal(head, a[:2])


def test_does_not_touch_global_rng(proxy):
    state = torch.random.get_rng_state()
    adv.li_ae_attack(crops(1, 112), 0.03, proxy=proxy, steps=1, ila_steps=1)
    assert torch.equal(state, torch.random.get_rng_state())


def test_proxy_instance_is_frozen():
    torch.manual_seed(2)
    net = ProxyAE(width=4).train()
    att = adv.LiAEAttack(net)
    assert att.net is net and not net.training
    assert att.device == torch.device("cpu")  # the device of the given module
    assert all(not p.requires_grad for p in net.parameters())


def test_registry_dispatch(proxy):
    x = crops(2, 112)
    kw = dict(proxy=proxy, device="cpu", steps=1, ila_steps=1)
    assert np.array_equal(
        adv.craft("liae", x, 0.06, **kw), adv.li_ae_attack(x, 0.06, **kw)
    )
    # the CLIP model is dropped for Li-AE, so one uniform craft call fits all attacks
    assert np.array_equal(
        adv.craft("liae", x, 0.06, model=object(), **kw),
        adv.li_ae_attack(x, 0.06, **kw),
    )


@pytest.mark.gpu
def test_cuda_budget():
    x = crops(2, 112)
    y = adv.li_ae_attack(x, 0.06, proxy=ProxyAE(width=4), device="cuda", steps=2)
    assert y.shape == x.shape and linf_uint8(y, x) <= math.ceil(0.06 * 255)


def test_cuda_blocks_random_starts(proxy):
    """With ``cuda_blocks`` the starts are the emulated CUDA stream, on any device."""
    from face1kb.adversarial import _philox

    x = crops(3, 112)
    eps = 0.06
    y = adv.li_ae_attack(
        x, eps, proxy=proxy, device="cpu", steps=0, ila_steps=0, cuda_blocks=272
    )
    n = x.size
    # stage 2 starts from the second draw of the stream
    v = _philox.cuda_uniform(n, -eps, eps, 0, _philox.offset_increment(n, 272), 272)
    clean = torch.from_numpy(x.astype(np.float32) / 255.0)
    start = clean + torch.from_numpy(v.reshape(x.shape))
    start = torch.min(torch.max(start, clean - eps), clean + eps).clamp(0, 1)
    want = (start.numpy() * 255.0).round().astype(np.uint8)
    assert np.array_equal(y, want)
    z = adv.li_ae_attack(x, eps, proxy=proxy, device="cpu", steps=1, ila_steps=1)
    w = adv.li_ae_attack(
        x, eps, proxy=proxy, device="cpu", steps=1, ila_steps=1, cuda_blocks=272
    )
    assert not np.array_equal(z, w)  # the CPU generator is a different stream
