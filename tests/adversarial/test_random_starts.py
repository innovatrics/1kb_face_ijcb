# SPDX-License-Identifier: MIT
"""Device-independent CUDA random starts (``cuda_blocks``): Philox and uniform_."""

from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np
import pytest

from face1kb import adversarial as adv
from face1kb.adversarial import _philox as P

# Known-answer vectors of Philox-4x32-10 (Random123 kat_vectors).
KAT = [
    ((0, 0, 0, 0), (0, 0), "6627e8d5 e169c58d bc57ac4c 9b00dbd8"),
    (
        (0xFFFFFFFF,) * 4,
        (0xFFFFFFFF, 0xFFFFFFFF),
        "408f276d 41c83b0e a20bc7c6 6d5451fd",
    ),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        "d16cfe09 94fdcceb 5001e420 24126ea1",
    ),
]

# sha256 of cuda_uniform(64 * 3 * 112 * 112, -0.06, 0.06, seed 0, ...) for the
# first draw (offset 0) and the second draw (offset 36) with 272 blocks, and the
# first draw with 288 blocks; equal to torch's CUDA uniform_ on a GeForce RTX 2080 Ti
# and (by construction of the launch geometry) on a 72-SM GPU.
N64 = 64 * 3 * 112 * 112
GOLDEN = {
    (0, 272): "b0d10015d96e3a94e49bb2b574d468d4ac7f763f414daa512ef24c7a39af416a",
    (36, 272): "1dd98b452e51e6065f243b69dca57ac6fa439b3327b27fde2966ad43952ae88b",
    (0, 288): "add2a63a5607114086382024c63bb397846ac85f158b43e588d38cf80cab53fc",
}


@pytest.mark.parametrize("ctr, key, want", KAT)
def test_philox_known_answers(ctr, key, want):
    words = P.philox4x32_10(tuple(np.array([c], np.uint64) for c in ctr), key)[0]
    assert " ".join(f"{int(w):08x}" for w in words) == want


@pytest.mark.parametrize("offset, blocks", sorted(GOLDEN))
def test_cuda_uniform_golden(offset, blocks):
    v = P.cuda_uniform(N64, -0.06, 0.06, 0, offset, blocks)
    assert v.dtype == np.float32 and v.shape == (N64,)
    assert hashlib.sha256(v.tobytes()).hexdigest() == GOLDEN[(offset, blocks)]
    assert v.min() >= np.float32(-0.06) and v.max() < np.float32(0.06)


def test_launch_geometry():
    assert P.launch_threads(N64, 272) == 272 * 256
    assert P.launch_threads(192, 272) == 256  # small tensors use fewer blocks
    assert P.offset_increment(N64, 272) == 36
    assert P.offset_increment(N64, 288) == 36
    assert P.offset_increment(7 * 3 * 112 * 112, 272) == 4
    assert P.offset_increment(0, 272) == 0
    assert P.cuda_uniform(0, -1, 1, 0, 0, 272).size == 0
    with pytest.raises(ValueError):
        P.cuda_uniform(10, -1, 1, 0, 2, 272)


def test_element_mapping():
    """Element t + T * (4k + i) takes word i of the k-th Philox call of thread t."""
    blocks, n = 1, 256 * 4 * 2 + 5  # T = 256 threads, 3 calls (the last partial)
    v = P.cuda_uniform(n, 0.0, 1.0, 7, 8, blocks)
    for e in (0, 1, 255, 256, 1023, 1024, 2047, 2048, n - 1):
        t, q = e % 256, e // 256
        k, i = q // 4, q % 4
        ctr = (np.uint64(2 + k), np.uint64(0), np.uint64(t), np.uint64(0))
        w = P.philox4x32_10(ctr, (7, 0))[i]
        u = np.float32(w) * np.float32(2.0**-32) + np.float32(2.0**-33)
        assert v[e] == P.fma_f32(np.array([u]), np.float32(1.0), np.float32(0.0))[0]


def _exact_f32(a, b, c) -> np.float32:
    """Correctly rounded float32 of a * b + c, with rational arithmetic."""
    x = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
    lo = np.float32(float(x))  # a float32 neighbour (float() rounds to float64)
    cands = [
        np.nextafter(lo, np.float32(-np.inf)),
        lo,
        np.nextafter(lo, np.float32(np.inf)),
    ]
    dist = [abs(Fraction(float(y)) - x) for y in cands]
    best = min(dist)
    ties = [y for y, d in zip(cands, dist) if d == best]
    if len(ties) == 1:
        return ties[0]
    return min(ties, key=lambda y: int(np.float32(y).view(np.uint32)) & 1)  # even


def test_fma_is_correctly_rounded():
    rng = np.random.default_rng(0)
    a = rng.random(2000).astype(np.float32)
    for b, c in (
        (np.float32(0.12), np.float32(-0.06)),
        (np.float32(3.0), np.float32(1)),
    ):
        got = P.fma_f32(a, b, c)
        want = np.array([_exact_f32(x, b, c) for x in a], np.float32)
        assert np.array_equal(got, want)


def test_fma_double_rounding_case():
    # a * b + c = (1 + 2**-23 + 2**-24) - 2**-70: just below a float32 midpoint whose
    # upper neighbour is even. float64 rounds the sum onto the midpoint, and rounding
    # that to float32 (ties to even) goes up; the single-rounding result is down.
    a = np.array([1.0 + 2.0**-23], np.float32)
    b = np.float32(2.0**-24 * (1.0 - 2.0**-23))
    c = np.float32(1.0 + 2.0**-23)
    naive = (a.astype(np.float64) * np.float64(b) + np.float64(c)).astype(np.float32)
    want = _exact_f32(a[0], b, c)
    assert want == c
    assert naive[0] == np.nextafter(c, np.float32(2.0))  # double rounding
    assert P.fma_f32(a, b, c)[0] == want
    # the mirrored case: 1 + 2**-24 + 2**-60, just above the midpoint between 1.0
    # (even) and its upper neighbour
    a2 = np.array([1.0 + 2.0**-12], np.float32)
    b2 = np.float32(2.0**-24 * (1.0 - 4095 * 2.0**-24))
    c2 = np.float32(1.0)
    naive2 = (a2.astype(np.float64) * np.float64(b2) + np.float64(c2)).astype(
        np.float32
    )
    want2 = _exact_f32(a2[0], b2, c2)
    assert want2 == np.nextafter(c2, np.float32(2.0))
    assert naive2[0] == c2  # double rounding
    assert P.fma_f32(a2, b2, c2)[0] == want2


def test_paper_blocks():
    assert adv.paper_cuda_blocks("colorferet", "liae") == 272
    assert adv.paper_cuda_blocks("kk", "liae") == 272
    assert adv.paper_cuda_blocks("colorferet", "clip") == 272
    assert adv.paper_cuda_blocks("kk", "clip") == 288
    assert adv.paper_cuda_blocks("kk", "hfc") is None
    with pytest.raises(ValueError):
        adv.paper_cuda_blocks("kk", "fgsm")
    with pytest.raises(ValueError):
        adv.paper_cuda_blocks("lfw", "clip")


def test_random_starts_layout_cpu():
    torch = pytest.importorskip("torch")
    from face1kb.adversarial.attacks import _RandomStarts

    n, h, w = 2, 5, 6
    clean = torch.zeros(n, h, w, 3).permute(0, 3, 1, 2)  # channels-last, as crafted
    st = _RandomStarts(3, torch.device("cpu"), cuda_blocks=272)
    a = st(clean, 0.1)
    v = P.cuda_uniform(clean.numel(), -0.1, 0.1, 3, 0, 272)
    assert torch.equal(a.permute(0, 2, 3, 1).reshape(-1), torch.from_numpy(v))
    b = st(clean, 0.1)  # the offset advances like torch's generator
    v2 = P.cuda_uniform(clean.numel(), -0.1, 0.1, 3, 4, 272)
    assert torch.equal(b.permute(0, 2, 3, 1).reshape(-1), torch.from_numpy(v2))
    c = st(torch.zeros(n, 3, h, w), 0.1)  # contiguous NCHW: memory order too
    v3 = P.cuda_uniform(clean.numel(), -0.1, 0.1, 3, 8, 272)
    assert torch.equal(c.reshape(-1), torch.from_numpy(v3))
    with pytest.raises(ValueError):
        st(torch.zeros(n, 3, h, 2 * w)[..., ::2], 0.1)
    with pytest.raises(ValueError):
        _RandomStarts(0, torch.device("cpu"), cuda_blocks=0)


@pytest.mark.gpu
def test_emulation_equals_torch_cuda():
    torch = pytest.importorskip("torch")
    from face1kb.adversarial.attacks import _RandomStarts

    dev = torch.device("cuda")
    blocks = adv.cuda_launch_blocks()
    for shape in ((64, 112, 112, 3), (7, 112, 112, 3), (1, 8, 8, 3)):
        base = torch.zeros(shape, device=dev).permute(0, 3, 1, 2)
        for seed in (0, 12345):
            gen = torch.Generator(device=dev).manual_seed(seed)
            st = _RandomStarts(seed, dev, cuda_blocks=blocks)
            for eps in (0.03, 0.1):
                want = base + torch.empty_like(base).uniform_(-eps, eps, generator=gen)
                assert torch.equal(st(base, eps), want)
