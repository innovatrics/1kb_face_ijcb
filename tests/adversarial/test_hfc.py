# SPDX-License-Identifier: MIT
"""HFC attack, registry and naming helpers (CPU, NumPy + OpenCV)."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from face1kb import adversarial as adv  # noqa: E402

from ._synthetic import crops  # noqa: E402

# sha256 of hfc_attack(crops(3, 112), eps, seed=0).tobytes(), recorded with the
# paper environment (OpenCV 4.13.0, NumPy 2.4.6).
GOLDEN_OPENCV = "4.13.0"
GOLDEN_SHA256 = {
    0.03: "bd56e1b9b9224dff66723260df296d0d03439035b87f4021f9940e85dbefa4f1",
    0.06: "0dda520129a77c521dc8925e56ddc42d2e5fa217501d96fb1c6eb44aebd380b4",
    0.1: "656f9e1724a96b9af3a2b8b36e64824d8443d3a7777079d129ec4cec64cafbce",
}


def linf_uint8(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.abs(a.astype(int) - b.astype(int)).max())


@pytest.mark.parametrize("eps", adv.DEFAULT_EPS)
def test_shape_dtype_and_budget(eps):
    x = crops(3, 112)
    y = adv.hfc_attack(x, eps)
    assert y.shape == x.shape and y.dtype == np.uint8
    assert linf_uint8(y, x) <= math.ceil(eps * 255)
    assert linf_uint8(y, x) >= math.floor(eps * 255) - 1  # the budget is used
    assert not np.array_equal(y, x)


def test_seeded_and_prefix_stable():
    x = crops(4, 112)
    a = adv.hfc_attack(x, 0.06, seed=0)
    assert np.array_equal(a, adv.hfc_attack(x, 0.06, seed=0))
    assert not np.array_equal(a, adv.hfc_attack(x, 0.06, seed=1))
    # noise is drawn crop by crop: the first k crops do not depend on what follows
    assert np.array_equal(adv.hfc_attack(x[:2], 0.06, seed=0), a[:2])


def test_suppress_zero_injects_only():
    x = crops(1, 64)
    y = adv.hfc_attack(x, 0.03, suppress=0.0)
    d = y.astype(int) - x.astype(int)
    # without suppression every pixel moves by +-eps (up to clipping at 0 / 255)
    inside = (x > 10) & (x < 245)
    assert np.all(np.abs(d[inside]) >= 7)


@pytest.mark.parametrize("eps", sorted(GOLDEN_SHA256))
def test_golden(eps):
    if cv2.__version__ != GOLDEN_OPENCV:
        pytest.skip(f"golden recorded with OpenCV {GOLDEN_OPENCV}")
    y = adv.hfc_attack(crops(3, 112), eps, seed=0)
    assert hashlib.sha256(y.tobytes()).hexdigest() == GOLDEN_SHA256[eps]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        adv.hfc_attack(crops(1, 32)[0], 0.03)  # missing batch axis
    with pytest.raises(ValueError):
        adv.hfc_attack(crops(1, 32).astype(np.float32), 0.03)


def test_registry_and_names():
    assert sorted(adv.ATTACKS) == ["clip", "hfc", "liae"]
    x = crops(2, 32)
    assert np.array_equal(adv.craft("hfc", x, 0.03), adv.hfc_attack(x, 0.03))
    with pytest.raises(ValueError):
        adv.craft("fgsm", x, 0.03)
    # options of the gradient attacks are ignored for HFC, so one call fits all
    blocks = adv.paper_cuda_blocks("kk", "hfc")
    y = adv.craft(
        "hfc", x, 0.03, device="cuda", batch_size=64, cuda_blocks=blocks, model=object()
    )
    assert np.array_equal(y, adv.hfc_attack(x, 0.03))
    with pytest.raises(ValueError):
        adv.craft("hfc", x, 0.03, cuda_blocks=272)
    assert [adv.eps_tag(e) for e in adv.DEFAULT_EPS] == ["003", "006", "010"]
    assert adv.adv_suffix("liae", 0.06) == "_adv_liae_006"
    with pytest.raises(ValueError):
        adv.adv_suffix("fgsm", 0.06)
    assert adv.step_size(0.03) == 0.03 / 4
    assert adv.step_size(0.01) == 1 / 255
    assert adv.step_size(0.06, alpha=0.5) == 0.5
