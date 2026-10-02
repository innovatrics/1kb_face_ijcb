# SPDX-License-Identifier: MIT
"""Budget search logic on a stand-in network with a known rate curve (CPU)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from face1kb.codec import budget as B  # noqa: E402
from face1kb.codec import container as C  # noqa: E402


class FakeNet:
    """Minimal codec stand-in: payload size grows linearly with the gain."""

    VARIANT_ID = 0
    downsampling_factor = 128

    def __init__(self, y_scale: float = 1000.0, z_len: int = 40, lo=0.1, hi=1.0):
        self.Gain = torch.tensor([lo, 0.3, 0.6, hi], dtype=torch.float32)
        self.y_scale = y_scale
        self.z_len = z_len
        self.updates = 0
        self.calls = []

    def update_for_gain(self, gain, force=True):
        self.updates += 1
        return True

    def compress(self, x, inputscale=0, res=None):
        self.calls.append(inputscale)
        y = b"y" * int(self.y_scale * inputscale)
        return {"strings": [[y], [b"z" * self.z_len]], "shape": (1, 1)}


def _x(res):
    return torch.zeros(1, 3, 128 * ((res + 127) // 128), 128 * ((res + 127) // 128))


def _size(net, g):
    return int(net.y_scale * g) + net.z_len


def test_gain_table():
    net = FakeNet(lo=0.04, hi=1.0)
    t = B.gain_table(net)
    assert len(t) == B.N_RATE_LEVELS == 64
    lo, hi = float(torch.tensor(0.04)), 1.0
    assert t[0] == lo and abs(t[-1] - hi) < 1e-12
    assert all(a < b for a, b in zip(t, t[1:]))
    assert t[10] == lo * (hi / lo) ** (10 / 63)


@pytest.mark.parametrize("budget", [100, 300, 512, 700, 1024])
@pytest.mark.parametrize("res", [64, 96, 112, 168, 224])
@pytest.mark.parametrize("paper_compat", [True, False])
def test_search_picks_largest_fitting_gain(budget, res, paper_compat):
    net = FakeNet()
    table = B.gain_table(net)
    target = budget - B.budget_overhead(res, 0, paper_compat)
    expected = max(
        (i for i, g in enumerate(table) if _size(net, g) <= target), default=None
    )
    data, info = B.encode_to_budget(
        net, _x(res), budget, res, paper_compat=paper_compat, overflow="floor"
    )
    if expected is None:
        assert info["over_budget"] and not info["fitted"]
        assert info["rate_index"] == 0
    else:
        assert info["fitted"] and info["rate_index"] == expected
    assert len(net.calls) <= 7  # binary search over 64 levels
    h = C.unpack(data)
    assert h.rate_index == info["rate_index"] and h.res == res
    assert info["bytes"] == len(data)


def test_paper_accounting_ignores_raw_geometry_trailer():
    # Find a budget at 96 px where the paper accounting overshoots by the trailer.
    net = FakeNet()
    overshoot = []
    for budget in range(200, 1100):
        paper, info = B.encode_to_budget(net, _x(96), budget, 96, paper_compat=True)
        fixed, _ = B.encode_to_budget(net, _x(96), budget, 96)
        assert len(fixed) <= budget
        assert len(paper) <= budget + C.RAW_GEOM_TRAILER
        assert info["within_budget"] == (len(paper) <= budget)
        if len(paper) > budget:
            overshoot.append(budget)
            # A stream that fit the search target, but not the budget.
            assert info["fitted"] and not info["over_budget"]
    assert overshoot, "expected paper accounting to overshoot at some budgets"
    # Bucketed resolutions are unaffected by the flag.
    for budget in (300, 512, 1024):
        a, _ = B.encode_to_budget(net, _x(112), budget, 112, paper_compat=True)
        b, _ = B.encode_to_budget(net, _x(112), budget, 112, paper_compat=False)
        assert a == b


def test_overflow_policies():
    net = FakeNet(y_scale=5000.0, z_len=400)  # even the lowest gain needs > 900 B
    x = _x(224)
    floor, info = B.encode_to_budget(net, x, 512, 224, overflow="floor")
    assert info["over_budget"] and len(floor) > 512
    ident, info = B.encode_to_budget(net, x, 512, 224)
    assert info["identity_only"] and not info["over_budget"] and len(ident) <= 512
    assert C.unpack(ident).variant_id == C.VARIANT_IDENTITY_ONLY
    paper, info = B.encode_to_budget(net, x, 512, 224, paper_compat=True)
    assert paper == floor and info["over_budget"]
    with pytest.raises(B.BudgetError):
        B.encode_to_budget(net, x, 512, 224, overflow="error")
    with pytest.raises(ValueError):
        B.encode_to_budget(net, x, 512, 224, overflow="nope")


@pytest.mark.parametrize("paper_compat", [True, False])
def test_search_respects_the_string_length_limit(paper_compat):
    # Large crop, huge budget: the budget alone would allow y strings > 65535 B.
    net = FakeNet(y_scale=200_000.0)
    table = B.gain_table(net)
    expected = max(
        i for i, g in enumerate(table) if int(net.y_scale * g) <= C.MAX_STRING_BYTES
    )
    data, info = B.encode_to_budget(
        net, _x(1024), 10**7, 1024, paper_compat=paper_compat
    )
    assert info["fitted"] and info["rate_index"] == expected
    assert len(C.unpack(data).y_string) <= C.MAX_STRING_BYTES
    assert expected < B.N_RATE_LEVELS - 1  # the limit, not the budget, binds


def test_string_length_limit_overflow_policies():
    net = FakeNet(y_scale=2_000_000.0)  # even the lowest gain needs > 65535 B
    x = _x(1024)
    _, info = B.encode_to_budget(net, x, 10**7, 1024)
    assert info["identity_only"]
    with pytest.raises(B.BudgetError, match="container limit"):
        B.encode_to_budget(net, x, 10**7, 1024, overflow="error")
    with pytest.raises(ValueError, match="length prefix"):
        B.encode_to_budget(net, x, 10**7, 1024, overflow="floor")


def test_tiny_budget_paper_compat_gives_identity_only():
    net = FakeNet()
    data, info = B.encode_to_budget(
        net, _x(112), 7, 112, sidechannel=b"12345678", paper_compat=True
    )
    assert info["identity_only"] and C.unpack(data).sidechannel == b"12345678"
    assert not net.calls
    with pytest.raises(B.BudgetError):
        B.encode_to_budget(
            net,
            _x(112),
            7,
            112,
            sidechannel=b"12345678",
            paper_compat=True,
            overflow="error",
        )


@pytest.mark.parametrize("res", [64, 96, 112, 168, 224])
@pytest.mark.parametrize("sc", [b"", b"12345678"])
def test_minimum_budget(res, sc):
    net = FakeNet()
    smallest = B.min_container_bytes(res, len(sc))
    raw = C.RAW_GEOM_TRAILER if C.needs_raw_geometry(res) else 0
    assert smallest == 3 + len(sc) + raw
    for budget in (1, smallest - 1):
        if budget < 1:
            continue
        with pytest.raises(ValueError):
            B.encode_to_budget(net, _x(res), budget, res, sidechannel=sc)
    # From the minimum up to the first budget with room for a spatial latent, the
    # default emits the identity-only container, which always fits.
    for budget in range(smallest, smallest + 6):
        data, info = B.encode_to_budget(net, _x(res), budget, res, sidechannel=sc)
        assert len(data) <= budget
        assert info["identity_only"] or info["fitted"]
    for budget in (0, -5):
        for paper_compat in (True, False):
            with pytest.raises(ValueError):
                B.encode_to_budget(
                    net, _x(res), budget, res, sidechannel=sc, paper_compat=paper_compat
                )


@pytest.mark.parametrize("res", [64, 96, 112, 168, 224])
def test_default_never_exceeds_budget_on_fake_net(res):
    for net in (FakeNet(), FakeNet(y_scale=5000.0, z_len=400)):
        for budget in range(B.min_container_bytes(res, 8), 1100, 7):
            data, info = B.encode_to_budget(
                net, _x(res), budget, res, sidechannel=b"12345678"
            )
            assert len(data) <= budget, (res, budget, info)
            assert info["within_budget"]


def test_decode_identity_only_returns_none():
    net = FakeNet()
    buf = C.pack(C.VARIANT_IDENTITY_ONLY, 0, 112, b"abc")
    x_hat, h = B.decode_from_container(net, buf, device="cpu")
    assert x_hat is None and h.res == 112


def test_budget_overhead():
    assert B.budget_overhead(112, 0) == 7
    assert B.budget_overhead(112, 8) == 15
    assert B.budget_overhead(96, 8) == 19
    assert B.budget_overhead(96, 8, paper_compat=True) == 15
    assert B.budget_overhead(168, 0) == 11


def test_codec_encode_warns_on_identity_only():
    import warnings

    import numpy as np

    from face1kb.codec.api import Codec

    img = np.zeros((112, 112, 3), np.uint8)
    codec = Codec(FakeNet(y_scale=5000.0, z_len=400), "fast", torch.device("cpu"))
    with pytest.warns(B.IdentityOnlyWarning):
        data, info = codec.encode(img, 512, return_info=True)
    assert info["identity_only"] and len(data) == 3
    codec = Codec(FakeNet(), "fast", torch.device("cpu"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", B.IdentityOnlyWarning)
        data, info = codec.encode(img, 1024, return_info=True)
    assert info["fitted"]
    with pytest.raises(ValueError):
        codec.encode(img, 2)
