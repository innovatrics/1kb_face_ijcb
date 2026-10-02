# SPDX-License-Identifier: MIT
"""Released weights: strict loading, metadata, parameter counts, gain ranges (CPU)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from face1kb import config  # noqa: E402
from face1kb.codec import budget as B  # noqa: E402
from face1kb.codec.api import WEIGHTS_FORMAT_VERSION, load_state_dict_file  # noqa: E402
from face1kb.codec.identity_loss import sha256_file  # noqa: E402
from face1kb.codec.variants import build_model  # noqa: E402

pytestmark = pytest.mark.weights

EXPECTED = {
    "fast": {
        "sha256": "244cb42ac0bb9f0469a28b0c975e0678671c1ad2d82f5d238595080878880e5c",
        "params": 1_352_021,
        "step": "1000000",
        "gain": (0.1, 1.0),
        "architecture": "FaceCodecFast",
    },
    "accurate": {
        "sha256": "577525fd07625c29a88b055c94f8fa867763d6cfd20c10a34e9bdc1211adda98",
        "params": 18_712_400,
        "step": "3000000",
        "gain": (0.04, 1.0),
        "architecture": "FaceCodecAccurate",
    },
}


@pytest.fixture(scope="module", params=list(EXPECTED))
def loaded(request):
    variant = request.param
    state, meta = load_state_dict_file(config.weights_path(variant))
    net = build_model(variant)
    net.load_state_dict(state, strict=True)
    return variant, net.eval(), state, meta


def test_file_hash(loaded):
    variant, *_ = loaded
    assert sha256_file(config.weights_path(variant)) == EXPECTED[variant]["sha256"]


def test_metadata(loaded):
    variant, _, _, meta = loaded
    assert meta["face1kb.variant"] == variant
    assert meta["face1kb.format_version"] == WEIGHTS_FORMAT_VERSION
    assert meta["face1kb.step"] == EXPECTED[variant]["step"]
    assert meta["face1kb.architecture"] == EXPECTED[variant]["architecture"]
    assert meta["face1kb.license"] == "CC-BY-NC-SA-4.0"


def test_strict_keys_and_dtypes(loaded):
    variant, net, state, _ = loaded
    assert set(state) == set(net.state_dict())
    assert all(v.dtype in (torch.float32, torch.int32) for v in state.values())


def test_parameter_counts(loaded):
    variant, net, _, _ = loaded
    assert sum(p.numel() for p in net.parameters()) == EXPECTED[variant]["params"]
    if variant == "accurate":
        anchor = sum(p.numel() for p in net.side.anchor.parameters())
        assert anchor == 3_652_520
        assert all(not p.requires_grad for p in net.side.anchor.parameters())


def test_gain_range_and_zqstep(loaded):
    variant, net, _, _ = loaded
    lo, hi = EXPECTED[variant]["gain"]
    table = B.gain_table(net)
    assert abs(table[0] - lo) < 1e-6 and abs(table[-1] - hi) < 1e-6
    assert net.downsampling_factor == 128
    with torch.no_grad():
        g = torch.tensor(table, dtype=torch.float32).view(-1, 1)
        qs = net.lower_bound_zqstep(net.gayn2zqstep(1.0 / g))
    assert torch.all(qs == 0.5)


def test_face1kb_load_cpu(loaded):
    import face1kb

    variant = loaded[0]
    with pytest.warns(UserWarning, match="device-specific"):
        codec = face1kb.load(variant, device="cpu", reproducible=False)
    assert codec.variant == variant
    assert codec.num_parameters == EXPECTED[variant]["params"]
