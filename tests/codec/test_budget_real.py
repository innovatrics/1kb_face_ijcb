# SPDX-License-Identifier: MIT
"""Budget guarantee of the released codecs on synthetic inputs.

With the default ``paper_compat=False`` the container never exceeds the budget, at
all five benchmark resolutions and both budgets. Runs on CUDA when available,
otherwise on CPU (slower; CPU bitstreams differ from CUDA ones but the guarantee is
the same).
"""

from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")

import face1kb  # noqa: E402
from face1kb import config  # noqa: E402
from face1kb.codec import container as C  # noqa: E402

from ._synthetic import noise, smooth_face  # noqa: E402

pytestmark = pytest.mark.weights


@pytest.fixture(scope="module", params=list(config.CODEC_VARIANTS))
def codec(request, device):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return face1kb.load(request.param, device=device)


@pytest.mark.filterwarnings("ignore::face1kb.codec.budget.IdentityOnlyWarning")
@pytest.mark.parametrize("res", config.RESOLUTIONS)
@pytest.mark.parametrize("make", [smooth_face, noise], ids=["smooth", "noise"])
def test_never_exceeds_budget(codec, res, make):
    img = make(res)
    for budget in config.BUDGETS:
        data, info = codec.encode(img, budget, return_info=True)
        assert len(data) <= budget, (codec.variant, res, budget, info)
        assert info["bytes"] == len(data)
        h = C.unpack(data)
        assert h.res == res
        out = codec.decode(data)
        assert out.shape == (res, res, 3) and out.dtype.name == "uint8"
        if info["identity_only"]:
            assert not out.any()  # the identity-only fallback decodes to black


def test_paper_compat_overshoot_is_bounded(codec):
    img = smooth_face(96)
    for budget in config.BUDGETS:
        data, info = codec.encode(img, budget, paper_compat=True, return_info=True)
        if info["fitted"]:
            assert len(data) <= budget + C.RAW_GEOM_TRAILER


def test_bucketed_resolutions_identical_across_modes(codec):
    img = smooth_face(112)
    a = codec.encode(img, 1024, paper_compat=True)
    b = codec.encode(img, 1024)
    assert a == b


def test_input_validation(codec):
    import numpy as np

    with pytest.raises(ValueError):
        codec.encode(np.zeros((112, 96, 3), np.uint8), 1024)
    with pytest.raises(ValueError):
        codec.encode(np.zeros((112, 112, 3), np.float32), 1024)
    with pytest.raises(ValueError):
        codec.encode(np.zeros((112, 112), np.uint8), 1024)
