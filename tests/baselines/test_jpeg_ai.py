# SPDX-License-Identifier: MIT
"""JPEG-AI: budget-fit logic and driver behaviour without the reference software.

The encoder is replaced by a synthetic size model, so these tests need neither the
reference software nor a GPU (``test_jpeg_ai_real`` runs the real codec).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from face1kb.baselines import jpeg_ai
from face1kb.baselines.jpeg_ai import (
    FAILED_SIZE,
    FitResult,
    analytic_fit,
    analytic_guess,
    hinted_fit,
    step_down_fit,
)


def _ref_analytic_fit(encode, guess, budget, lo, hi):
    # The benchmark's predict-then-correct search, written out as the reference.
    bpp, size, data = encode(guess)
    best = (bpp, size, data) if size <= budget else None
    tried = {bpp}
    for _ in range(2):
        if best is not None:
            break
        nxt = int(max(lo, min(hi, round(bpp * budget / max(size, 1) * 0.92))))
        if nxt in tried:
            nxt -= 2
        if nxt < lo or nxt in tried:
            break
        tried.add(nxt)
        bpp, size, data = encode(nxt)
        if size <= budget:
            best = (bpp, size, data)
    if best is not None:
        return best[0], best[1], best[2], True
    return bpp, size, data, False


def _model(pixels, gain=1.0, overhead=40, fail=()):
    """Synthetic encoder: size ~ bpp * pixels / 800 * gain + overhead."""
    calls = []

    def encode(bpp):
        b = int(max(2, min(50, round(bpp))))
        calls.append(b)
        if b in fail:
            return b, FAILED_SIZE, b""
        size = int(b * pixels / 800 * gain) + overhead
        return b, size, bytes([b % 256]) * size

    return encode, calls


def test_guess_formula():
    assert analytic_guess(1024, 112 * 112) == pytest.approx(65.306, abs=1e-3)
    assert analytic_guess(1024, 224 * 224) == pytest.approx(16.327, abs=1e-3)
    assert analytic_guess(512, 64 * 64) == 100.0


def test_single_encode_when_the_guess_fits():
    enc, calls = _model(112 * 112)
    r = analytic_fit(enc, analytic_guess(1024, 112 * 112), 1024, 2, 50)
    assert calls == [50] and r.fitted and r.setting == 50


def test_correction_after_overshoot():
    enc, calls = _model(224 * 224, gain=1.3)
    r = analytic_fit(enc, analytic_guess(1024, 224 * 224), 1024, 2, 50)
    assert calls[0] == 16 and len(calls) == 2 and r.fitted
    assert calls[1] == round(16 * 1024 / (int(16 * 224 * 224 / 800 * 1.3) + 40) * 0.92)


@pytest.mark.parametrize("seed", range(300))
def test_matches_reference(seed):
    rng = random.Random(seed)
    res = rng.choice([64, 96, 112, 168, 224])
    budget = rng.choice([512, 768, 960, 1024])
    fail = {rng.randint(2, 50) for _ in range(rng.randint(0, 3))}
    enc, _ = _model(res * res, gain=rng.uniform(0.5, 3.0), fail=fail)
    guess = analytic_guess(budget, res * res)
    assert tuple(analytic_fit(enc, guess, budget, 2, 50)) == _ref_analytic_fit(
        enc, guess, budget, 2, 50
    )


def test_step_down_fit():
    def enc_factory(pixels, gain):
        calls = []

        def enc(bpp):
            calls.append(bpp)
            size = int(bpp * pixels / 800 * gain) + 40
            return size, b"x" * size

        return enc, calls

    enc, calls = enc_factory(224 * 224, 1.5)
    r = step_down_fit(enc, 1024, 224 * 224)
    assert calls[0] == 16 and r.fitted and r.size <= 1024
    # the loop stops when the next target would be below 2; the last encode is kept
    enc, calls = enc_factory(112 * 112, 50.0)
    r = step_down_fit(enc, 1024, 112 * 112)
    assert not r.fitted and calls == [50] and r.setting == 50
    assert step_down_fit(lambda b: None, 1024, 112 * 112) is None


def test_hinted_fit():
    def enc(bpp):
        size = bpp * 60
        return size, b"x" * size

    r = hinted_fit(enc, 1024, 224 * 224)
    assert r == FitResult(16, 960, b"x" * 960, True)
    r = hinted_fit(enc, 1024, 224 * 224, hint=12)
    assert r.setting == 12
    # a stream below the plausibility floor is not accepted
    assert hinted_fit(lambda b: (100, b"x" * 100), 1024, 224 * 224) is None


def test_cfg_files():
    assert jpeg_ai.cfg_files() == ["cfg/tools_off.json", "cfg/profiles/high.json"]
    assert jpeg_ai.cfg_files("sop") == [
        "cfg/tools_off.json",
        "cfg/profiles/simple.json",
    ]
    assert jpeg_ai.cfg_files("base", tools_off=False) == ["cfg/profiles/base.json"]
    assert jpeg_ai.cfg_files(None, tools_off=False) == []
    with pytest.raises(ValueError):
        jpeg_ai.cfg_files("xop")


def test_constants():
    assert jpeg_ai.REFERENCE_COMMIT.startswith("e9648f9")
    assert jpeg_ai.REFERENCE_URL.startswith("https://gitlab.com/wg1/jpeg-ai/")
    assert jpeg_ai.BPP_RANGE == (2, 50)
    assert jpeg_ai.MIN_PLAUSIBLE_FRAC == 0.25
    assert jpeg_ai.DEFAULT_PROFILE == "hop"


def test_check_setup_on_empty_dir(tmp_path):
    problems = jpeg_ai.check_setup(tmp_path)
    assert problems and "no JPEG-AI reference software" in problems[0]
    assert not jpeg_ai.available(tmp_path)
    ref = jpeg_ai.ReferenceSoftware(tmp_path)
    with pytest.raises(jpeg_ai.JpegAIError):
        ref.encode_to_budget(np.zeros((64, 64, 3), np.uint8), 1024)


class _FakeReference(jpeg_ai.ReferenceSoftware):
    """Driver whose reference-software call is a synthetic size model."""

    def __init__(self, sizes):
        super().__init__(".")
        self.sizes = sizes
        self.calls = []

    def require(self):
        pass

    def _encode_png(self, png, td, bpp, profile, tools_off, return_recon):
        self.calls.append((bpp, profile, tools_off))
        size = self.sizes(bpp)
        return None if size is None else b"j" * size


def test_driver_fit_and_floor():
    img = np.zeros((224, 224, 3), np.uint8)
    ref = _FakeReference(lambda b: b * 60)
    data, info = ref.encode_to_budget(img, 1024)
    assert info == {"fitted": True, "setting": 16, "size": 960} and len(data) == 960
    assert ref.calls == [(16, "hop", True)]
    # a truncated stream (< 25 % of the budget) is a failed encode, not a fit; the
    # correction retries at the lowest target
    sizes = {16: 100, 2: 300}
    ref = _FakeReference(lambda b: sizes.get(b, 5000))
    data, info = ref.encode_to_budget(img, 1024)
    assert [c[0] for c in ref.calls] == [16, 2]
    assert info["setting"] == 2 and info["fitted"] and len(data) == 300
    # nothing plausible at all -> error instead of an empty file
    ref = _FakeReference(lambda b: 10)
    with pytest.raises(jpeg_ai.JpegAIError):
        ref.encode_to_budget(img, 1024)
    # over budget at the lowest target -> the stream is kept, fitted=False
    ref = _FakeReference(lambda b: 2000)
    data, info = ref.encode_to_budget(img, 1024, profile="sop")
    assert not info["fitted"] and len(data) == 2000
    assert all(c[1] == "sop" for c in ref.calls)


# ---------------------------------------------------------------- bitstreams
def _expgolomb_k0(value: int) -> bytes:
    n = (value + 1).bit_length() - 1
    bits = "0" * n + format(value + 1, "b")
    bits += "0" * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big")


def _substream(marker: int, payload: bytes) -> bytes:
    return marker.to_bytes(2, "big") + _expgolomb_k0(len(payload)) + payload


def _stream(*subs: bytes) -> bytes:
    return b"\xff\x80" + b"".join(subs) + b"\xff\x81"


def test_expgolomb_helper():
    # unsigned exp-Golomb k=0 as written by the reference encoder
    assert _expgolomb_k0(0) == b"\x80"
    assert _expgolomb_k0(1) == b"\x40"
    assert _expgolomb_k0(300) == bytes([0b00000000, 0b10010110, 0b10000000])


@pytest.mark.parametrize("sizes", [(0,), (5, 0, 17), (300, 1, 4096)])
def test_check_bitstream_accepts_complete_streams(sizes):
    markers = [0xFF82, 0xFF83, 0xFF88, 0xFF89, 0xFF8A]
    subs = [_substream(m, bytes([0xFF]) * n) for m, n in zip(markers, sizes)]
    data = _stream(*subs)
    jpeg_ai.check_bitstream(data)
    jpeg_ai.check_bitstream(data + b"trailing")  # ignored after EOC, as upstream


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\xff",
        b"\xff\x80",
        b"\xff\x81",
        bytes(range(256)) * 3,
        b"\xff\x80\xff\x88\x80\xff\x81",  # first substream is not the header
        b"\xff\x80\xff\x90\x80\xff\x81",  # unknown marker
        b"\xff\x80\xff\x82\x00",  # size never terminates
    ],
)
def test_check_bitstream_rejects_malformed(data):
    with pytest.raises(jpeg_ai.JpegAIError):
        jpeg_ai.check_bitstream(data)


def test_check_bitstream_rejects_every_truncation():
    data = _stream(_substream(0xFF82, b"h" * 12), _substream(0xFF88, b"z" * 40))
    jpeg_ai.check_bitstream(data)
    for n in range(len(data)):
        with pytest.raises(jpeg_ai.JpegAIError):
            jpeg_ai.check_bitstream(data[:n])


def test_decode_refuses_malformed_input_without_the_decoder(tmp_path):
    # The reference decoder loops forever on such input; it must never be reached.
    ref = jpeg_ai.ReferenceSoftware(tmp_path / "no-checkout")

    def boom(*args, **kwargs):
        raise AssertionError("reference decoder called")

    ref._coder = boom
    good = _stream(_substream(0xFF82, b"h" * 3))
    for bad in (b"", good[:-1], good[: len(good) // 2], bytes(range(256))):
        assert ref.decode(bad, workdir=tmp_path) is None
    with pytest.raises(AssertionError, match="reference decoder called"):
        ref.decode(good, workdir=tmp_path)
