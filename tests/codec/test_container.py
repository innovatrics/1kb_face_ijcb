# SPDX-License-Identifier: MIT
"""Container format: layout, round trips and error handling (pure Python)."""

from __future__ import annotations

import random
import struct

import pytest

from face1kb.codec import container as C

RESOLUTIONS = (64, 96, 112, 128, 168, 192, 224, 256)


def _payload(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(256) for _ in range(n))


@pytest.mark.parametrize("res", RESOLUTIONS)
@pytest.mark.parametrize("variant", [C.VARIANT_FAST, C.VARIANT_ACCURATE])
@pytest.mark.parametrize("sc_len", [0, 8, 255])
def test_round_trip(res, variant, sc_len):
    rng = random.Random(res * 1000 + variant * 300 + sc_len)
    sc = _payload(rng, sc_len)
    y = _payload(rng, rng.randrange(0, 900))
    z = _payload(rng, rng.randrange(1, 220))
    idx = rng.randrange(64)
    buf = C.pack(variant, idx, res, sc, y, z)
    h = C.unpack(buf)
    assert (h.variant_id, h.rate_index, h.res) == (variant, idx, res)
    assert (h.sidechannel, h.y_string, h.z_string) == (sc, y, z)
    assert len(buf) == C.packed_size(res, sc_len, len(y), len(z))


@pytest.mark.parametrize("res", RESOLUTIONS)
def test_identity_only(res):
    sc = bytes(range(8))
    buf = C.pack(C.VARIANT_IDENTITY_ONLY, 0, res, sc)
    raw = 4 if C.needs_raw_geometry(res) else 0
    assert len(buf) == 3 + len(sc) + raw
    h = C.unpack(buf)
    assert h.variant_id == C.VARIANT_IDENTITY_ONLY
    assert h.res == res and h.sidechannel == sc and h.y_string == h.z_string == b""
    assert len(buf) == C.packed_size(res, len(sc), 0, 0, identity_only=True)


def test_header_bytes():
    buf = C.pack(C.VARIANT_ACCURATE, 37, 112, b"S" * 8, b"yy", b"z")
    assert buf[0] == (1 << 6) | 37
    assert buf[1] == C.RES_BUCKETS[112] == 4
    assert buf[2] == 8
    assert buf[3:11] == b"S" * 8
    assert buf[11:13] == struct.pack(">H", 2) and buf[13:15] == b"yy"
    assert buf[15:17] == struct.pack(">H", 1) and buf[17:18] == b"z"
    assert len(buf) == 18


def test_fixed_overhead_is_seven_bytes():
    assert C.HEADER_OVERHEAD == 7
    assert len(C.pack(C.VARIANT_FAST, 0, 112)) == 7
    assert len(C.pack(C.VARIANT_FAST, 0, 224, b"", b"a" * 10, b"b" * 5)) == 22


@pytest.mark.parametrize("res", [96, 168, 80, 300])
def test_raw_geometry_trailer(res):
    assert C.needs_raw_geometry(res)
    buf = C.pack(C.VARIANT_FAST, 5, res, b"", b"abc", b"de")
    assert buf[1] == C.RAW_GEOM
    assert buf[-4:] == struct.pack(">HH", res, res)
    assert len(buf) == 7 + 5 + C.RAW_GEOM_TRAILER
    assert C.unpack(buf).res == res


def test_bucketed_resolutions_have_no_trailer():
    for res in C.RES_BUCKETS:
        assert not C.needs_raw_geometry(res)
        assert len(C.pack(C.VARIANT_FAST, 0, res)) == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variant_id": 0, "rate_index": 64, "res": 112},
        {"variant_id": 4, "rate_index": 0, "res": 112},
        {"variant_id": 1, "rate_index": 0, "res": 112, "sidechannel": b"x" * 256},
    ],
)
def test_pack_rejects_bad_fields(kwargs):
    with pytest.raises(ValueError):
        C.pack(**kwargs)


@pytest.mark.parametrize("which", ["y_string", "z_string"])
def test_pack_rejects_strings_longer_than_the_length_prefix(which):
    ok = {"y_string": b"y" * 10, "z_string": b"z" * 10}
    buf = C.pack(C.VARIANT_FAST, 0, 1024, **{**ok, which: b"a" * C.MAX_STRING_BYTES})
    assert len(C.unpack(buf).__dict__[which]) == C.MAX_STRING_BYTES
    with pytest.raises(ValueError, match="length prefix"):
        C.pack(C.VARIANT_FAST, 0, 1024, **{**ok, which: b"a" * 65536})


def test_unpack_rejects_truncated_and_unknown():
    buf = C.pack(C.VARIANT_FAST, 3, 112, b"", b"a" * 20, b"b" * 7)
    with pytest.raises(ValueError):
        C.unpack(buf[:-1])
    with pytest.raises(ValueError):
        C.unpack(buf[:2])
    bad = bytearray(buf)
    bad[1] = 17  # no such bucket
    with pytest.raises(ValueError):
        C.unpack(bytes(bad))


def test_variant_names():
    assert C.VARIANT_NAMES[C.VARIANT_FAST] == "fast"
    assert C.VARIANT_NAMES[C.VARIANT_ACCURATE] == "accurate"
    assert C.VARIANT_NAMES[C.VARIANT_IDENTITY_ONLY] == "identity-only"


@pytest.mark.parametrize("res", [112, 168])
@pytest.mark.parametrize("variant", [C.VARIANT_FAST, C.VARIANT_ACCURATE, 2])
def test_unpack_rejects_every_truncation(res, variant):
    sc = b"" if variant == C.VARIANT_FAST else b"s" * 8
    y, z = (b"", b"") if variant == 2 else (b"y" * 21, b"z" * 6)
    buf = C.pack(variant, 9, res, sc, y, z)
    assert C.unpack(buf).res == res
    for n in range(len(buf)):
        with pytest.raises(ValueError):
            C.unpack(buf[:n])


@pytest.mark.parametrize("res", [112, 168])
def test_unpack_rejects_trailing_bytes(res):
    buf = C.pack(C.VARIANT_FAST, 3, res, b"", b"a" * 12, b"b" * 4)
    for extra in (b"\x00", b"\x00\x00", b"junk"):
        with pytest.raises(ValueError, match="trailing"):
            C.unpack(buf + extra)


def test_unpack_errors_are_value_errors_not_struct_errors():
    buf = C.pack(C.VARIANT_FAST, 3, 96, b"", b"a" * 30, b"b" * 8)
    for n in (3, 4, 5, 6, 20, len(buf) - 2):
        try:
            C.unpack(buf[:n])
        except ValueError:
            pass
        except struct.error as e:  # pragma: no cover - the regression
            pytest.fail(f"struct.error for a {n} B prefix: {e}")
