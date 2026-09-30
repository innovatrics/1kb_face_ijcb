# SPDX-License-Identifier: MIT
"""Bitstream container of the face1kb codecs (pack / unpack).

Layout (big-endian; everything before the entropy-coded payloads is fixed-size):

    byte 0     flags     = (variant_id << 6) | rate_index   # variant 0..3, rate 0..63
    byte 1     res_bucket                                    # -> side length; 255 = raw
    byte 2     sc_len    = identity side-channel length in bytes (0 if none)
    [sc_len B] identity side-channel payload                 # ACCURATE only
    [2 B]      uint16 len(y_string)                          # absent if identity-only
    [len B]    y_string
    [2 B]      uint16 len(z_string)                          # absent if identity-only
    [len B]    z_string
    [4 B]      uint16 H, uint16 W                            # only if res_bucket == 255

``variant_id``: 0 = FAST, 1 = ACCURATE, 2 = identity-only fallback (no spatial
latent), 3 = reserved (never emitted).

The fixed overhead excluding payloads is 7 B (2 header bytes, 1 side-channel length
byte, 2 + 2 latent-length prefixes). rANS streams are not self-delimiting, so the two
CompressAI strings are explicitly length-prefixed. Resolutions without a bucket code
(for example 96 and 168 px) additionally carry the 4-byte raw-geometry trailer. There
is no CRC: integer rANS decoding is deterministic.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

#: Square side lengths with a 1-byte bucket code.
RES_BUCKETS: dict[int, int] = {64: 0, 128: 1, 192: 2, 256: 3, 112: 4, 224: 5}
BUCKET_TO_RES: dict[int, int] = {v: k for k, v in RES_BUCKETS.items()}
#: Bucket code meaning "geometry stored in the 4-byte trailer".
RAW_GEOM = 255

VARIANT_FAST = 0
VARIANT_ACCURATE = 1
VARIANT_IDENTITY_ONLY = 2
VARIANT_RAW_GEOM = 3

#: Variant id -> human-readable name.
VARIANT_NAMES: dict[int, str] = {
    VARIANT_FAST: "fast",
    VARIANT_ACCURATE: "accurate",
    VARIANT_IDENTITY_ONLY: "identity-only",
    VARIANT_RAW_GEOM: "reserved",
}

#: Fixed bytes excluding payloads: 2 header + 1 sc-len + 2 + 2 length prefixes.
HEADER_OVERHEAD = 7
#: Size of the raw-geometry trailer (uint16 H, uint16 W).
RAW_GEOM_TRAILER = 4
#: Longest rANS string the uint16 length prefix can hold.
MAX_STRING_BYTES = 0xFFFF


@dataclass
class Header:
    """Parsed container fields."""

    variant_id: int
    rate_index: int
    res: int  # true side length (square crop)
    sidechannel: bytes
    y_string: bytes
    z_string: bytes


def _res_to_bucket(res: int) -> int:
    return RES_BUCKETS.get(res, RAW_GEOM)


def needs_raw_geometry(res: int) -> bool:
    """Return True when ``res`` has no bucket code and needs the 4-byte trailer."""
    return _res_to_bucket(res) == RAW_GEOM


def pack(
    variant_id: int,
    rate_index: int,
    res: int,
    sidechannel: bytes = b"",
    y_string: bytes = b"",
    z_string: bytes = b"",
) -> bytes:
    """Serialise one compressed image to the container bytes.

    Parameters
    ----------
    variant_id
        Codec variant in ``[0, 3]``.
    rate_index
        Index into the model's 64-entry gain table, in ``[0, 63]``.
    res
        Square side length of the encoded crop.
    sidechannel
        Identity side-channel payload (at most 255 B; empty for FAST).
    y_string, z_string
        rANS strings of the spatial latent and the hyperprior latent, each at most
        :data:`MAX_STRING_BYTES` (65535) bytes. Pass empty strings for the
        identity-only variant (``variant_id == 2``).

    Returns
    -------
    bytes
        The container.

    Raises
    ------
    ValueError
        If a field is out of range or a payload is too long for its length prefix.
    """
    if not 0 <= rate_index <= 63:
        raise ValueError(f"rate_index {rate_index} out of [0,63]")
    if not 0 <= variant_id <= 3:
        raise ValueError(f"variant_id {variant_id} out of [0,3]")
    if len(sidechannel) > 255:
        raise ValueError(f"side-channel {len(sidechannel)} B exceeds 1-byte length")
    if variant_id != VARIANT_IDENTITY_ONLY:
        for name, string in (("y", y_string), ("z", z_string)):
            if len(string) > MAX_STRING_BYTES:
                raise ValueError(
                    f"{name} string of {len(string)} B exceeds the "
                    f"{MAX_STRING_BYTES} B limit of its 2-byte length prefix"
                )
    if not 1 <= res <= 0xFFFF:
        raise ValueError(f"resolution {res} px out of [1,65535]")
    bucket = _res_to_bucket(res)
    out = bytearray()
    out.append((variant_id << 6) | rate_index)
    out.append(bucket)
    out.append(len(sidechannel))
    out += sidechannel
    if variant_id != VARIANT_IDENTITY_ONLY:
        out += struct.pack(">H", len(y_string)) + y_string
        out += struct.pack(">H", len(z_string)) + z_string
    if bucket == RAW_GEOM:
        out += struct.pack(">HH", res, res)
    return bytes(out)


def unpack(buf: bytes) -> Header:
    """Parse container bytes back into a :class:`Header`.

    The buffer must hold exactly one container: truncated buffers and trailing bytes
    after the last field are rejected.

    Raises
    ------
    ValueError
        If the buffer is truncated, has trailing bytes or carries an unknown
        resolution bucket.
    """
    buf = bytes(buf)
    n = len(buf)

    def need(off: int, k: int, what: str) -> None:
        if off + k > n:
            raise ValueError(
                f"container truncated in {what}: need {off + k} B, got {n} B"
            )

    need(0, 3, "the header")
    flags = buf[0]
    variant_id = flags >> 6
    rate_index = flags & 0x3F
    bucket = buf[1]
    sc_len = buf[2]
    if bucket != RAW_GEOM and bucket not in BUCKET_TO_RES:
        raise ValueError(f"unknown resolution bucket {bucket}")
    off = 3
    need(off, sc_len, "the side-channel")
    sidechannel = buf[off : off + sc_len]
    off += sc_len
    y_string = z_string = b""
    if variant_id != VARIANT_IDENTITY_ONLY:
        need(off, 2, "the y length prefix")
        (ylen,) = struct.unpack(">H", buf[off : off + 2])
        off += 2
        need(off, ylen, "the y string")
        y_string = buf[off : off + ylen]
        off += ylen
        need(off, 2, "the z length prefix")
        (zlen,) = struct.unpack(">H", buf[off : off + 2])
        off += 2
        need(off, zlen, "the z string")
        z_string = buf[off : off + zlen]
        off += zlen
    if bucket == RAW_GEOM:
        need(off, RAW_GEOM_TRAILER, "the raw-geometry trailer")
        res, _w = struct.unpack(">HH", buf[off : off + RAW_GEOM_TRAILER])
        off += RAW_GEOM_TRAILER
    else:
        res = BUCKET_TO_RES[bucket]
    if off != n:
        raise ValueError(f"{n - off} trailing byte(s) after a {off} B container")
    return Header(variant_id, rate_index, res, sidechannel, y_string, z_string)


def packed_size(
    res: int, sc_len: int, y_len: int, z_len: int, identity_only: bool = False
) -> int:
    """Return the container size in bytes for the given payload lengths."""
    n = 3 + sc_len
    if not identity_only:
        n += 4 + y_len + z_len
    if needs_raw_geometry(res):
        n += RAW_GEOM_TRAILER
    return n


def describe(buf: bytes) -> dict:
    """Return the header fields and payload sizes of a container (no decoding)."""
    h = unpack(buf)
    return {
        "variant": VARIANT_NAMES.get(h.variant_id, str(h.variant_id)),
        "variant_id": h.variant_id,
        "rate_index": h.rate_index,
        "res": h.res,
        "bytes": len(buf),
        "side_bytes": len(h.sidechannel),
        "y_bytes": len(h.y_string),
        "z_bytes": len(h.z_string),
        "raw_geometry": needs_raw_geometry(h.res),
    }
