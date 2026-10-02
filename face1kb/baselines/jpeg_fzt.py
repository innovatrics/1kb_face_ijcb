# SPDX-License-Identifier: MIT
"""JPEG-FzT: F-transform preprocessing around JPEG.

JPEG-FzT (Perfilieva and Hurtik, "The F-transform preprocessing for JPEG strong
compression of high-resolution images", Information Sciences 550, 2021) compresses
an image in two stages:

1. **Encode.** A direct F-transform with a raised-cosine basis of radius ``h = 2``
   (``hh = 0``) filters the image and samples it on a grid of step ``h``, giving a
   half-resolution image. That image is stored as a baseline JPEG
   (``optimize=True``); the ``.fzt`` file *is* this JPEG.
2. **Decode.** The JPEG is decoded and the inverse F-transform with the
   tabulated inverse basis (``data/jpeg_fzt_ibf.txt``, 4,990 values) and a
   neighbourhood radius ``rad = 1`` reconstructs the full resolution.

This is a NumPy implementation of the original C++ code, including its boundary
handling. The benchmark searches the JPEG quality over ``1, 3, ..., 95``
(:data:`QUALITIES`); the recompression, preprocessing-through-codec and codec
comparison studies use ``2, 4, ..., 94`` (:data:`QUALITIES_EVEN`).
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image

from .search import FitResult, binary_search_fit, scan_fit

__all__ = [
    "EXTENSION",
    "H",
    "HH",
    "QUALITIES",
    "QUALITIES_EVEN",
    "RAD",
    "compress_only_with_quality",
    "compress_with_quality",
    "decode",
    "decode_stage",
    "downsample",
    "encode_quality",
    "encode_stage",
    "encode_to_budget",
    "fit",
    "load_ibf",
]

#: File extension of a JPEG-FzT bitstream (a baseline JPEG of the low-res image).
EXTENSION = ".fzt"
#: F-transform basis radius (encode) and its extension.
H, HH = 2, 0
#: Inverse F-transform neighbourhood radius used for every result of the paper.
RAD = 1
#: JPEG quality grid of the benchmark (``range(1, 96, 2)``).
QUALITIES: tuple[int, ...] = tuple(range(1, 96, 2))
#: JPEG quality grid of the recompression / preprocessing / comparison studies.
QUALITIES_EVEN: tuple[int, ...] = tuple(range(2, 96, 2))

_IBF_RESOURCE = "jpeg_fzt_ibf.txt"


def _read_values(text: str, source) -> np.ndarray:
    values = [float(s) for s in (line.strip() for line in text.splitlines()) if s]
    if not values:
        raise ValueError(f"no weights loaded from {source}")
    return np.asarray(values, dtype=np.float32)


@lru_cache(maxsize=1)
def _packaged_ibf() -> np.ndarray:
    ref = resources.files("face1kb.baselines").joinpath("data", _IBF_RESOURCE)
    try:
        text = ref.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"the JPEG-FzT basis table {ref} is missing from the installed package; "
            "it must be installed as package data of face1kb.baselines "
            "(data/*.txt), or pass its path to load_ibf()"
        ) from exc
    return _read_values(text, ref)


def load_ibf(path: str | Path | None = None) -> np.ndarray:
    """Load the inverse F-transform basis table (float32, 4,990 values).

    ``path`` defaults to the table shipped with the package.
    """
    if path is None:
        return _packaged_ibf().copy()
    path = Path(path)
    return _read_values(path.read_text(encoding="utf-8"), path)


def encode_stage(src: np.ndarray, h: int = H, hh: int = HH) -> np.ndarray:
    """Direct F-transform: filter and subsample an ``H x W x 3`` ``uint8`` image.

    Returns the ``(H // h) x (W // h) x 3`` ``uint8`` low-resolution image.
    """
    height, width, _ = src.shape
    nw = max(1, width // h)
    nh = max(1, height // h)

    bf = np.array(
        [
            (np.cos(np.pi * (float(i) / float(h + hh))) + 1.0) / 2.0
            for i in range(h + hh + 1)
        ],
        dtype=np.float32,
    )

    # Weighted sums over the (2r+1)^2 neighbourhood by shifted-slice accumulation.
    # Out-of-image samples are skipped and the normalisation uses the weights of
    # the valid samples only, as in the reference loops.
    src_f = src.astype(np.float32, copy=False)
    weighted_sum = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    radius = h + hh

    for dy in range(-radius, radius + 1):
        wy = float(bf[abs(dy)])
        if dy >= 0:
            dst_y = slice(0, height - dy)
            src_y = slice(dy, height)
        else:
            dst_y = slice(-dy, height)
            src_y = slice(0, height + dy)

        for dx in range(-radius, radius + 1):
            wx = float(bf[abs(dx)])
            w = wx * wy
            if dx >= 0:
                dst_x = slice(0, width - dx)
                src_x = slice(dx, width)
            else:
                dst_x = slice(-dx, width)
                src_x = slice(0, width + dx)

            weighted_sum[dst_y, dst_x] += src_f[src_y, src_x] * w
            weight_sum[dst_y, dst_x] += w

    filtered = weighted_sum / np.maximum(weight_sum[..., None], 1e-12)

    # When a dimension is not divisible by h, the last output cell takes the last
    # anchor inside the image (the largest multiple of h), as in the reference.
    x_coords = np.arange(nw, dtype=np.int32) * h
    y_coords = np.arange(nh, dtype=np.int32) * h
    x_coords[-1] = ((width - 1) // h) * h
    y_coords[-1] = ((height - 1) // h) * h

    compressed = filtered[y_coords[:, None], x_coords[None, :]]
    return np.clip(np.rint(compressed), 0, 255).astype(np.uint8)


def decode_stage(  # noqa: C901 - mirrors the reference inverse-transform loops
    lowres_jpeg_decoded: np.ndarray,
    output_size: tuple[int, int],
    ibf: np.ndarray,
    h: int = H,
    rad: int = RAD,
) -> np.ndarray:
    """Inverse F-transform of a decoded low-resolution image.

    Parameters
    ----------
    lowres_jpeg_decoded
        ``h x w x 3`` ``uint8`` low-resolution image (the decoded JPEG).
    output_size
        ``(width, height)`` of the reconstruction.
    ibf
        Inverse basis table (:func:`load_ibf`).
    h
        Grid step of the direct transform.
    rad
        Neighbourhood radius of the reconstruction (the paper used ``1``).

    Returns
    -------
    numpy.ndarray
        ``height x width x 3`` ``uint8`` reconstruction.
    """
    out_w, out_h = output_size
    nh, nw, _ = lowres_jpeg_decoded.shape
    mult = (1.0 / float(h)) * 20.0

    max_neighbors = 2 * rad + 1
    max_ibf_idx = len(ibf) - 1

    # Per output column: the contributing low-res columns and their weights, in the
    # order of the reference per-pixel loop (ascending neighbour position).
    x_nx = np.zeros((out_w, max_neighbors), dtype=np.int32)
    x_w = np.zeros((out_w, max_neighbors), dtype=np.float32)
    x_count = np.zeros(out_w, dtype=np.int32)
    for x in range(out_w):
        cnt = 0
        x0 = max(x - rad, 0)
        x1 = min(x + rad, out_w - 1)
        for xx in range(x0, x1 + 1):
            if xx % h:
                continue
            wx_idx = min(int(round(abs(x - xx) * mult)), max_ibf_idx)
            x_nx[x, cnt] = min(xx // h, nw - 1)
            x_w[x, cnt] = float(ibf[wx_idx])
            cnt += 1
        x_count[x] = cnt

    max_x_count = int(np.max(x_count)) if out_w else 0
    x_valid_masks = [(x_count > i) for i in range(max_x_count)]
    x_nx_cols = [x_nx[mask, i] for i, mask in enumerate(x_valid_masks)]
    x_w_cols = [x_w[mask, i] for i, mask in enumerate(x_valid_masks)]

    # The same for the output rows.
    y_ny = np.zeros((out_h, max_neighbors), dtype=np.int32)
    y_w = np.zeros((out_h, max_neighbors), dtype=np.float32)
    y_count = np.zeros(out_h, dtype=np.int32)
    for y in range(out_h):
        cnt = 0
        y0 = max(y - rad, 0)
        y1 = min(y + rad, out_h - 1)
        for yy in range(y0, y1 + 1):
            if yy % h:
                continue
            wy_idx = min(int(round(abs(y - yy) * mult)), max_ibf_idx)
            y_ny[y, cnt] = min(yy // h, nh - 1)
            y_w[y, cnt] = float(ibf[wy_idx])
            cnt += 1
        y_count[y] = cnt

    lowres_f = lowres_jpeg_decoded.astype(np.float32, copy=False)
    result = np.zeros((out_h, out_w, 3), dtype=np.float32)

    for y in range(out_h):
        sum_rgb = np.zeros((out_w, 3), dtype=np.float32)
        num = np.zeros(out_w, dtype=np.float32)
        for yi in range(y_count[y]):
            ny = y_ny[y, yi]
            wy = y_w[y, yi]
            for xi in range(max_x_count):
                mask = x_valid_masks[xi]
                if not np.any(mask):
                    continue
                w = x_w_cols[xi] * wy
                sum_rgb[mask] += lowres_f[ny, x_nx_cols[xi]] * w[:, None]
                num[mask] += w

        valid = num > 0.0
        result[y, valid] = sum_rgb[valid] / num[valid, None]

    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


# Names of the original module, kept for scripts that call the stages directly.
_encode_stage = encode_stage
_decode_stage = decode_stage
_load_weights = load_ibf


def _to_rgb_array(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(
            f"expected an H x W x 3 uint8 RGB image, got {arr.shape} {arr.dtype}"
        )
    return arr


def downsample(img) -> Image.Image:
    """Low-resolution F-transform image of ``img`` as a PIL image (encode stage 1)."""
    return Image.fromarray(encode_stage(_to_rgb_array(img), h=H, hh=HH))


def encode_quality(lowres: Image.Image, quality: int) -> bytes:
    """JPEG-encode a low-resolution image from :func:`downsample` (encode stage 2)."""
    buf = io.BytesIO()
    lowres.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()


def fit(
    img,
    budget: int,
    *,
    qualities: Sequence[int] = QUALITIES,
    search: str = "binary",
    **scan_kwargs,
) -> FitResult:
    """Budget search over the JPEG quality (see :func:`encode_to_budget`)."""
    low = downsample(img)

    def enc(q):
        data = encode_quality(low, q)
        return len(data), data

    if search == "binary":
        if scan_kwargs:
            raise TypeError(f"unexpected arguments for a binary search: {scan_kwargs}")
        return binary_search_fit(enc, list(qualities), budget)
    if search == "scan":
        return scan_fit(enc, list(qualities), budget, **scan_kwargs)
    raise ValueError(f"search must be 'binary' or 'scan', got {search!r}")


def encode_to_budget(
    img,
    budget: int,
    *,
    qualities: Sequence[int] = QUALITIES,
    search: str = "binary",
    **scan_kwargs,
) -> tuple[bytes, dict]:
    """Encode ``img`` with the largest JPEG quality whose ``.fzt`` fits ``budget``.

    Parameters
    ----------
    img
        ``H x W x 3`` ``uint8`` RGB array (or PIL image).
    budget
        Byte budget.
    qualities
        Quality grid in ascending order (benchmark :data:`QUALITIES`; the
        recompression and preprocessing studies use :data:`QUALITIES_EVEN`).
    search
        ``"binary"`` or ``"scan"`` (see :mod:`face1kb.baselines.search`).

    Returns
    -------
    tuple[bytes, dict]
        The ``.fzt`` bytes and ``{"fitted", "setting", "size"}``.
    """
    r = fit(img, budget, qualities=qualities, search=search, **scan_kwargs)
    return r.payload, {"fitted": r.fitted, "setting": r.setting, "size": r.size}


def decode(
    data: bytes,
    size: tuple[int, int] | int | None = None,
    *,
    rad: int = RAD,
    ibf: np.ndarray | None = None,
) -> np.ndarray:
    """Decode a ``.fzt`` bitstream to an ``H x W x 3`` ``uint8`` RGB array.

    Parameters
    ----------
    data
        The ``.fzt`` bytes (a JPEG of the low-resolution image).
    size
        Output ``(width, height)``, or an int for a square output. Default: twice
        the low-resolution size, which is the original size for even dimensions
        (all benchmark resolutions are even).
    rad
        Neighbourhood radius of the inverse transform (the paper used ``1``).
    ibf
        Inverse basis table (default: the packaged table).
    """
    with Image.open(io.BytesIO(data)) as im:
        low = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if size is None:
        size = (low.shape[1] * H, low.shape[0] * H)
    elif isinstance(size, int):
        size = (size, size)
    table = _packaged_ibf() if ibf is None else ibf
    return decode_stage(low, (int(size[0]), int(size[1])), table, h=H, rad=rad)


def compress_with_quality(img, quality: int, weights_path: str | Path | None = None):
    """Encode and decode at one JPEG quality.

    Returns ``(lowres_image, reconstruction, size_bytes)``: the PIL low-resolution
    image before JPEG coding, the PIL full-resolution reconstruction of the JPEG
    and the JPEG size. ``quality`` is clamped to ``1..100``.
    """
    quality = int(max(1, min(100, quality)))
    src = _to_rgb_array(img)
    low = Image.fromarray(encode_stage(src, h=H, hh=HH))
    data = encode_quality(low, quality)
    ibf = _packaged_ibf() if weights_path is None else load_ibf(weights_path)
    rec = decode(data, (src.shape[1], src.shape[0]), ibf=ibf)
    return low, Image.fromarray(rec), len(data)


def compress_only_with_quality(img, quality: int):
    """Encode at one JPEG quality; return ``(lowres_image, size_bytes)``."""
    quality = int(max(1, min(100, quality)))
    low = downsample(img)
    return low, len(encode_quality(low, quality))
