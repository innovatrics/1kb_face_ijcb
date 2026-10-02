# SPDX-License-Identifier: MIT
"""Classical image codecs through Pillow and its plugins.

=========  ==========================================  ===========================
codec      backend                                     knob (search order)
=========  ==========================================  ===========================
jpeg       Pillow / libjpeg-turbo                      quality 2, 4, ..., 94;
                                                       ``optimize=True``, default
                                                       4:2:0 chroma subsampling
jpeg2000   Pillow / OpenJPEG                           compression ratio 400, 396,
                                                       ..., 12 (``quality_mode=
                                                       "rates"``, one layer)
webp       Pillow / libwebp                            quality 2..94, ``method=6``
jpeg_xl    pillow-jxl-plugin / libjxl                  quality 2..94, default effort
avif       Pillow (native libavif, aom encoder)        quality 2..94, default speed
heif       pillow-heif / libheif + x265 (HEVC)         quality 2..94, default preset
=========  ==========================================  ===========================

The stored file is the container Pillow writes (``.jpg``, ``.jp2``, ``.webp``,
``.jxl``, ``.avif``, ``.heic``) and its full size counts against the budget.

Image metadata: a PIL input is encoded as given, including its ``info`` (as the
benchmark did with ``Image.open(path).convert("RGB")``). The aligned crops carry no
metadata, but decoded images do, and several encoders embed it: an image decoded
from JPEG XL carries a 536 B ICC profile that the AVIF and HEIF encoders write into
their files, and the comment of a decoded JPEG 2000 file is written by the JPEG
encoder. The recompression study chained codecs that way (:func:`decode_pil` returns
such an image). Pass ``strip_metadata=True`` to encode pixels only; NumPy inputs
never carry metadata.

The encoded bytes depend on the exact library builds; the paper used Pillow 12.2.0,
pillow-heif 1.4.0 and pillow-jxl-plugin 1.3.7 (see ``docs/baselines.md``).
"""

from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np
from PIL import Image

from .search import FitResult, binary_search_fit, scan_fit

__all__ = [
    "CODECS",
    "EXTENSIONS",
    "PIL_FORMATS",
    "decode",
    "decode_pil",
    "encode_setting",
    "encode_to_budget",
    "fit",
    "save_kwargs",
    "settings",
    "to_pil",
]

#: Classical codecs, in the order of the paper tables.
CODECS: tuple[str, ...] = ("jpeg", "jpeg2000", "webp", "jpeg_xl", "avif", "heif")
#: Pillow format name of each codec.
PIL_FORMATS: dict[str, str] = {
    "jpeg": "JPEG",
    "jpeg2000": "JPEG2000",
    "webp": "WEBP",
    "jpeg_xl": "JXL",
    "avif": "AVIF",
    "heif": "HEIF",
}
#: File extension of each codec.
EXTENSIONS: dict[str, str] = {
    "jpeg": ".jpg",
    "jpeg2000": ".jp2",
    "webp": ".webp",
    "jpeg_xl": ".jxl",
    "avif": ".avif",
    "heif": ".heic",
}

#: Quality grid of the benchmark: ``range(2, 96, 2)`` = 2, 4, ..., 94.
QUALITY_START, QUALITY_STOP, QUALITY_STEP = 2, 96, 2
#: JPEG 2000 ratio grid of the benchmark: ``range(400, 8, -4)`` = 400, 396, ..., 12.
RATIO_START, RATIO_STOP, RATIO_STEP = 400, 8, 4

_PLUGINS_READY = False


def _ensure_plugins() -> None:
    """Register the HEIF and JPEG XL plugins with Pillow (AVIF is native)."""
    global _PLUGINS_READY  # noqa: PLW0603
    if _PLUGINS_READY:
        return
    try:
        import pillow_heif  # noqa: PLC0415

        pillow_heif.register_heif_opener()
    except ImportError:  # pragma: no cover - optional dependency
        pass
    try:
        import pillow_jxl  # noqa: F401, PLC0415  (registers JXL on import)
    except ImportError:  # pragma: no cover - optional dependency
        pass
    _PLUGINS_READY = True


def _check_codec(codec: str) -> str:
    if codec not in PIL_FORMATS:
        raise ValueError(f"unknown classical codec {codec!r}; choose from {CODECS}")
    return codec


def to_pil(img, *, strip_metadata: bool = False) -> Image.Image:
    """Return an RGB :class:`PIL.Image.Image`.

    Parameters
    ----------
    img
        ``H x W x 3`` ``uint8`` array, or a PIL image (converted with
        ``convert("RGB")``, which keeps its ``info``).
    strip_metadata
        Drop the ``info`` of a PIL input (ICC profile, EXIF, XMP, comments, ...)
        so that no encoder can embed it.
    """
    if isinstance(img, Image.Image):
        if not strip_metadata:
            return img if img.mode == "RGB" else img.convert("RGB")
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected an H x W x 3 RGB image, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            raise ValueError(f"expected a uint8 image, got dtype {arr.dtype}")
    return Image.fromarray(np.ascontiguousarray(arr))


def settings(
    codec: str, *, quality_step: int = QUALITY_STEP, ratio_step: int = RATIO_STEP
) -> list[int]:
    """Search grid of ``codec``, ordered by non-decreasing output size.

    Parameters
    ----------
    codec
        Classical codec name.
    quality_step
        Step of the quality grid ``range(2, 96, quality_step)`` (benchmark: 2).
    ratio_step
        Step of the JPEG 2000 ratio grid ``range(400, 8, -ratio_step)``
        (benchmark: 4).
    """
    _check_codec(codec)
    if codec == "jpeg2000":
        return list(range(RATIO_START, RATIO_STOP, -int(ratio_step)))
    return list(range(QUALITY_START, QUALITY_STOP, int(quality_step)))


def save_kwargs(codec: str, setting: int) -> dict:
    """Pillow ``save`` keyword arguments of ``codec`` at ``setting``."""
    _check_codec(codec)
    if codec == "jpeg":
        return {"quality": int(setting), "optimize": True}
    if codec == "webp":
        return {"quality": int(setting), "method": 6}
    if codec == "jpeg2000":
        return {"quality_mode": "rates", "quality_layers": [setting]}
    return {"quality": int(setting)}


def encode_setting(
    img, codec: str, setting: int, *, strip_metadata: bool = False
) -> bytes:
    """Encode ``img`` with ``codec`` at one ``setting`` and return the file bytes."""
    _ensure_plugins()
    pil = to_pil(img, strip_metadata=strip_metadata)
    buf = io.BytesIO()
    fmt = PIL_FORMATS[_check_codec(codec)]
    pil.save(buf, format=fmt, **save_kwargs(codec, setting))
    return buf.getvalue()


def fit(
    img,
    codec: str,
    budget: int,
    *,
    grid: Sequence[int] | None = None,
    search: str = "binary",
    strip_metadata: bool = False,
    **scan_kwargs,
) -> FitResult:
    """Budget search for ``codec`` on ``img`` (see :func:`encode_to_budget`)."""
    _check_codec(codec)
    pil = to_pil(img, strip_metadata=strip_metadata)
    grid = settings(codec) if grid is None else list(grid)

    def enc(s):
        data = encode_setting(pil, codec, s)
        return len(data), data

    if search == "binary":
        if scan_kwargs:
            raise TypeError(f"unexpected arguments for a binary search: {scan_kwargs}")
        return binary_search_fit(enc, grid, budget)
    if search == "scan":
        return scan_fit(enc, grid, budget, **scan_kwargs)
    raise ValueError(f"search must be 'binary' or 'scan', got {search!r}")


def encode_to_budget(
    img,
    codec: str,
    budget: int,
    *,
    grid: Sequence[int] | None = None,
    search: str = "binary",
    strip_metadata: bool = False,
    **scan_kwargs,
) -> tuple[bytes, dict]:
    """Encode ``img`` with the largest setting of ``codec`` that fits ``budget``.

    Parameters
    ----------
    img
        ``H x W x 3`` ``uint8`` RGB array (or PIL image).
    codec
        One of :data:`CODECS`.
    budget
        Byte budget (the whole file counts).
    grid
        Settings to search, in non-decreasing output-size order (default:
        :func:`settings` with the benchmark steps).
    search
        ``"binary"`` (benchmark, :func:`~face1kb.baselines.search.binary_search_fit`)
        or ``"scan"`` (:func:`~face1kb.baselines.search.scan_fit`, which takes
        ``keep``, ``stop_at_overflow`` and ``skip_errors``).
    strip_metadata
        Encode the pixels of a PIL input without its ``info`` (see the module
        notes; the benchmark kept it).

    Returns
    -------
    tuple[bytes, dict]
        The file bytes and ``{"fitted", "setting", "size"}``. When nothing fits,
        the smallest-output setting is returned with ``fitted=False`` and a size
        above the budget.
    """
    r = fit(
        img,
        codec,
        budget,
        grid=grid,
        search=search,
        strip_metadata=strip_metadata,
        **scan_kwargs,
    )
    return r.payload, {"fitted": r.fitted, "setting": r.setting, "size": r.size}


def decode_pil(data: bytes) -> Image.Image:
    """Decode a classical-codec file to an RGB PIL image, keeping its ``info``.

    This is the decoded image the recompression study passed to the second codec.
    """
    _ensure_plugins()
    with Image.open(io.BytesIO(data)) as im:
        return im.convert("RGB")


def decode(data: bytes) -> np.ndarray:
    """Decode a classical-codec file to an ``H x W x 3`` ``uint8`` RGB array."""
    return np.asarray(decode_pil(data), dtype=np.uint8)
