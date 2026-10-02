# SPDX-License-Identifier: MIT
"""Registry of the ten baseline codecs of the benchmark.

Each entry is a :class:`BaselineCodec` with a uniform interface::

    from face1kb.baselines import get_codec

    codec = get_codec("webp")
    data, info = codec.encode_to_budget(img, 1024)   # info: fitted, setting, size
    img_hat = codec.decode(data)

``img`` is an ``H x W x 3`` ``uint8`` RGB array. Codec names are the ones used in
the result files (``jpeg``, ``jpeg2000``, ``webp``, ``jpeg_xl``, ``avif``,
``heif``, ``jpeg_fzt``, ``jpeg_ai``, ``neural_bmshj2018``,
``neural_mbt2018_mean``); the two face1kb codecs are ``ours_fast`` and
``ours_accurate`` (encode them with :func:`face1kb.load`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import classical, compressai_codecs, jpeg_fzt

__all__ = [
    "ALL_EXTENSIONS",
    "BASELINES",
    "CODECS",
    "CPU_CODECS",
    "GPU_CODECS",
    "BaselineCodec",
    "codec_for_extension",
    "extension",
    "get_codec",
]


@dataclass(frozen=True)
class BaselineCodec:
    """A baseline codec with a budget-constrained encoder and a decoder.

    Attributes
    ----------
    name : str
        Codec name used in file paths and result tables.
    label : str
        Display name of the paper.
    ext : str
        File extension of the stored bitstream.
    device : str
        ``"cpu"`` or ``"cuda"`` (where the paper ran it).
    family : str
        ``"classical"``, ``"jpeg_fzt"``, ``"jpeg_ai"`` or ``"compressai"``.
    knob : str
        The rate knob the budget search varies.
    """

    name: str
    label: str
    ext: str
    device: str
    family: str
    knob: str
    _encode: Callable = field(default=None, repr=False, compare=False)
    _decode: Callable = field(default=None, repr=False, compare=False)

    def encode_to_budget(self, img, budget: int, **kwargs) -> tuple[bytes, dict]:
        """Encode ``img`` to at most ``budget`` bytes (see the codec module).

        Returns ``(data, info)`` with ``info = {"fitted", "setting", "size"}``;
        ``fitted`` is False when no setting fits (the smallest-output setting is
        then returned). Keyword arguments go to the codec module's
        ``encode_to_budget`` (e.g. ``device`` for CompressAI, ``profile`` for
        JPEG-AI, ``grid`` / ``search`` for the classical codecs, ``qualities``
        for JPEG-FzT).
        """
        return self._encode(img, int(budget), **kwargs)

    def decode(self, data: bytes, **kwargs) -> np.ndarray:
        """Decode ``data`` to an ``H x W x 3`` ``uint8`` RGB array.

        Keyword arguments: ``res`` (output size of JPEG-FzT), ``device``
        (CompressAI), JPEG-AI options (``profile``,
        ``tools_off``, ``workdir``, ``repo_dir``). Arguments a codec does not use
        are ignored.
        """
        return self._decode(data, **kwargs)


def _classical_entry(name: str, label: str, knob: str) -> BaselineCodec:
    def enc(img, budget, **kw):
        return classical.encode_to_budget(img, name, budget, **kw)

    def dec(data, **_kw):
        return classical.decode(data)

    return BaselineCodec(
        name, label, classical.EXTENSIONS[name], "cpu", "classical", knob, enc, dec
    )


def _fzt_decode(data, *, res=None, **_kw):
    return jpeg_fzt.decode(data, res)


def _jpegai_encode(img, budget, **kw):
    from . import jpeg_ai  # noqa: PLC0415

    return jpeg_ai.encode_to_budget(img, budget, **kw)


def _jpegai_decode(data, **kw):
    from . import jpeg_ai  # noqa: PLC0415

    out = jpeg_ai.decode(
        data, **{k: v for k, v in kw.items() if k in jpeg_ai.DECODE_OPTIONS}
    )
    if out is None:
        raise jpeg_ai.JpegAIError("JPEG-AI decoding failed")
    return out


def _compressai_entry(name: str, label: str) -> BaselineCodec:
    def enc(img, budget, **kw):
        return compressai_codecs.encode_to_budget(img, name, budget, **kw)

    def dec(data, *, device="cuda", **_kw):
        return compressai_codecs.decode(data, device=device)

    return BaselineCodec(
        name,
        label,
        compressai_codecs.EXTENSION,
        "cuda",
        "compressai",
        "model quality 1..8",
        enc,
        dec,
    )


_Q = "quality 2..94 (step 2)"
_ENTRIES = (
    _classical_entry("jpeg", "JPEG", _Q),
    _classical_entry("jpeg2000", "JPEG 2000", "compression ratio 400..12 (step 4)"),
    _classical_entry("webp", "WebP", _Q),
    _classical_entry("jpeg_xl", "JPEG XL", _Q),
    _classical_entry("avif", "AVIF", _Q),
    _classical_entry("heif", "HEIF", _Q),
    BaselineCodec(
        "jpeg_fzt",
        "JPEG-FzT",
        jpeg_fzt.EXTENSION,
        "cpu",
        "jpeg_fzt",
        "JPEG quality 1..95 (step 2)",
        jpeg_fzt.encode_to_budget,
        _fzt_decode,
    ),
    BaselineCodec(
        "jpeg_ai",
        "JPEG-AI",
        ".jpegai",
        "cuda",
        "jpeg_ai",
        "target bpp x 100 in 2..50 (analytic fit)",
        _jpegai_encode,
        _jpegai_decode,
    ),
    _compressai_entry("neural_bmshj2018", "bmshj2018"),
    _compressai_entry("neural_mbt2018_mean", "mbt2018"),
)

#: The ten baseline codecs by name, in the order of the paper tables.
BASELINES: dict[str, BaselineCodec] = {c.name: c for c in _ENTRIES}
#: Names of the baseline codecs.
CODECS: tuple[str, ...] = tuple(BASELINES)
#: Baselines the paper ran on the CPU / on a GPU.
CPU_CODECS: tuple[str, ...] = tuple(c.name for c in _ENTRIES if c.device == "cpu")
GPU_CODECS: tuple[str, ...] = tuple(c.name for c in _ENTRIES if c.device == "cuda")
#: File extension of every codec of the benchmark, including the face1kb codecs.
ALL_EXTENSIONS: dict[str, str] = {
    **{c.name: c.ext for c in _ENTRIES},
    "ours_fast": ".bin",
    "ours_accurate": ".bin",
}


def get_codec(name: str) -> BaselineCodec:
    """Return the baseline codec ``name``."""
    if not isinstance(name, str):
        raise TypeError(
            f"codec name must be a str, got {type(name).__name__}; note that "
            "face1kb.baselines.encode_to_budget takes (codec, img, budget)"
        )
    try:
        return BASELINES[name]
    except KeyError:
        raise ValueError(
            f"unknown baseline codec {name!r}; choose from {CODECS}"
        ) from None


def extension(codec: str) -> str:
    """File extension of ``codec`` (a baseline or ``ours_fast``/``ours_accurate``)."""
    try:
        return ALL_EXTENSIONS[codec]
    except KeyError:
        raise ValueError(
            f"unknown codec {codec!r}; choose from {tuple(ALL_EXTENSIONS)}"
        ) from None


def codec_for_extension(ext: str) -> tuple[str, ...]:
    """Return the codecs whose files use extension ``ext`` (e.g. ``".ptci"``)."""
    ext = ext if ext.startswith(".") else "." + ext
    return tuple(c for c, e in ALL_EXTENSIONS.items() if e == ext.lower())
