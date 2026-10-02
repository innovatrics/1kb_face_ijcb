# SPDX-License-Identifier: MIT
"""The ten baseline codecs of the benchmark.

Classical codecs (JPEG, JPEG 2000, WebP, JPEG XL, AVIF, HEIF) run through Pillow and
its plugins, JPEG-FzT is implemented here, JPEG-AI runs through its reference
software, and bmshj2018-factorized / mbt2018-mean come from CompressAI. Every codec
is driven to the largest setting whose output fits a byte budget.

>>> from face1kb.baselines import get_codec, decode_file
>>> data, info = get_codec("webp").encode_to_budget(img, 1024)   # doctest: +SKIP
>>> img_hat = decode_file("crop.webp")                            # doctest: +SKIP

Modules
-------
registry
    :class:`BaselineCodec` entries (name, extension, device, encoder, decoder).
search
    The budget searches (:func:`binary_search_fit`, :func:`scan_fit`,
    :func:`hinted_binary_search_fit`).
classical, jpeg_fzt, jpeg_ai, compressai_codecs
    The codec families.
decode
    :func:`decode_file` / :func:`decode_bytes`: decoding by file extension,
    including the face1kb ``.bin`` containers.
verify
    ``python -m face1kb.baselines.verify``: round-trip check of the installation.
"""

from __future__ import annotations

from .decode import decode_bytes, decode_file, decoded_cache_path
from .registry import (
    ALL_EXTENSIONS,
    BASELINES,
    CODECS,
    CPU_CODECS,
    GPU_CODECS,
    BaselineCodec,
    codec_for_extension,
    extension,
    get_codec,
)
from .search import FitResult, binary_search_fit, hinted_binary_search_fit, scan_fit

__all__ = [
    "ALL_EXTENSIONS",
    "BASELINES",
    "CODECS",
    "CPU_CODECS",
    "GPU_CODECS",
    "BaselineCodec",
    "FitResult",
    "binary_search_fit",
    "codec_for_extension",
    "decode_bytes",
    "decode_file",
    "decoded_cache_path",
    "encode_to_budget",
    "extension",
    "get_codec",
    "hinted_binary_search_fit",
    "scan_fit",
]


def encode_to_budget(codec: str, img, budget: int, **kwargs):
    """Encode ``img`` with baseline ``codec`` to at most ``budget`` bytes.

    Shorthand for ``get_codec(codec).encode_to_budget(img, budget, **kwargs)``;
    returns ``(data, info)``. The codec name comes first here, whereas the family
    functions ``classical.encode_to_budget`` and ``compressai_codecs.encode_to_budget``
    take ``(img, codec, budget)``.
    """
    return get_codec(codec).encode_to_budget(img, budget, **kwargs)
