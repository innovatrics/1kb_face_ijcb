# SPDX-License-Identifier: MIT
"""face1kb: sub-1 kB identity-preserving face compression.

The package provides the two learned face codecs face1kb-FAST and face1kb-ACCURATE
(``Ours-FAST`` / ``Ours-ACCURATE`` in the paper) and the code that reproduces the
experiments of "Toward Sub-1 kB Identity-Preserving Face Compression".

>>> import face1kb
>>> codec = face1kb.load("fast", device="cuda")   # doctest: +SKIP
>>> data = codec.encode(img, budget=1024)          # doctest: +SKIP
>>> img_hat = codec.decode(data)                   # doctest: +SKIP
"""

from __future__ import annotations

__version__ = "2.0.0"

__all__ = ["__version__", "load", "decode"]


def load(variant: str = "fast", device="cuda", weights_dir=None, **kwargs):
    """Load a released face1kb codec; see :func:`face1kb.codec.api.load`."""
    from face1kb.codec.api import load as _load  # noqa: PLC0415

    return _load(variant, device=device, weights_dir=weights_dir, **kwargs)


def decode(data: bytes, device="cuda", weights_dir=None):
    """Decode any face1kb container; see :func:`face1kb.codec.api.decode`."""
    from face1kb.codec.api import decode as _decode  # noqa: PLC0415

    return _decode(data, device=device, weights_dir=weights_dir)
