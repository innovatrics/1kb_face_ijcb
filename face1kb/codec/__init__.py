# SPDX-License-Identifier: MIT
"""The face1kb learned face codecs (FAST and ACCURATE).

Both variants are mean-scale hyperprior codecs with gain-unit variable rate that emit
a self-describing container of at most ``B`` bytes.

Modules
-------
api
    Public API: :func:`load`, :class:`Codec` (``encode``/``decode``), :func:`decode`.
variants
    ``FaceCodecFast`` / ``FaceCodecAccurate`` and :func:`build_model`.
backbone
    ``MeanScaleHyperpriorVBR3``: the shared hyperprior backbone.
budget
    Hard byte-budget encoding (binary search over the gain table) and decoding.
container
    The container format (pack / unpack).
side_stream, refine_head, film, layers, resize
    Building blocks.
identity_loss, losses, data, train
    Training: frozen EdgeFace models, the objective, data loading, the trainer.
cli
    ``python -m face1kb.codec {encode,decode,info}``.
"""

from __future__ import annotations

__all__ = ["Codec", "decode", "inspect", "load", "set_reproducible_backend"]


def __getattr__(name):
    # Lazy re-exports: keep ``import face1kb.codec.container`` free of torch.
    if name in __all__:
        from . import api  # noqa: PLC0415

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
