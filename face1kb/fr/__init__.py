# SPDX-License-Identifier: MIT
"""Face-recognition evaluators: the 14-model roster of the paper and user hooks.

Every evaluator maps aligned **112 x 112 RGB uint8** crops to an ``(N, 512)``
float32 array of raw embeddings (L2-normalise at scoring time), with its family's
preprocessing built in:

>>> from face1kb import fr
>>> model = fr.load(fr.ARCFACE, device="cuda")     # doctest: +SKIP
>>> emb = model.embed(crops)                       # doctest: +SKIP

Roster (:data:`ROSTER`): ArcFace R100 (insightface ``antelopev2``), TopoFR
R50/R100/R200, LVFace T/S/B/L, CVLface ViT-B (KP-RPE) and IR-101, EdgeFace
XXS/XS/S/Base. The four anchors of the headline results are :data:`ANCHORS`;
:data:`HELDOUT` is the held-out verification matcher and :data:`IDCOS_DEFAULT` the
default matcher for identity-cosine scores.

Weights are not part of face1kb: they are downloaded from the official sources
into ``FACE1KB_MODELS_ROOT`` and verified by SHA-256 (``scripts/fetch_models.py``
fetches them all; :func:`load` fetches a missing file on first use). Most of them
are licensed for non-commercial research only; see ``docs/models.md``.

Your own matcher (e.g. a private or commercial model) can be plugged in with
:func:`register_onnx`, :func:`register_torch` or :func:`register_embedder`; a module
that registers it can be named in ``FACE1KB_FR_PLUGINS`` to make it available in every
process (:func:`load_plugins`).
"""

from __future__ import annotations

from .download import ManualDownloadRequired, fetch, present, verify
from .embedders import (
    EMB_DIM,
    INPUT_SIZE,
    Embedder,
    OnnxEmbedder,
    TorchEmbedder,
    as_batch,
)
from .families import ARCFACE_TEMPLATE_112
from .registry import (
    ANCHORS,
    ARCFACE,
    EDGEFACE_XS,
    HELDOUT,
    IDCOS_DEFAULT,
    LVFACE_L,
    MODEL_LABELS,
    PLUGINS_ENV,
    REGISTRY,
    ROSTER,
    TOPOFR_R100,
    ModelSpec,
    available,
    label,
    load,
    load_plugins,
    register_embedder,
    register_onnx,
    register_torch,
    torch_builder,
    unregister,
)
from .sources import ModelSource, model_source

__all__ = [
    "ANCHORS",
    "ARCFACE",
    "ARCFACE_TEMPLATE_112",
    "EDGEFACE_XS",
    "EMB_DIM",
    "HELDOUT",
    "IDCOS_DEFAULT",
    "INPUT_SIZE",
    "LVFACE_L",
    "MODEL_LABELS",
    "PLUGINS_ENV",
    "REGISTRY",
    "ROSTER",
    "TOPOFR_R100",
    "Embedder",
    "ManualDownloadRequired",
    "ModelSource",
    "ModelSpec",
    "OnnxEmbedder",
    "TorchEmbedder",
    "as_batch",
    "available",
    "fetch",
    "label",
    "load",
    "load_plugins",
    "model_source",
    "present",
    "register_embedder",
    "register_onnx",
    "register_torch",
    "torch_builder",
    "unregister",
    "verify",
]
