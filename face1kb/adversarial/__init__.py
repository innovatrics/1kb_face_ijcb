# SPDX-License-Identifier: MIT
"""No-box adversarial attacks and the compression-as-defence metric.

Modules
-------
attacks
    :func:`hfc_attack` (training-free, CPU), :func:`clip_surrogate_attack` (OpenAI
    CLIP ViT-B/32 surrogate), :func:`li_ae_attack` / :class:`LiAEAttack` (proxy
    auto-encoder + ILA), the :data:`ATTACKS` registry, :func:`craft`, and the naming
    helpers :func:`eps_tag` / :func:`adv_suffix`.
proxy
    :class:`ProxyAE`, the Li-AE surrogate, and its weight I/O (:func:`load_proxy`,
    :func:`save_proxy`); the released weights are ``weights/liae_proxy.safetensors``.
train
    Training recipe of the proxy (:func:`select_identities`, :func:`train_proxy`);
    ``python -m face1kb.adversarial.train``.
sanitization
    The residual / sanitization metric (:func:`sanitization_record`).

Example
-------
>>> from face1kb import adversarial as adv
>>> x_adv = adv.hfc_attack(crops, eps=0.06)                 # doctest: +SKIP
>>> x_adv = adv.li_ae_attack(crops, eps=0.06, device="cuda")  # doctest: +SKIP
"""

from __future__ import annotations

from ._philox import cuda_launch_blocks
from .attacks import (
    ATTACK_RES,
    ATTACKS,
    DEFAULT_EPS,
    PAPER_CUDA_BLOCKS,
    LiAEAttack,
    adv_suffix,
    clip_surrogate_attack,
    craft,
    eps_tag,
    hfc_attack,
    li_ae_attack,
    load_clip,
    paper_cuda_blocks,
    step_size,
)
from .sanitization import (
    METRIC_COLUMNS,
    cell_cosines,
    mean_cosine,
    sanitization_record,
    valid_rows,
)

# Names that need torch at import time are loaded lazily, so that the HFC attack and
# the metric can be used without it.
_LAZY = {
    "ProxyAE": "proxy",
    "default_proxy_path": "proxy",
    "load_proxy": "proxy",
    "read_proxy_file": "proxy",
    "save_proxy": "proxy",
    "select_identities": "train",
    "train_proxy": "train",
}

__all__ = [
    "ATTACKS",
    "ATTACK_RES",
    "DEFAULT_EPS",
    "LiAEAttack",
    "METRIC_COLUMNS",
    "PAPER_CUDA_BLOCKS",
    "adv_suffix",
    "cell_cosines",
    "clip_surrogate_attack",
    "craft",
    "cuda_launch_blocks",
    "eps_tag",
    "hfc_attack",
    "li_ae_attack",
    "load_clip",
    "mean_cosine",
    "paper_cuda_blocks",
    "sanitization_record",
    "step_size",
    "valid_rows",
    *_LAZY,
]


def __getattr__(name):
    if name in _LAZY:
        import importlib  # noqa: PLC0415

        module = importlib.import_module(f"{__name__}.{_LAZY[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
