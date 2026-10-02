# SPDX-License-Identifier: MIT
"""Resolution helpers: pad to a multiple of 128, crop back, differentiable resizes.

The backbone downsamples by 128
(:class:`~face1kb.codec.backbone.MeanScaleHyperpriorVBR3`), so every input is
edge-padded (replicate, right/bottom) to a multiple of 128 before the analysis
transform, and the decoded image is cropped back to the true ``(H, W)``.
Inputs that are not a multiple of 128 would otherwise fail in the entropy stage with a
latent-vs-mean shape mismatch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Total geometric downsampling factor of the codec backbone.
DOWNSAMPLE = 128


def pad_to_multiple(x: torch.Tensor, d: int = DOWNSAMPLE) -> tuple[torch.Tensor, tuple]:
    """Edge-pad ``(N, C, H, W)`` up to a multiple of ``d``; return ``(x, (H, W))``."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % d, (-w) % d
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (h, w)


def crop_to(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Crop ``(N, C, H', W')`` back to the true ``(H, W)``."""
    h, w = hw
    return x[..., :h, :w]


def resize_to(x: torch.Tensor, side: int) -> torch.Tensor:
    """Resize a batch to ``side x side`` (area down, antialiased bilinear up)."""
    if x.shape[-2:] == (side, side):
        return x
    if side < x.shape[-1]:
        return F.interpolate(x, size=(side, side), mode="area")
    return F.interpolate(
        x, size=(side, side), mode="bilinear", align_corners=False, antialias=True
    )


def resize112(x: torch.Tensor) -> torch.Tensor:
    """Differentiable resize to the 112 x 112 input of the FR models (antialiased)."""
    if x.shape[-2:] == (112, 112):
        return x
    return F.interpolate(
        x, size=(112, 112), mode="bilinear", align_corners=False, antialias=True
    )
