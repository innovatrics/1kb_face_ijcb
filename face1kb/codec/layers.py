# SPDX-License-Identifier: MIT
"""Custom codec layers."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class ResizeConv(nn.Module):
    """Nearest-neighbour x2 upsample followed by a convolution.

    A checkerboard-free replacement for the stride-2 transposed convolution
    (Odena et al., 2016, "Deconvolution and Checkerboard Artifacts"). It is a drop-in
    for CompressAI's ``deconv(in_ch, out_ch)`` (same x2 upsample and channel map), but
    the upsampling is a fixed nearest-neighbour resize, so no learned transposed-conv
    grid pattern can appear; the following convolution learns the detail.

    Parameters
    ----------
    in_ch, out_ch
        Input and output channel counts.
    kernel
        Convolution kernel size (``padding = kernel // 2``).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=1, padding=kernel // 2)

    def forward(self, x):
        """Upsample ``x`` by 2 (nearest) and convolve."""
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))
