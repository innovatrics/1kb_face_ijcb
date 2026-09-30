# SPDX-License-Identifier: MIT
"""FiLM conditioning of the synthesis transform on (gain, resolution).

The decoder is told the operating rate (the continuous gain ``g``) and the input
resolution ``R`` so it can adapt to both. A small MLP maps the conditioning vector
``[g, R / 256]`` to per-channel ``(gamma, beta)`` that modulate a synthesis feature map.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def film(h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Apply channel-wise ``gamma * h + beta`` (``gamma``, ``beta``: ``(N, C)``)."""
    return gamma[..., None, None] * h + beta[..., None, None]


class RateResFiLM(nn.Module):
    """Map ``(gain, resolution)`` to per-channel ``(gamma, beta)`` for one stage.

    ``gamma`` is centred at 1 (the MLP predicts a residual), so an untrained module
    starts near the identity. The same conditioning serves every spatial location.

    Parameters
    ----------
    channels
        Channel count of the modulated feature map.
    hidden
        Width of the hidden MLP layer.
    """

    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * channels),
        )

    def forward(self, gain: float, res: int, device=None) -> tuple:
        """Return ``(gamma, beta)``, each shaped ``(1, channels)``."""
        dev = device or next(self.parameters()).device
        v = torch.tensor([[float(gain), res / 256.0]], device=dev, dtype=torch.float32)
        gb = self.net(v)
        gamma, beta = gb.chunk(2, dim=-1)
        return 1.0 + gamma, beta
