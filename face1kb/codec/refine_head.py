# SPDX-License-Identifier: MIT
"""Gated refinement head of the ACCURATE variant.

A light residual head that refines the decoded image conditioned on the decoded
identity code ``p_hat``. The residual is gated low (0.1), so the head sharpens a
faithful reconstruction rather than synthesising one; the spatial latent stays
dominant. It is trained only as a refiner under the reconstruction and identity
losses (no adversarial training).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RefinementHead(nn.Module):
    """Predict a small residual on ``x_hat`` conditioned on the identity code.

    Parameters
    ----------
    p_dim
        Dimensionality of the decoded identity code.
    ch
        Hidden channel count.
    """

    def __init__(self, p_dim: int = 128, ch: int = 32):
        super().__init__()
        self.cond = nn.Linear(p_dim, ch)
        self.net = nn.Sequential(
            nn.Conv2d(3 + ch, ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch, 3, 3, padding=1),
        )

    def forward(
        self, x_hat: torch.Tensor, p_hat: torch.Tensor, gate: float = 0.1
    ) -> torch.Tensor:
        """Return ``clamp(x_hat + gate * residual, 0, 1)``."""
        n, _, h, w = x_hat.shape
        cmap = self.cond(p_hat).view(n, -1, 1, 1).expand(-1, -1, h, w)
        residual = self.net(torch.cat([x_hat, cmap], dim=1))
        return (x_hat + gate * residual).clamp(0.0, 1.0)
