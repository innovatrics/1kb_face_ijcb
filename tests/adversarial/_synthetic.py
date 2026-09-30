# SPDX-License-Identifier: MIT
"""Deterministic synthetic crops built with integer arithmetic only (no faces)."""

from __future__ import annotations

import numpy as np


def crops(n: int = 3, size: int = 112) -> np.ndarray:
    """Return ``(n, size, size, 3)`` uint8 images: gradients, stripes and a disc.

    Only integer operations are used, so the images are identical on every
    platform and NumPy version.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.int64)
    out = np.empty((n, size, size, 3), np.uint8)
    for k in range(n):
        r = (xx * 2 + yy + 37 * k) % 256
        g = ((xx // (4 + k)) % 2) * 120 + (yy * 3) % 100 + 20
        b = np.where(
            (xx - size // 2) ** 2 + (yy - size // 2) ** 2 < (size // 3) ** 2,
            200 - 5 * k,
            40 + (xx * yy) % 60,
        )
        out[k] = np.stack([r, g, b], -1).astype(np.uint8)
    return out
