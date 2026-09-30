# SPDX-License-Identifier: MIT
"""Deterministic, licence-free synthetic test images (no dataset faces)."""

from __future__ import annotations

import numpy as np


def smooth_face(res: int) -> np.ndarray:
    """A smooth, face-like synthetic RGB image (ellipses and gradients), uint8."""
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float64) / res
    img = np.empty((res, res, 3))
    img[..., 0] = 0.55 + 0.25 * xx
    img[..., 1] = 0.45 + 0.20 * yy
    img[..., 2] = 0.40 + 0.10 * (xx + yy)
    face = ((xx - 0.5) / 0.32) ** 2 + ((yy - 0.52) / 0.42) ** 2 < 1.0
    img[face] = img[face] * 0.6 + np.array([0.78, 0.60, 0.50]) * 0.4
    for cx in (0.38, 0.62):  # eyes
        eye = ((xx - cx) / 0.06) ** 2 + ((yy - 0.42) / 0.03) ** 2 < 1.0
        img[eye] = (0.15, 0.12, 0.10)
    mouth = ((xx - 0.5) / 0.12) ** 2 + ((yy - 0.72) / 0.025) ** 2 < 1.0
    img[mouth] = (0.55, 0.25, 0.25)
    img += 0.02 * np.sin(40 * xx)[..., None] * np.cos(33 * yy)[..., None]
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def noise(res: int, seed: int = 0) -> np.ndarray:
    """Uniform RGB noise (a worst case for the rate), uint8."""
    return np.random.default_rng(seed).integers(0, 256, (res, res, 3), dtype=np.uint8)
