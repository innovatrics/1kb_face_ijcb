# SPDX-License-Identifier: MIT
"""Synthetic probe crops and embedding summaries for the golden evaluator test.

The crops are generated (no dataset faces), so their embeddings may be shipped.
Run this module to embed them with one evaluator and print the summary as JSON::

    python tests/fr/_golden.py lvface_l [--device cuda]
"""

from __future__ import annotations

import json
import sys

import numpy as np

#: Number of leading embedding dimensions stored per crop.
N_DIMS = 8


def probe_crops() -> np.ndarray:
    """Two deterministic 112 x 112 RGB uint8 crops: a smooth face-like pattern
    and uniform noise."""
    res = 112
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float64) / res
    img = np.empty((res, res, 3))
    img[..., 0] = 0.55 + 0.25 * xx
    img[..., 1] = 0.45 + 0.20 * yy
    img[..., 2] = 0.40 + 0.10 * (xx + yy)
    face = ((xx - 0.5) / 0.32) ** 2 + ((yy - 0.52) / 0.42) ** 2 < 1.0
    img[face] = img[face] * 0.6 + np.array([0.78, 0.60, 0.50]) * 0.4
    for cx in (0.34, 0.66):  # eyes, near the ArcFace template positions
        eye = ((xx - cx) / 0.06) ** 2 + ((yy - 0.46) / 0.03) ** 2 < 1.0
        img[eye] = (0.15, 0.12, 0.10)
    mouth = ((xx - 0.5) / 0.13) ** 2 + ((yy - 0.82) / 0.025) ** 2 < 1.0
    img[mouth] = (0.55, 0.25, 0.25)
    smooth = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    noise = np.random.default_rng(0).integers(0, 256, (res, res, 3), dtype=np.uint8)
    return np.stack([smooth, noise])


def summarise(emb: np.ndarray) -> dict:
    """Norms, leading normalised dimensions and the cross cosine of two embeddings."""
    emb = np.asarray(emb, dtype=np.float64)
    norm = np.linalg.norm(emb, axis=1)
    unit = emb / norm[:, None]
    return {
        "dim": int(emb.shape[1]),
        "norm": [float(v) for v in norm],
        "head": [[float(v) for v in row[:N_DIMS]] for row in unit],
        "cos01": float(unit[0] @ unit[1]),
    }


def main(argv: list[str]) -> None:
    """Embed the probe crops with ``argv[0]`` and print the summary as JSON."""
    from face1kb import fr

    name = argv[0]
    device = argv[argv.index("--device") + 1] if "--device" in argv else None
    model = fr.load(name, device=device, download=False)
    print(json.dumps({name: summarise(model.embed(probe_crops()))}))


if __name__ == "__main__":
    main(sys.argv[1:])
