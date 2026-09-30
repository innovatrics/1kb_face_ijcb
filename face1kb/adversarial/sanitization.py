# SPDX-License-Identifier: MIT
"""Sanitization metric of the compression-as-defence study.

For one face matcher, one attack at one ``eps`` and one (codec, budget) cell, every
crop gives three identity cosines to the embedding of its **clean, uncompressed**
crop (the reference):

* ``a = cos(clean, adv)`` -- attack strength before compression (low = strong);
* ``b = cos(clean, codec(clean))`` -- the codec's own identity cost;
* ``c = cos(clean, codec(adv))`` -- identity left after attack and codec.

The cell reports their means over the crops for which all four embeddings exist,
and derives

* ``residual = b - c`` -- the identity loss added by the attack that survives the
  codec (about 0: the perturbation was removed; large: it survived). This is the
  headline metric of the paper;
* ``sanitization = (c - a) / (b - a)`` -- the fraction of the attack's gap that
  the codec recovers (1 = fully sanitized, <= 0 = not at all).

A crop that a codec could not produce is stored as an all-zero embedding row (or a
NaN row); such rows are excluded through :func:`valid_rows`, so a codec that covers
only part of a dataset is compared with itself on the same crops.
"""

from __future__ import annotations

import numpy as np

#: Columns of a sanitization record (after the identifying columns).
METRIC_COLUMNS: tuple[str, ...] = (
    "n",
    "cos_adv",
    "cos_clean_comp",
    "cos_adv_comp",
    "residual",
    "sanitization",
)


def valid_rows(e: np.ndarray) -> np.ndarray:
    """Boolean mask of rows that hold a real embedding (finite and not all zero)."""
    return np.isfinite(e).all(axis=1) & (np.abs(e).sum(axis=1) > 0)


def unit(e: np.ndarray) -> np.ndarray:
    """Row-normalise ``e`` (``1e-12`` guards against zero rows)."""
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def mean_cosine(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean per-row cosine of ``a`` and ``b`` over the rows selected by ``mask``."""
    c = (unit(a[mask]) * unit(b[mask])).sum(axis=1)
    c = c[np.isfinite(c)]
    return float(c.mean()) if c.size else np.nan


def cell_cosines(
    clean: np.ndarray,
    adv: np.ndarray | None,
    clean_comp: np.ndarray | None,
    adv_comp: np.ndarray | None,
) -> tuple[float, float, float, int]:
    """Return ``(a, b, c, n)`` of one cell (see the module docstring).

    All arrays are ``(N, D)`` embeddings with row ``i`` for crop ``i``. A missing
    array (``None``) or an empty common mask gives ``(nan, nan, nan, 0)``.
    """
    if adv is None or clean_comp is None or adv_comp is None:
        return np.nan, np.nan, np.nan, 0
    mask = valid_rows(clean) & valid_rows(adv) & valid_rows(clean_comp)
    mask &= valid_rows(adv_comp)
    n = int(mask.sum())
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    return (
        mean_cosine(clean, adv, mask),
        mean_cosine(clean, clean_comp, mask),
        mean_cosine(clean, adv_comp, mask),
        n,
    )


def sanitization_record(
    clean: np.ndarray,
    adv: np.ndarray | None,
    clean_comp: np.ndarray | None,
    adv_comp: np.ndarray | None,
) -> dict[str, float]:
    """Metric record of one cell (the :data:`METRIC_COLUMNS`).

    Parameters
    ----------
    clean
        Embeddings of the clean, uncompressed crops (the reference).
    adv
        Embeddings of the adversarial, uncompressed crops.
    clean_comp
        Embeddings of the clean crops after the codec.
    adv_comp
        Embeddings of the adversarial crops after the codec.

    Returns
    -------
    dict
        The :data:`METRIC_COLUMNS`; NaN where the cell has no data.
    """
    a, b, c, n = cell_cosines(clean, adv, clean_comp, adv_comp)
    return {
        "n": n,
        "cos_adv": a,
        "cos_clean_comp": b,
        "cos_adv_comp": c,
        "residual": b - c,
        "sanitization": (c - a) / (b - a) if (b - a) else np.nan,
    }
