# SPDX-License-Identifier: MIT
"""Complete verification-pair matrix over a dataset index.

Every unordered image pair is enumerated once -- the strict upper triangle
``idx1 < idx2`` of the ``N x N`` matrix in row-major order (the order of
``numpy.triu_indices(N, k=1)``) -- and labelled mated (``1``, same subject) or
non-mated (``0``). Pairs are stored index-based as ``pairs.parquet`` with columns
``idx1: int32, idx2: int32, label: int8`` (zstd), the indices being ``id`` values of
the dataset index (:mod:`face1kb.data.index`). The accuracy stage scores all mated
pairs and a seeded sample of the non-mated ones.

For the paper's crop sets this gives 64,235,445 pairs (95,839 mated) on Color FERET
(11,335 crops) and 153,711,811 pairs (1,515,807 mated) on AI-Solutions-KK (17,534
crops).

A small human-readable sample (``name1, name2, label``; :func:`sample_pairs`) can
be written next to it; its policy is dataset-specific (:data:`SAMPLE_POLICIES`).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Seed of the readable pair sample.
SAMPLE_SEED = 0
#: Readable-sample policy per dataset: (mated pairs, non-mated pairs); ``None``
#: keeps all mated pairs.
SAMPLE_POLICIES: dict[str, tuple[int | None, int]] = {
    "colorferet": (None, 200_000),
    "kk": (200_000, 200_000),
}
#: Rows generated per block by :func:`pair_indices` (bounds the temporary memory).
_BLOCK_ROWS = 1 << 22


def n_pairs(n: int) -> int:
    """Return the number of unordered pairs ``C(n, 2)``."""
    return n * (n - 1) // 2


def pair_counts(subjects) -> tuple[int, int]:
    """Return ``(total pairs, mated pairs)`` for a list of per-image subjects."""
    sizes = pd.Series(np.asarray(subjects)).value_counts().to_numpy(dtype=np.int64)
    n = int(sizes.sum())
    return n_pairs(n), int((sizes * (sizes - 1) // 2).sum())


def pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """All pairs ``i < j`` of ``n`` items in row-major order, as int32 arrays.

    Equal to ``numpy.triu_indices(n, k=1)`` cast to int32, without the int64
    intermediates.
    """
    if n >= 2**31:
        raise ValueError("too many images for int32 pair indices")
    total = n_pairs(n)
    idx1 = np.empty(total, dtype=np.int32)
    idx2 = np.empty(total, dtype=np.int32)
    pos = 0
    i = 0
    while i < n - 1:
        # rows i..k-1 in one block of about _BLOCK_ROWS pairs
        k = i
        block = 0
        while k < n - 1 and (block == 0 or block + (n - 1 - k) <= _BLOCK_ROWS):
            block += n - 1 - k
            k += 1
        lengths = n - 1 - np.arange(i, k, dtype=np.int64)
        rows = np.repeat(np.arange(i, k, dtype=np.int32), lengths)
        starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
        cols = (np.arange(block, dtype=np.int64) - starts) + rows + 1
        idx1[pos : pos + block] = rows
        idx2[pos : pos + block] = cols
        pos += block
        i = k
    assert pos == total
    return idx1, idx2


def build_pairs(subjects) -> pd.DataFrame:
    """Build the full pair table for per-image ``subjects`` in index order.

    Parameters
    ----------
    subjects : array-like
        Subject of each index row (row ``i`` = image id ``i``).

    Returns
    -------
    DataFrame
        ``idx1`` (int32), ``idx2`` (int32), ``label`` (int8, 1 = mated).
    """
    codes = pd.factorize(pd.Series(np.asarray(subjects)), sort=True)[0]
    codes = codes.astype(np.int32)
    idx1, idx2 = pair_indices(len(codes))
    label = (codes[idx1] == codes[idx2]).astype(np.int8)
    return pd.DataFrame({"idx1": idx1, "idx2": idx2, "label": label})


def write_pairs(pairs: pd.DataFrame, path: str | os.PathLike) -> Path:
    """Write the pair table as zstd-compressed parquet (no pandas index)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(path, index=False, compression="zstd")
    return path


def read_pairs(path: str | os.PathLike) -> pd.DataFrame:
    """Read a pair table written by :func:`write_pairs`."""
    return pd.read_parquet(path)


def sample_pairs(
    pairs: pd.DataFrame,
    rel_paths,
    n_mated: int | None,
    n_non_mated: int,
    seed: int = SAMPLE_SEED,
) -> pd.DataFrame:
    """Seeded readable sample of the pair table as ``name1, name2, label``.

    Parameters
    ----------
    pairs : DataFrame
        Output of :func:`build_pairs`.
    rel_paths : array-like
        ``rel_path`` of each index row.
    n_mated : int or None
        Mated pairs to draw (``None`` keeps all of them, in pair order).
    n_non_mated : int
        Non-mated pairs to draw.
    seed : int
        Seed of ``numpy.random.default_rng``; mated pairs are drawn first.
    """
    label = pairs["label"].to_numpy()
    idx1 = pairs["idx1"].to_numpy()
    idx2 = pairs["idx2"].to_numpy()
    rel = np.asarray(rel_paths, dtype=object)
    rng = np.random.default_rng(seed)
    mated = np.flatnonzero(label == 1)
    if n_mated is not None:
        mated = rng.choice(mated, size=min(n_mated, mated.size), replace=False)
    non_mated = np.flatnonzero(label == 0)
    non_mated = rng.choice(
        non_mated, size=min(n_non_mated, non_mated.size), replace=False
    )
    take = np.concatenate([mated, non_mated])
    return pd.DataFrame(
        {"name1": rel[idx1[take]], "name2": rel[idx2[take]], "label": label[take]}
    )
