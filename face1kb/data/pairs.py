"""Canonical verification-pair protocol of the paper.

The evaluation uses *all* non-redundant image pairs: every unordered pair of
distinct images ``(i, j)`` with ``i < j`` in the canonical ``names.csv``
ordering. Pairs of images from the same identity directory are mated
(label 1), all remaining pairs are non-mated (label 0). Symmetric duplicates
and self-pairs are excluded. On the full dataset this yields 1.52 M mated
and 152.2 M non-mated pairs.

Because the pair set is simply the strict upper triangle of the image x image
matrix, it is never materialised as a list; metric computations iterate over
it in row blocks (see ``face1kb.metrics.verification``). This module provides
the identity labels that define pair labels, plus helpers to inspect or
export the pair list.
"""

import csv
from pathlib import Path

import numpy as np

from face1kb import config


def load_names(names_csv: Path | None = None) -> tuple[list[str], np.ndarray]:
    """Load the canonical image list and integer identity labels.

    Parameters
    ----------
    names_csv : Path, optional
        CSV with ``name`` and ``identity`` columns; defaults to
        ``data/names.csv``.

    Returns
    -------
    names : list of str
        Relative image file names in canonical (sorted) order.
    identity_ids : numpy.ndarray
        Integer identity label per image, shape ``(N,)``.
    """
    names_csv = names_csv or config.NAMES_CSV
    names = []
    identities = []
    with open(names_csv, newline="") as f:
        for row in csv.DictReader(f):
            names.append(row["name"])
            identities.append(row["identity"])

    unique = sorted(set(identities))
    identity_to_id = {identity: i for i, identity in enumerate(unique)}
    identity_ids = np.array(
        [identity_to_id[i] for i in identities], dtype=np.int32
    )
    return names, identity_ids


def pair_counts(identity_ids: np.ndarray) -> tuple[int, int]:
    """Return ``(n_mated, n_nonmated)`` over all non-redundant pairs."""
    n = len(identity_ids)
    total = n * (n - 1) // 2
    _, per_identity = np.unique(identity_ids, return_counts=True)
    n_mated = int(np.sum(per_identity * (per_identity - 1) // 2))
    return n_mated, total - n_mated
