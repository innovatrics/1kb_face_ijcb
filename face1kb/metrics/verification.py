"""Core verification metrics: pair scores, EER, FAR/FRR operating points.

The evaluation protocol scores *all* non-redundant image pairs (strict upper
triangle of the image x image similarity matrix, see
``face1kb.data.pairs``). For the compressed conditions the enrollment side
uses embeddings of the original images and the probe side embeddings of the
compressed images, exactly as in the paper's original-to-compressed
scenario. Scores are cosine similarities of L2-normalised embeddings.
"""

import numpy as np

from face1kb import config


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise embeddings row-wise (zero rows are left untouched)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (embeddings / norms).astype(np.float32)


def pair_scores(
    enroll: np.ndarray,
    probe: np.ndarray,
    identity_ids: np.ndarray,
    block_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity and mated-labels for all non-redundant pairs.

    Parameters
    ----------
    enroll : numpy.ndarray
        Enrollment-side embeddings, shape ``(N, D)`` (original images).
    probe : numpy.ndarray
        Probe-side embeddings, shape ``(N, D)`` (compressed images; pass the
        same array as *enroll* for the uncompressed baseline).
    identity_ids : numpy.ndarray
        Integer identity label per image, shape ``(N,)``.
    block_size : int
        Number of rows processed per matrix product.

    Returns
    -------
    scores : numpy.ndarray
        Cosine similarities, float32, shape ``(N * (N - 1) / 2,)``.
    labels : numpy.ndarray
        Boolean mated/non-mated label per pair, same shape.
    """
    if len(enroll) != len(probe) or len(enroll) != len(identity_ids):
        raise ValueError("enroll, probe and identity_ids must align")

    enroll_n = l2_normalize(enroll)
    probe_n = l2_normalize(probe)
    n = len(enroll_n)
    total = n * (n - 1) // 2

    scores = np.empty(total, dtype=np.float32)
    labels = np.empty(total, dtype=bool)
    offset = 0
    for start in range(0, n - 1, block_size):
        stop = min(start + block_size, n - 1)
        block = enroll_n[start:stop] @ probe_n.T
        for i in range(start, stop):
            row = block[i - start, i + 1 :]
            scores[offset : offset + row.size] = row
            labels[offset : offset + row.size] = (
                identity_ids[i] == identity_ids[i + 1 :]
            )
            offset += row.size
    assert offset == total
    return scores, labels


def compute_eer(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Equal Error Rate and the similarity threshold where it occurs.

    Returns
    -------
    tuple of (float, float)
        ``(eer, threshold)``; the EER is the FAR at the point where the
        absolute difference between FAR and FRR is minimal.
    """
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]

    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    true_pos = np.cumsum(sorted_labels)
    false_pos = np.arange(1, labels.size + 1) - true_pos

    far = false_pos / n_neg
    frr = 1.0 - true_pos / n_pos

    eer_idx = int(np.argmin(np.abs(far - frr)))
    return float(far[eer_idx]), float(scores[order[eer_idx]])


def far_frr_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[float, float]:
    """FAR and FRR of the accept-if-``score >= threshold`` decision."""
    accepted = scores >= threshold
    far = float(np.mean(accepted[~labels])) if (~labels).any() else 0.0
    frr = float(np.mean(~accepted[labels])) if labels.any() else 0.0
    return far, frr


def accuracy_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> float:
    """Classification accuracy of the accept-if-``score >= threshold`` rule."""
    return float(np.mean((scores >= threshold) == labels))


def frr_at_far_levels(
    scores: np.ndarray,
    labels: np.ndarray,
    far_levels: tuple[float, ...] = config.FAR_LEVELS,
) -> dict[float, tuple[float, float, float]]:
    """FRR at fixed FAR operating points.

    The threshold for a target FAR is the corresponding percentile of the
    non-mated score distribution (nudged down so the achieved FAR is at
    least the target), identical to the paper's implementation.

    Returns
    -------
    dict
        ``{target_far: (threshold, achieved_far, frr)}``.
    """
    negatives = scores[~labels]
    results = {}
    for target_far in far_levels:
        percentile = (1.0 - target_far) * 100.0
        if percentile >= 100:
            threshold = float(negatives.max()) + 1e-4
        elif percentile <= 0:
            threshold = float(negatives.min()) - 1e-4
        else:
            threshold = float(np.percentile(negatives, percentile)) - 1e-10
        far, frr = far_frr_at_threshold(scores, labels, threshold)
        results[target_far] = (threshold, far, frr)
    return results
