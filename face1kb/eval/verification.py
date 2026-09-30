# SPDX-License-Identifier: MIT
"""1:1 verification accuracy: pairs, cosine scoring, EER, FNMR@FMR and bootstrap CIs.

Protocol
--------
* **Trials.** Every mated pair of ``pairs.parquet`` (``label == 1``) plus one seeded
  sample of non-mated pairs (``label == 0``): ``np.random.default_rng(seed).choice``
  of ``sample_nonmated`` positions without replacement (:func:`sample_pairs`). The
  study uses seed 0 and 5,000,000 non-mated pairs for the accuracy grid, 2,000,000
  for the significance tests, 5,000,000 for the subgroup EERs, 300,000 for the
  subgroup FMRs and 400,000 for the disparity CIs (:data:`NONMATED_SAMPLE`).
* **Scores.** Cosine similarity of L2-normalised embeddings (``e / (|e| + 1e-12)``),
  in float32 (:func:`cosine_scores`). Both members of a pair are taken from the same
  embedding array, i.e. a compressed source is scored compressed-vs-compressed.
* **EER.** A grid of :data:`EER_GRID_POINTS` = 4000 thresholds, ``np.linspace`` over
  ``[min(all scores), max(all scores)]`` (:func:`eer_grid`); FNMR(t) =
  P(genuine < t), FMR(t) = P(impostor >= t); the grid point minimising
  ``|FNMR - FMR|`` is taken and EER = (FNMR + FMR) / 2 there (:func:`eer`,
  :func:`eer_threshold`).
* **FNMR @ FMR = f.** With ``m = round(f * N_neg)``, the threshold is the ``m``-th
  largest impostor score (``sorted_neg[N_neg - m]``); FNMR is the fraction of
  genuine scores below it. When ``m < 10`` (:data:`MIN_NEG_ABOVE`) the operating
  point is not estimable and FNMR is NaN (:func:`verification_metrics`).
* **CIs.** Percentile 95 % intervals from a subject-cluster bootstrap
  (:func:`bootstrap_ci`): subjects are resampled with replacement and every trial
  enters with the multiplicity of its subject (mated pairs: the shared subject;
  impostor pairs: the subject of ``idx1``). 200 replicates, seed 0, with the
  impostors capped at 1,000,000 by a seeded subsample. The grid computes CIs only
  for the clean cells (empty suffix) of :data:`BOOTSTRAP_MODELS`.

The scores are float32. The published numbers were computed under NumPy 2.4. Under
its scalar promotion rules (NEP 50), ``np.linspace`` over float32 endpoints evaluates
in float32 (as in every NumPy 2 release), and ``np.quantile`` of a float32 array with
a Python-float ``q`` interpolates in float32 from a float64 virtual index
``(n - 1) * q``. NumPy 1.x evaluates both in float64, which moves some EERs by one
trial; NumPy 2.0-2.3 cast ``q`` to float32 and so compute the virtual index in
float32, which can move an FMR threshold. :func:`eer_grid` and
:func:`linear_quantile` perform the published arithmetic explicitly, so the results
do not depend on the installed NumPy version.

Two scorers compute the same formula: :func:`cosine_scores` (NumPy) and
:func:`cosine_scores_torch` (CUDA). Their float32 sums differ in the last bit for
most pairs, which moves an EER by one trial when a score lands on the other side
of a grid threshold. The published accuracy grid and subgroup EERs were scored with
the CUDA scorer (``device="cuda"``), the published significance tests with the
NumPy scorer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from face1kb import config

from .embeddings import EMB_DIM, check_rows, list_sources

log = logging.getLogger(__name__)

#: Target false-match rates of the reported operating points.
FMR_TARGETS: tuple[float, ...] = (1e-2, 1e-3, 1e-4)
#: Number of thresholds of the EER grid.
EER_GRID_POINTS = 4000
#: Minimum number of impostor scores above an FMR operating point for it to be
#: estimable; below that FNMR is reported as NaN.
MIN_NEG_ABOVE = 10
#: Non-mated sample sizes (seed 0) used by the studies.
NONMATED_SAMPLE: dict[str, int] = {
    "accuracy": 5_000_000,
    "significance": 2_000_000,
    "fairness": 5_000_000,
    "fmr_fairness": 300_000,
    "disparity_ci": 400_000,
}
#: Bootstrap replicates of the accuracy CIs.
BOOTSTRAP_REPS = 200
#: Impostor cap of the accuracy bootstrap.
BOOTSTRAP_NONMATED_CAP = 1_000_000
#: Matchers that get bootstrap CIs in the accuracy grid (clean cells only).
BOOTSTRAP_MODELS: tuple[str, ...] = (
    "arcface_antelopev2",
    "lvface_l",
    "topofr_r100",
    "edgeface_xs",
    "cvlface_ir101",
)

#: Column order of ``accuracy/metrics.csv``.
METRICS_COLUMNS: tuple[str, ...] = (
    "dataset",
    "model",
    "tag",
    "kind",
    "res",
    "suffix",
    "codec",
    "budget",
    "n_pos",
    "n_neg",
    "eer",
    "fnmr_0.01",
    "fmr_realised_0.01",
    "fnmr_0.001",
    "fmr_realised_0.001",
    "fnmr_0.0001",
    "fmr_realised_0.0001",
    "eer_lo",
    "eer_hi",
    "fnmr_0.01_lo",
    "fnmr_0.01_hi",
    "fnmr_0.001_lo",
    "fnmr_0.001_hi",
    "fnmr_0.0001_lo",
    "fnmr_0.0001_hi",
    "median_bytes",
    "eer_aligned",
    "delta_eer",
)

Pairs = tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
Scorer = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


# ------------------------------------------------------------------------- pairs
def load_pairs(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ``pairs.parquet`` of ``dataset`` as ``(idx1, idx2, label)`` arrays."""
    import pandas as pd  # noqa: PLC0415

    pairs = pd.read_parquet(config.pairs_parquet(dataset))
    return (
        pairs["idx1"].to_numpy(),
        pairs["idx2"].to_numpy(),
        pairs["label"].to_numpy(),
    )


def sample_pairs(
    idx1: np.ndarray,
    idx2: np.ndarray,
    label: np.ndarray,
    sample_nonmated: int,
    seed: int = 0,
) -> Pairs:
    """All mated pairs plus a seeded sample of non-mated pairs.

    Parameters
    ----------
    idx1, idx2 : numpy.ndarray
        Image ids of the two members of every pair (rows of the index).
    label : numpy.ndarray
        1 for mated, 0 for non-mated pairs.
    sample_nonmated : int
        Number of non-mated pairs to draw without replacement; 0 (or a number not
        smaller than the population) keeps them all.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    tuple
        ``((pos_idx1, pos_idx2), (neg_idx1, neg_idx2))``. The non-mated sample is in
        draw order.
    """
    pos = np.flatnonzero(label == 1)
    neg = np.flatnonzero(label == 0)
    rng = np.random.default_rng(seed)
    if sample_nonmated and neg.size > sample_nonmated:
        neg = rng.choice(neg, size=sample_nonmated, replace=False)
    return (idx1[pos], idx2[pos]), (idx1[neg], idx2[neg])


def pair_indices(dataset: str, sample_nonmated: int, seed: int = 0) -> Pairs:
    """:func:`sample_pairs` on the ``pairs.parquet`` of ``dataset``."""
    return sample_pairs(*load_pairs(dataset), sample_nonmated, seed)


def subject_labels(dataset: str) -> np.ndarray:
    """Subject id of every index row (the bootstrap resampling unit)."""
    return config.read_index(dataset)["subject"].to_numpy()


def read_index_or_none(dataset: str):
    """:func:`face1kb.config.read_index`, or ``None`` (with a warning) when missing.

    The scorers use the index to check that every embedding array has one row per
    index entry (:func:`face1kb.eval.embeddings.check_rows`); without an index that
    check is skipped.
    """
    try:
        return config.read_index(dataset)
    except FileNotFoundError:
        log.warning("[%s] no index: embedding row counts are not checked", dataset)
        return None


# ------------------------------------------------------------------------ scores
def unit_rows(emb: np.ndarray) -> np.ndarray:
    """L2-normalise the rows of ``emb`` (``e / (|e| + 1e-12)``)."""
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)


def cosine_scores(
    emb: np.ndarray, i1: np.ndarray, i2: np.ndarray, chunk: int = 4_000_000
) -> np.ndarray:
    """Cosine scores of the pairs ``(i1[k], i2[k])`` (float32; NaN for NaN rows)."""
    en = unit_rows(emb)
    out = np.empty(i1.size, dtype=np.float32)
    for s in range(0, i1.size, chunk):
        e = slice(s, s + chunk)
        out[e] = (en[i1[e]] * en[i2[e]]).sum(axis=1)
    return out


# device -> largest chunk that fitted, so the halving ladder is paid once per device
_CHUNK_OK: dict[str, int] = {}


def cosine_scores_torch(
    emb: np.ndarray,
    i1: np.ndarray,
    i2: np.ndarray,
    device,
    chunk: int = 500_000,
    *,
    cpu_fallback: bool = True,
) -> np.ndarray:
    """Cosine scores on a torch device (same formula as :func:`cosine_scores`).

    On CUDA out-of-memory the chunk is halved (down to 25,000 pairs). If even that
    does not fit, the call scores with :func:`cosine_scores` when ``cpu_fallback``
    is true (a warning is logged) and raises ``RuntimeError`` otherwise. Results of
    the two scorers can differ in the last float32 bit, so a fallback can move a
    metric by one trial; pass ``cpu_fallback=False`` for an exact reproduction of
    CUDA-scored outputs.
    """
    import torch  # noqa: PLC0415

    chunk = min(chunk, _CHUNK_OK.get(str(device), chunk))
    while chunk >= 25_000:
        en = t1 = t2 = out = None
        try:
            en = torch.from_numpy(np.ascontiguousarray(emb)).to(
                device=device, dtype=torch.float32
            )
            en = en / (en.norm(dim=1, keepdim=True) + 1e-12)
            t1 = torch.from_numpy(i1.astype(np.int64)).to(device)
            t2 = torch.from_numpy(i2.astype(np.int64)).to(device)
            out = torch.empty(i1.size, dtype=torch.float32, device=device)
            for s in range(0, i1.size, chunk):
                e = slice(s, s + chunk)
                out[e] = (en[t1[e]] * en[t2[e]]).sum(dim=1)
            _CHUNK_OK[str(device)] = chunk
            return out.cpu().numpy()
        except torch.OutOfMemoryError:
            chunk //= 2
            log.warning("CUDA OOM on %s; retrying with chunk=%d", device, chunk)
        finally:
            del en, t1, t2, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not cpu_fallback:
        raise RuntimeError(
            f"device {device} out of memory even at the smallest chunk and "
            "cpu_fallback=False"
        )
    log.warning(
        "device %s too contended; scoring on CPU (NumPy-scorer numbers)", device
    )
    return cosine_scores(emb, i1, i2)


def make_scorer(device=None, *, cpu_fallback: bool = True) -> Scorer:
    """Scoring function ``(emb, i1, i2) -> scores``: NumPy when ``device`` is None.

    ``cpu_fallback`` is passed to :func:`cosine_scores_torch`.
    """
    if device is None:
        return cosine_scores
    return lambda e, a, b: cosine_scores_torch(
        e, a, b, device, cpu_fallback=cpu_fallback
    )


def paired_cosines(ref: np.ndarray, rec: np.ndarray) -> np.ndarray:
    """Per-image cosine between two embedding arrays of the same index.

    Rows where either array is NaN (first component) are dropped. Used for the
    reconstruction-vs-original identity similarity.
    """
    ok = ~np.isnan(ref[:, 0]) & ~np.isnan(rec[:, 0])
    if not ok.any():
        return np.empty(0)
    a, b = ref[ok], rec[ok]
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(axis=1)


# ----------------------------------------------------------------------- metrics
def eer_grid(lo, hi, num: int = EER_GRID_POINTS) -> np.ndarray:
    """``np.linspace(lo, hi, num)`` as NumPy 2 (any 2.x release) evaluates it.

    For float32 endpoints (the score type) the grid is computed in float32 exactly
    as NumPy 2 does it (``delta = hi - lo``, ``step = delta / (num - 1)``,
    ``arange(num) * step + lo``, last point set to ``hi``), independently of the
    installed NumPy version. Other endpoint types fall back to ``np.linspace``.
    """
    lo, hi = np.asarray(lo), np.asarray(hi)
    if lo.dtype != np.float32 or hi.dtype != np.float32 or num < 2:
        return np.linspace(lo, hi, num)
    div = np.float32(num - 1)
    delta = np.subtract(hi, lo, dtype=np.float32)
    y = np.arange(0, num, dtype=np.float32)
    step = np.divide(delta, div, dtype=np.float32)
    if step == 0:  # NumPy's branch for a zero (or underflowing) step
        y = np.multiply(np.divide(y, div, dtype=np.float32), delta, dtype=np.float32)
    else:
        y = np.multiply(y, step, dtype=np.float32)
    y = np.add(y, lo, dtype=np.float32)
    y[-1] = hi
    return y


def linear_quantile(x: np.ndarray, q: float):
    """``np.quantile(x, q)`` (linear method, float ``q``) as NumPy 2.4 computes it.

    For a float32 ``x`` the virtual index ``(n - 1) * q`` is computed in float64 and
    the interpolation between the two order statistics in float32 (NumPy >= 2.4
    promotion of a Python-float weight; the published numbers used 2.4.6),
    independently of the installed NumPy version. NumPy 1.x interpolates in
    float64, and NumPy 2.0-2.3 cast ``q`` to float32 before forming the virtual
    index; both can give a different float32 value. Any NaN in ``x`` gives NaN.
    Other dtypes fall back to ``np.quantile``.
    """
    x = np.asarray(x).ravel()
    if x.dtype != np.float32 or x.size == 0:
        return np.quantile(x, q)
    if np.isnan(x).any():
        return np.float32(np.nan)
    xs = np.sort(x)
    n = xs.size
    v = (n - 1) * np.float64(q)
    if v >= n - 1:
        return xs[-1]
    p = int(np.floor(v))
    t = float(v - np.floor(v))
    a, b = xs[p], xs[p + 1]
    diff = b - a
    if t >= 0.5:
        return np.float32(b - diff * np.float32(1 - t))
    return np.float32(a + diff * np.float32(t))


def _eer_sweep(sp: np.ndarray, sn: np.ndarray):
    """Threshold grid, FNMR, FMR and the EER index for sorted genuine/impostor scores.

    ``sp`` and ``sn`` must be sorted ascending and non-empty. Returns
    ``(thr, fnmr, fmr, j)`` with ``j = argmin |fnmr - fmr|``.
    """
    thr = eer_grid(min(sp[0], sn[0]), max(sp[-1], sn[-1]), EER_GRID_POINTS)
    fnmr = np.searchsorted(sp, thr, side="left") / sp.size  # genuine < t -> reject
    fmr = (sn.size - np.searchsorted(sn, thr, side="left")) / sn.size  # imp >= t
    j = int(np.argmin(np.abs(fnmr - fmr)))
    return thr, fnmr, fmr, j


def eer(pos: np.ndarray, neg: np.ndarray) -> float:
    """Equal-error rate of genuine ``pos`` and impostor ``neg`` scores (NaN dropped).

    Returns NaN when either side is empty.
    """
    pos = np.sort(pos[~np.isnan(pos)])
    neg = np.sort(neg[~np.isnan(neg)])
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    _, fnmr, fmr, j = _eer_sweep(pos, neg)
    return float((fnmr[j] + fmr[j]) / 2)


def eer_threshold(pos: np.ndarray, neg: np.ndarray) -> float:
    """Score threshold at the equal-error grid point (inputs must be NaN-free)."""
    thr, _, _, j = _eer_sweep(np.sort(pos), np.sort(neg))
    return float(thr[j])


def verification_metrics(
    pos: np.ndarray, neg: np.ndarray, fmrs: Sequence[float] = FMR_TARGETS
) -> dict:
    """EER and FNMR at each target FMR (NaN scores are dropped).

    Returns
    -------
    dict
        ``n_pos``, ``n_neg``, ``eer`` and, per target ``f``, ``fnmr_<f:g>`` and
        ``fmr_realised_<f:g>`` (= ``m / n_neg``). Both are NaN when fewer than
        :data:`MIN_NEG_ABOVE` impostors lie above the operating point. With an empty
        side only ``eer`` and ``fnmr_*`` are set, to NaN.
    """
    pos = np.sort(pos[~np.isnan(pos)])
    neg = np.sort(neg[~np.isnan(neg)])
    out: dict[str, float] = {"n_pos": int(pos.size), "n_neg": int(neg.size)}
    if pos.size == 0 or neg.size == 0:
        out["eer"] = np.nan
        for f in fmrs:
            out[f"fnmr_{f:g}"] = np.nan
        return out
    _, fnmr, fmr, j = _eer_sweep(pos, neg)
    out["eer"] = float((fnmr[j] + fmr[j]) / 2)
    for f in fmrs:
        # the operating point admitting a fraction f of impostors is the m-th
        # largest impostor score
        m = int(round(f * neg.size))
        if m < MIN_NEG_ABOVE:
            out[f"fnmr_{f:g}"] = np.nan
            out[f"fmr_realised_{f:g}"] = np.nan
            continue
        t = neg[neg.size - m]
        out[f"fnmr_{f:g}"] = float(np.searchsorted(pos, t, side="left") / pos.size)
        out[f"fmr_realised_{f:g}"] = float(m / neg.size)
    return out


def bootstrap_ci(  # noqa: C901 - subject/pair branches and the resampling loop
    pos: np.ndarray,
    neg: np.ndarray,
    fmrs: Sequence[float] = FMR_TARGETS,
    reps: int = BOOTSTRAP_REPS,
    neg_cap: int = BOOTSTRAP_NONMATED_CAP,
    seed: int = 0,
    pos_subj: np.ndarray | None = None,
    neg_subj: np.ndarray | None = None,
    level: str = "subject",
) -> dict[str, tuple[float, float]]:
    """Percentile 95 % bootstrap intervals of EER and each FNMR@FMR.

    Parameters
    ----------
    pos, neg : numpy.ndarray
        Genuine and impostor scores (NaN entries are dropped together with their
        subject labels).
    fmrs : sequence of float
        Target FMRs.
    reps : int
        Bootstrap replicates; 0 disables the bootstrap.
    neg_cap : int
        Impostors are subsampled (seeded, without replacement) to at most this many
        before resampling; the point estimate always uses all of them.
    seed : int
        Seed of the single ``numpy.random.default_rng`` driving the cap and the
        replicates.
    pos_subj, neg_subj : numpy.ndarray, optional
        Cluster label of every genuine / impostor trial (the shared subject of a
        mated pair, the subject of ``idx1`` of an impostor pair).
    level : {"subject", "pair"}
        ``"subject"`` resamples subjects with replacement and weights each trial by
        the draw count of its subject; it needs ``pos_subj`` and ``neg_subj``.
        Otherwise individual trials are resampled.

    Returns
    -------
    dict
        ``{"eer": (lo, hi), "fnmr_<f:g>": (lo, hi), ...}`` from
        ``np.nanpercentile(.., [2.5, 97.5])``; empty when ``reps <= 0`` or a side
        is empty.
    """
    if reps <= 0:
        return {}
    pm = ~np.isnan(pos)
    nm = ~np.isnan(neg)
    pos = pos[pm]
    neg = neg[nm]
    if pos.size == 0 or neg.size == 0:
        return {}
    rng = np.random.default_rng(seed)
    keys = ["eer"] + [f"fnmr_{f:g}" for f in fmrs]
    acc: dict[str, list] = {k: [] for k in keys}

    if level == "subject" and pos_subj is not None and neg_subj is not None:
        ps = np.asarray(pos_subj)[pm]
        ns = np.asarray(neg_subj)[nm]
        if neg.size > neg_cap:
            sel = rng.choice(neg.size, size=neg_cap, replace=False)
            neg, ns = neg[sel], ns[sel]
        # factorise the subjects of both sides to contiguous codes 0..K-1 (sorted)
        uniq, inv = np.unique(np.concatenate([ps, ns]), return_inverse=True)
        k_subj = uniq.size
        ps_code, ns_code = inv[: ps.size], inv[ps.size :]
        for _ in range(reps):
            counts = np.bincount(rng.integers(0, k_subj, k_subj), minlength=k_subj)
            pb = np.repeat(pos, counts[ps_code])
            nb = np.repeat(neg, counts[ns_code])
            if pb.size == 0 or nb.size == 0:
                continue
            m = verification_metrics(pb, nb, fmrs)
            for key in keys:
                acc[key].append(m[key])
    else:
        if neg.size > neg_cap:
            neg = rng.choice(neg, size=neg_cap, replace=False)
        for _ in range(reps):
            pb = pos[rng.integers(0, pos.size, pos.size)]
            nb = neg[rng.integers(0, neg.size, neg.size)]
            m = verification_metrics(pb, nb, fmrs)
            for key in keys:
                acc[key].append(m[key])

    return {
        k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        for k, v in acc.items()
        if v
    }


# -------------------------------------------------------------------- the grid
def attach_delta_eer(df):
    """Add ``eer_aligned`` and ``delta_eer = eer - eer_aligned`` to a metrics frame.

    The reference is the clean aligned source (``kind == "aligned"``, empty suffix)
    of the same ``(dataset, model, res)``; crop-variant aligned sources are not
    references.
    """
    aligned = df[df.kind == "aligned"]
    if "suffix" in aligned.columns:
        aligned = aligned[aligned["suffix"].fillna("") == ""]
    base = aligned.set_index(["dataset", "model", "res"])["eer"].to_dict()
    df["eer_aligned"] = [
        base.get((r.dataset, r.model, r.res), np.nan) for r in df.itertuples()
    ]
    df["delta_eer"] = df["eer"] - df["eer_aligned"]
    return df


def accuracy_rows(  # noqa: C901 - one loop over models x sources with CI options
    dataset: str,
    models: Iterable[str],
    *,
    sample_nonmated: int = NONMATED_SAMPLE["accuracy"],
    bootstrap: int = BOOTSTRAP_REPS,
    boot_nonmated: int = BOOTSTRAP_NONMATED_CAP,
    boot_level: str = "subject",
    boot_models: Iterable[str] | None = BOOTSTRAP_MODELS,
    seed: int = 0,
    device=None,
    cpu_fallback: bool = True,
    tags: Iterable[str] | None = None,
    pairs: Pairs | None = None,
    median_bytes: Callable[[dict], float] | None = None,
) -> list[dict]:
    """Score every embedding source of ``models`` on one dataset (``metrics.csv`` rows).

    Parameters
    ----------
    dataset : str
        Dataset name.
    models : iterable of str
        Matchers to score (see :func:`face1kb.eval.embeddings.resolve_models`).
    sample_nonmated : int
        Non-mated sample size (seed ``seed``).
    bootstrap : int
        Bootstrap replicates for the CI columns (0 disables them).
    boot_nonmated : int
        Impostor cap of the bootstrap.
    boot_level : {"subject", "pair"}
        Resampling unit of the bootstrap.
    boot_models : iterable of str or None
        Matchers that get CIs (clean cells only); ``None`` or empty = all.
    seed : int
        Seed of the pair sample and of the bootstrap.
    device : optional
        Torch device for scoring (the published grid used ``"cuda"``); ``None``
        scores with NumPy.
    cpu_fallback : bool
        Whether a CUDA scorer that runs out of memory at the smallest chunk may
        score that call with NumPy (see :func:`cosine_scores_torch`); ``False``
        raises instead, which guarantees CUDA-scorer numbers.
    tags : iterable of str, optional
        Restrict to these source tags.
    pairs : tuple, optional
        Precomputed :func:`pair_indices` result (saves re-reading the pairs).
    median_bytes : callable, optional
        ``meta -> float`` filling the ``median_bytes`` column (NaN otherwise).

    Returns
    -------
    list of dict
        One row per ``(model, source)`` in model order, then tag order;
        :func:`metrics_frame` turns rows of one or more datasets into the
        ``metrics.csv`` frame.

    Raises
    ------
    ValueError
        If an embedding array does not have one row per index entry.
    RuntimeError
        If ``cpu_fallback`` is false and the device cannot score a source.
    """
    (p1, p2), (n1, n2) = (
        pairs if pairs is not None else pair_indices(dataset, sample_nonmated, seed)
    )
    log.info("[%s] mated=%d sampled-nonmated=%d", dataset, p1.size, n1.size)
    score = make_scorer(device, cpu_fallback=cpu_fallback)
    index = read_index_or_none(dataset)
    subj = None
    if boot_level == "subject":
        if index is not None:
            subj = index["subject"].to_numpy()
        else:
            log.warning("[%s] no index: bootstrap falls back to pairs", dataset)
    pos_subj = subj[p1] if subj is not None else None
    neg_subj = subj[n1] if subj is not None else None
    boot_set = set(boot_models or ())
    tag_set = set(tags) if tags is not None else None
    rows = []
    for model in models:
        for path, meta in list_sources(dataset, model, aligned_codec=""):
            if tag_set is not None and path.stem not in tag_set:
                continue
            emb = np.load(path)
            if emb.shape[1] != EMB_DIM:
                continue
            if index is not None:
                check_rows(emb, len(index), f"{model}/{path.stem}")
            pos = score(emb, p1, p2)
            neg = score(emb, n1, n2)
            met = verification_metrics(pos, neg, FMR_TARGETS)
            want_ci = (not boot_set or model in boot_set) and meta["suffix"] == ""
            ci = bootstrap_ci(
                pos,
                neg,
                FMR_TARGETS,
                bootstrap if want_ci else 0,
                boot_nonmated,
                seed,
                pos_subj=pos_subj,
                neg_subj=neg_subj,
                level=boot_level,
            )
            row = {"dataset": dataset, "model": model, "tag": path.stem, **meta}
            row.update(met)
            for k, (lo, hi) in ci.items():
                row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
            row["median_bytes"] = median_bytes(meta) if median_bytes else np.nan
            rows.append(row)
            log.info(
                "  %-22s %-28s eer=%.4f fnmr@1e-3=%.4f",
                model,
                path.stem,
                met["eer"],
                met.get("fnmr_0.001", float("nan")),
            )
    return rows


def metrics_frame(rows: list[dict]):
    """Rows of :func:`accuracy_rows` (any datasets) as a ``metrics.csv`` frame.

    Adds ``eer_aligned``/``delta_eer`` and orders the columns as
    :data:`METRICS_COLUMNS` (extra columns are kept at the end). No rows give an
    empty frame with these columns.
    """
    import pandas as pd  # noqa: PLC0415

    if not rows:
        return pd.DataFrame(columns=list(METRICS_COLUMNS))
    df = attach_delta_eer(pd.DataFrame(rows))
    for c in METRICS_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    extra = [c for c in df.columns if c not in METRICS_COLUMNS]
    return df[list(METRICS_COLUMNS) + extra]


def write_merged_csv(frames: Iterable, path) -> None:
    """Write finished per-part frames as one CSV, the way the published files were.

    The published ``metrics.csv`` and ``significance_<dataset>.csv`` were assembled
    from per-dataset (or per-shard) CSV files: each part was written, read back with
    pandas' default float parser, and the parts were concatenated and written once
    more. That parser is not exactly round-trip, so the printed text of some values
    depends on this single read/write. This function applies it in memory.

    Every frame must already be final: derived columns (``eer_aligned`` and
    ``delta_eer`` of :func:`metrics_frame`, ``p_adj`` and ``significant`` of
    :func:`face1kb.eval.significance.significance_frame`) are computed on the exact
    values before the round trip and are never recomputed from parsed values, which
    would change their text.

    Parameters
    ----------
    frames : iterable of pandas.DataFrame
        Final frames, one per dataset or shard, in output order.
    path : path-like
        Output CSV.
    """
    import io  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    parts = [pd.read_csv(io.StringIO(df.to_csv(index=False))) for df in frames]
    if not parts:
        raise ValueError("no frames to write")
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)


def write_metrics_csv(frames: Iterable, path) -> None:
    """Write ``metrics.csv`` from per-dataset :func:`metrics_frame` outputs.

    Run :func:`metrics_frame` on each dataset's in-memory rows first, then pass the
    frames here; see :func:`write_merged_csv` for the single read/write that makes the
    text identical to the published file. Do not run :func:`metrics_frame` or
    :func:`attach_delta_eer` again on a frame read back from a CSV.
    """
    write_merged_csv(frames, path)
