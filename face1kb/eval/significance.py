# SPDX-License-Identifier: MIT
"""Paired significance tests between codecs, BH-FDR, and cross-matcher rank statistics.

Paired tests (one ``(dataset, model, res, budget)`` cell)
--------------------------------------------------------
All sources of a cell (every codec plus the aligned baseline) are scored on one shared
trial set: all mated pairs plus a seeded sample of 2,000,000 non-mated pairs
(:func:`face1kb.eval.verification.pair_indices`, seed 0). Sources with fewer than half
of their trials valid (non-NaN) are dropped, and the rest are restricted to the trials
valid for every remaining source, so every comparison is exactly paired
(:func:`finalize_cell`). Each source decides at its own EER threshold on that set
(:func:`face1kb.eval.verification.eer_threshold`): a mated trial is correct when
``score >= t``, a non-mated one when ``score < t``.

* **McNemar** (:func:`mcnemar`): discordant counts ``b`` (A correct, B wrong) and
  ``c`` (A wrong, B correct); continuity-corrected ``chi2 = (|b - c| - 1)^2 / (b + c)``
  with a chi-square(1) p-value. The exact binomial p is also computed.
* **DeLong** (:func:`delong_paired`): paired test of the ROC-AUC difference (mated =
  positive) with the structural components of Sun & Xu (2014); two-sided normal p.
* **BH-FDR** (:func:`bh_fdr`, :func:`correct_family`): McNemar p-values are adjusted
  per family; ``significant = p_adj < 0.05``. The published tables use one family per
  cell (55 tests with 11 sources), :data:`CELL_FAMILY`.

Rank statistics over matchers (:func:`posthoc`)
-----------------------------------------------
From the matcher x codec EER matrix of one cell of ``metrics.csv``: Friedman test,
Kendall's W = chi2 / (n (k - 1)), mean ranks (1 = lowest EER), and for headline codec
pairs a Wilcoxon signed-rank test with Holm correction plus Cliff's delta.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Sequence

import numpy as np

from .embeddings import EMB_DIM, check_rows, list_sources
from .verification import (
    NONMATED_SAMPLE,
    eer_threshold,
    make_scorer,
    pair_indices,
    read_index_or_none,
    write_merged_csv,
)

log = logging.getLogger(__name__)

#: Minimum fraction of valid (non-NaN) trials for a source to enter a cell.
MIN_COVERAGE = 0.5
#: Significance level applied to the BH-adjusted p-values.
ALPHA = 0.05
#: BH family of the published significance tables: one family per cell.
CELL_FAMILY: tuple[str, ...] = ("dataset", "model", "res", "budget")
#: Column order of ``accuracy/significance_<dataset>.csv``.
SIGNIFICANCE_COLUMNS: tuple[str, ...] = (
    "dataset",
    "model",
    "res",
    "budget",
    "codec_a",
    "codec_b",
    "n_trials",
    "acc_a",
    "acc_b",
    "b",
    "c",
    "mcnemar_chi2",
    "mcnemar_p",
    "auc_a",
    "auc_b",
    "auc_diff",
    "auc_test",
    "auc_p",
    "p_adj",
    "significant",
)


# ------------------------------------------------------------------- decisions
def correct_vector(pos: np.ndarray, neg: np.ndarray, thr: float) -> np.ndarray:
    """Per-trial correctness at ``thr``: mated trials first, then non-mated."""
    return np.concatenate([pos >= thr, neg < thr])


# ---------------------------------------------------------------------- DeLong
def midrank(x: np.ndarray) -> np.ndarray:
    """1-based mid-ranks of ``x`` (tied values share the mean of their ranks).

    Vectorised; returns exactly the ranks of the straightforward loop over the
    sorted values (every rank is a half-integer, computed as
    ``0.5 * (i + j - 1) + 1`` for the tie block ``[i, j)`` of the sorted order).
    Each NaN forms a block of its own.
    """
    x = np.asarray(x)
    n = x.size
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    # block starts: first element, and every element that differs from its
    # predecessor (NaN != NaN, so NaNs are singleton blocks)
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(xs[1:], xs[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n)
    block_rank = 0.5 * (starts + ends - 1) + 1.0
    out[order] = np.repeat(block_rank, ends - starts)
    return out


def delong_components(pos: np.ndarray, neg: np.ndarray):
    """AUC and the DeLong placement values ``(v10, v01)`` of one classifier.

    ``v10`` (one per positive) and ``v01`` (one per negative) are the structural
    components whose covariances give the DeLong variance (Sun & Xu, 2014).
    """
    m, n = pos.size, neg.size
    tx = midrank(pos)
    ty = midrank(neg)
    tz = midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
    v10 = (tz[:m] - tx) / n
    v01 = 1.0 - (tz[m:] - ty) / m
    return float(auc), v10, v01


def delong_paired(pos_a, neg_a, pos_b, neg_b):
    """DeLong's paired two-sided test of ``AUC_a - AUC_b`` on the same trials.

    Returns
    -------
    tuple
        ``(auc_a, auc_b, auc_diff, p)``. With a zero or non-finite variance ``p`` is
        0 when the AUCs differ and 1 when they are equal.
    """
    from scipy import stats  # noqa: PLC0415

    auc_a, v10a, v01a = delong_components(pos_a, neg_a)
    auc_b, v10b, v01b = delong_components(pos_b, neg_b)
    m, n = pos_a.size, neg_a.size
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    s = s10 / m + s01 / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    diff = auc_a - auc_b
    if not np.isfinite(var) or var <= 0:
        p = 0.0 if diff != 0 else 1.0
        return auc_a, auc_b, diff, p
    z = diff / np.sqrt(var)
    p = float(2.0 * stats.norm.sf(abs(z)))
    return auc_a, auc_b, diff, p


def auc_bootstrap_paired(pos_a, neg_a, pos_b, neg_b, reps: int, seed: int = 0):
    """Paired bootstrap of the AUC difference (an alternative to DeLong).

    Positives and negatives are resampled with the same indices for both
    classifiers. Returns ``(auc_a, auc_b, diff, lo, hi, p)`` with the percentile
    95 % interval and a two-sided bootstrap p.
    """
    auc_a, _, _ = delong_components(pos_a, neg_a)
    auc_b, _, _ = delong_components(pos_b, neg_b)
    diff = auc_a - auc_b
    rng = np.random.default_rng(seed)
    m, n = pos_a.size, neg_a.size
    diffs = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        pi = rng.integers(0, m, m)
        ni = rng.integers(0, n, n)
        da, _, _ = delong_components(pos_a[pi], neg_a[ni])
        db, _, _ = delong_components(pos_b[pi], neg_b[ni])
        diffs[r] = da - db
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    frac = min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = float(min(1.0, 2.0 * frac))
    return auc_a, auc_b, diff, float(lo), float(hi), p


# --------------------------------------------------------------------- McNemar
def mcnemar(ca: np.ndarray, cb: np.ndarray):
    """McNemar test of two paired correctness vectors.

    Returns
    -------
    tuple
        ``(b, c, chi2, p_chi2, p_exact)``: discordant counts, the
        continuity-corrected statistic, its chi-square(1) p-value and the exact
        two-sided binomial p. Without discordant trials: ``(b, c, 0, 1, 1)``.
    """
    from scipy import stats  # noqa: PLC0415

    b = int(np.count_nonzero(ca & ~cb))
    c = int(np.count_nonzero(~ca & cb))
    nd = b + c
    if nd == 0:
        return b, c, 0.0, 1.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / nd
    p_chi2 = float(stats.chi2.sf(chi2, 1))
    p_exact = float(stats.binomtest(b, nd, 0.5, alternative="two-sided").pvalue)
    return b, c, float(chi2), p_chi2, p_exact


# ---------------------------------------------------------------------- BH-FDR
def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (monotone, clipped to 1)."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.minimum(ranked, 1.0)
    return out


def correct_family(df, family: Sequence[str] = CELL_FAMILY, alpha: float = ALPHA):
    """Add ``p_adj`` (BH of ``mcnemar_p`` per ``family`` group) and ``significant``.

    The default family is one ``(dataset, model, res, budget)`` cell, as in the
    published tables; ``family=("dataset", "model")`` pools all cells of a matcher.
    """
    df = df.copy()
    df["p_adj"] = np.nan
    for _, idx in df.groupby(list(family)).groups.items():
        sub = df.loc[idx]
        df.loc[idx, "p_adj"] = bh_fdr(sub["mcnemar_p"].to_numpy())
    df["significant"] = df["p_adj"] < alpha
    return df


# ----------------------------------------------------------------------- cells
def finalize_cell(raws: dict) -> dict:
    """Restrict the sources of a cell to their common valid trials and decide.

    Parameters
    ----------
    raws : dict
        ``codec -> {"pos", "neg", "valid"}`` with ``valid`` laid out as
        ``[mated trials..., non-mated trials...]``.

    Returns
    -------
    dict
        ``codec -> {"pos", "neg", "correct", "acc"}`` on the common subset, or ``{}``
        when it has no mated or no non-mated trial.
    """
    pos_masks = [r["valid"][: r["pos"].size] for r in raws.values()]
    neg_masks = [r["valid"][r["pos"].size :] for r in raws.values()]
    keep_p = np.logical_and.reduce(pos_masks)
    keep_n = np.logical_and.reduce(neg_masks)
    if not keep_p.any() or not keep_n.any():
        return {}
    final: dict = {}
    for name, r in raws.items():
        pos = r["pos"][keep_p]
        neg = r["neg"][keep_n]
        thr = eer_threshold(pos, neg)
        correct = correct_vector(pos, neg, thr)
        final[name] = {
            "pos": pos,
            "neg": neg,
            "correct": correct,
            "acc": float(correct.mean()),
        }
    return final


def cell_rows(dataset: str, model: str, res: int, budget: int, sources: dict) -> list:
    """McNemar and DeLong rows for every unordered pair of sources (sorted names)."""
    names = sorted(sources)
    rows = []
    for a, b in itertools.combinations(names, 2):
        pa, pb = sources[a], sources[b]
        bb, cc, chi2, p_chi2, p_exact = mcnemar(pa["correct"], pb["correct"])
        auc_a, auc_b, auc_diff, auc_p = delong_paired(
            pa["pos"], pa["neg"], pb["pos"], pb["neg"]
        )
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "res": res,
                "budget": budget,
                "codec_a": a,
                "codec_b": b,
                "n_trials": int(pa["correct"].size),
                "acc_a": pa["acc"],
                "acc_b": pb["acc"],
                "b": bb,
                "c": cc,
                "mcnemar_chi2": chi2,
                "mcnemar_p": p_chi2,
                "mcnemar_p_exact": p_exact,
                "auc_a": auc_a,
                "auc_b": auc_b,
                "auc_diff": auc_diff,
                "auc_test": "delong",
                "auc_p": auc_p,
            }
        )
    return rows


def load_cell_sources(
    dataset: str,
    model: str,
    score,
    pairs,
    *,
    res: Iterable[int] = (),
    budgets: Iterable[int] = (),
    min_coverage: float = MIN_COVERAGE,
    n_index: int | None = None,
) -> dict:
    """Score the core-grid sources of one model on the shared trial set.

    Returns ``{res: {budget: {codec: raw}, "aligned": raw}}`` with
    ``raw = {"pos", "neg", "valid"}``. Suffix variants are ignored; sources below
    ``min_coverage`` valid trials are dropped. With ``n_index`` set, an array
    without exactly that many rows raises ``ValueError``
    (:func:`face1kb.eval.embeddings.check_rows`).
    """
    (p1, p2), (n1, n2) = pairs
    res_f, bud_f = set(res), set(budgets)
    cells: dict = {}
    for path, meta in list_sources(dataset, model):
        if meta["suffix"]:
            continue
        if res_f and meta["res"] not in res_f:
            continue
        if meta["kind"] == "compressed" and bud_f and meta["budget"] not in bud_f:
            continue
        emb = np.load(path)
        if emb.shape[1] != EMB_DIM:
            continue
        if n_index is not None:
            check_rows(emb, n_index, f"{model}/{path.stem}")
        pos = score(emb, p1, p2)
        neg = score(emb, n1, n2)
        valid = np.concatenate([~np.isnan(pos), ~np.isnan(neg)])
        coverage = float(valid.mean())
        if coverage < min_coverage:
            log.info("  [skip] %s %s: coverage %.3f", model, path.stem, coverage)
            continue
        raw = {"pos": pos, "neg": neg, "valid": valid}
        res_cell = cells.setdefault(meta["res"], {})
        if meta["kind"] == "aligned":
            res_cell["aligned"] = raw
        else:
            res_cell.setdefault(meta["budget"], {})[meta["codec"]] = raw
    return cells


def significance_rows(
    dataset: str,
    models: Iterable[str],
    *,
    res: Iterable[int] = (),
    budgets: Iterable[int] = (),
    sample_nonmated: int = NONMATED_SAMPLE["significance"],
    seed: int = 0,
    min_coverage: float = MIN_COVERAGE,
    device=None,
    cpu_fallback: bool = True,
    pairs=None,
) -> list[dict]:
    """Paired McNemar/DeLong rows of every requested cell of one dataset.

    Returns the rows without ``p_adj``; :func:`significance_frame` applies
    :func:`correct_family` (per cell, as published) and selects
    :data:`SIGNIFICANCE_COLUMNS`. The list is empty when no cell has two usable
    sources. Raises ``ValueError`` if an embedding array does not have one row per
    index entry. ``device`` selects the scorer (``None`` = NumPy, as published);
    ``cpu_fallback`` is passed to :func:`~face1kb.eval.verification.make_scorer`.
    """
    res, budgets = tuple(res), tuple(budgets)
    if pairs is None:
        pairs = pair_indices(dataset, sample_nonmated, seed)
    log.info("[%s] mated=%d sampled-nonmated=%d", dataset, *(p[0].size for p in pairs))
    index = read_index_or_none(dataset)
    n_index = None if index is None else len(index)
    score = make_scorer(device, cpu_fallback=cpu_fallback)
    rows: list[dict] = []
    for model in models:
        cells = load_cell_sources(
            dataset,
            model,
            score,
            pairs,
            res=res,
            budgets=budgets,
            min_coverage=min_coverage,
            n_index=n_index,
        )
        for r, by_budget in cells.items():
            aligned = by_budget.get("aligned")
            for budget in [k for k in by_budget if isinstance(k, int)]:
                raws = dict(by_budget[budget])
                if aligned is not None:
                    raws["aligned"] = aligned
                if len(raws) < 2:
                    continue
                sources = finalize_cell(raws)
                if len(sources) < 2:
                    log.info(
                        "  [skip] %s res=%s budget=%s: no common trials",
                        model,
                        r,
                        budget,
                    )
                    continue
                rows += cell_rows(dataset, model, r, budget, sources)
                log.info(
                    "  %-22s res=%s budget=%s sources=%d n_trials=%d",
                    model,
                    r,
                    budget,
                    len(sources),
                    sources[next(iter(sources))]["correct"].size,
                )
    return rows


def significance_frame(rows: list[dict], family: Sequence[str] = CELL_FAMILY):
    """Rows of :func:`significance_rows` as the published CSV frame (BH per family).

    No rows give an empty frame with :data:`SIGNIFICANCE_COLUMNS`.
    """
    import pandas as pd  # noqa: PLC0415

    if not rows:
        return pd.DataFrame(columns=list(SIGNIFICANCE_COLUMNS))
    return correct_family(pd.DataFrame(rows), family)[list(SIGNIFICANCE_COLUMNS)]


def write_significance_csv(frames: Iterable, path) -> None:
    """Write ``significance_<dataset>.csv`` from per-shard :func:`significance_frame`.

    Apply :func:`significance_frame` (BH per cell) to each shard's in-memory rows,
    then pass the frames here; they get the single default-parser read/write of the
    published files (:func:`face1kb.eval.verification.write_merged_csv`). Never
    re-apply :func:`correct_family` to values read back from a CSV.
    """
    write_merged_csv(frames, path)


# ------------------------------------------------------------ rank statistics
#: Codecs of the cross-matcher ranking (CF, 112 px, 1024 B).
POSTHOC_CODECS: tuple[str, ...] = (
    "jpeg_ai",
    "jpeg",
    "jpeg2000",
    "webp",
    "jpeg_xl",
    "avif",
    "heif",
    "jpeg_fzt",
    "neural_bmshj2018",
    "neural_mbt2018_mean",
    "ours_fast",
    "ours_accurate",
)
#: Headline codec pairs of the Holm-Wilcoxon table.
POSTHOC_PAIRS: tuple[tuple[str, str], ...] = (
    ("ours_accurate", "webp"),
    ("ours_accurate", "avif"),
    ("ours_accurate", "jpeg2000"),
    ("webp", "jpeg2000"),
    ("ours_accurate", "ours_fast"),
    ("ours_accurate", "jpeg_fzt"),
    ("jpeg_fzt", "webp"),
    ("jpeg", "jpeg2000"),
)


def cliffs_delta(a, b) -> float:
    """Cliff's delta: ``(#(a_i > b_j) - #(a_i < b_j)) / (len(a) len(b))``."""
    a, b = np.asarray(a), np.asarray(b)
    gt = int(np.count_nonzero(a[:, None] > b[None, :]))
    lt = int(np.count_nonzero(a[:, None] < b[None, :]))
    return (gt - lt) / (len(a) * len(b))


def holm_adjust(pvals: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, clipped to 1)."""
    raw = list(pvals)
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    holm = [0.0] * len(raw)
    mfam = len(raw)
    for rank, i in enumerate(order):
        holm[i] = min(1.0, raw[i] * (mfam - rank))
    for j in range(1, len(order)):
        holm[order[j]] = max(holm[order[j]], holm[order[j - 1]])
    return holm


def eer_matrix(
    metrics,
    *,
    dataset: str = "colorferet",
    res: int = 112,
    budget: int = 1024,
    codecs: Sequence[str] = POSTHOC_CODECS,
    exclude_prefix: str | None = None,
):
    """Matcher x codec EER matrix of one clean cell of a ``metrics.csv`` frame.

    The columns are the ``codecs`` that have at least one EER in the cell (a codec
    whose rows are all NaN, e.g. an empty cell, is left out). Only matchers with an
    EER for every such codec are kept (sorted by name). ``exclude_prefix`` drops
    matchers whose name starts with it (e.g. ``"edgeface"``).

    Raises
    ------
    ValueError
        If the cell has no EER at all or no matcher covers every codec.
    """
    import pandas as pd  # noqa: PLC0415

    df = metrics[
        (metrics.dataset == dataset)
        & (metrics.res == res)
        & (metrics.budget == budget)
        & (metrics.kind == "compressed")
        & (metrics.suffix.isna() | (metrics.suffix == ""))
    ]
    if exclude_prefix:
        df = df[~df.model.str.startswith(exclude_prefix)]
    df = df[df.eer.notna()]
    present = set(df.codec)
    cols = [c for c in codecs if c in present]
    mat = {}
    for m in sorted(set(df.model)):
        row = {}
        for c in cols:
            r = df[(df.model == m) & (df.codec == c)]
            if not r.empty:
                row[c] = float(r.iloc[0].eer)
        if len(row) == len(cols):
            mat[m] = row
    if not cols or not mat:
        raise ValueError(
            f"eer_matrix: no matcher has an EER for every codec of the cell "
            f"({dataset}, {res} px, {budget} B)"
        )
    return pd.DataFrame(mat).T[cols]


def posthoc(matrix, pairs: Sequence[tuple[str, str]] = POSTHOC_PAIRS) -> dict:
    """Friedman / Kendall's W / mean ranks and Holm-Wilcoxon + Cliff's delta.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Matchers x codecs EER matrix (:func:`eer_matrix`).
    pairs : sequence of (str, str)
        Codec pairs to test; pairs with a codec missing from ``matrix`` are skipped.

    Returns
    -------
    dict
        ``n``, ``k``, ``friedman_chi2``, ``friedman_p``, ``kendall_w``,
        ``mean_rank`` (Series, ascending; 1 = lowest EER) and ``pairs``: a list of
        ``(a, b, p_wilcoxon, p_holm, cliffs_delta)``.
    """
    from scipy.stats import friedmanchisquare, wilcoxon  # noqa: PLC0415

    codecs = list(matrix.columns)
    n, k = matrix.shape
    chi2, p = friedmanchisquare(*[matrix[c].values for c in codecs])
    w = chi2 / (n * (k - 1))
    mean_rank = matrix.rank(axis=1, method="average").mean(axis=0).sort_values()
    raw = []
    for a, b in pairs:
        if a not in codecs or b not in codecs:
            continue
        try:
            _, pw = wilcoxon(matrix[a].values, matrix[b].values)
        except ValueError:
            pw = 1.0
        raw.append((a, b, pw, cliffs_delta(matrix[a].values, matrix[b].values)))
    holm = holm_adjust([r[2] for r in raw])
    return {
        "n": n,
        "k": k,
        "friedman_chi2": float(chi2),
        "friedman_p": float(p),
        "kendall_w": float(w),
        "mean_rank": mean_rank,
        "pairs": [(a, b, pw, ph, d) for (a, b, pw, d), ph in zip(raw, holm)],
    }
