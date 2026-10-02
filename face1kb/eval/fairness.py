# SPDX-License-Identifier: MIT
"""Subgroup (demographic) verification accuracy, disparity measures and their CIs.

Attributes (one value per index row, :func:`load_attributes`)
------------------------------------------------------------
* Color FERET, from the NIST ground-truth labels (``labels.csv``, joined on the image
  stem): ``pose`` -- coarse class of the textual pose (:func:`pose_class`: frontal /
  quarter / half / profile); ``skin_tone`` -- the ``race`` field; ``gender``; ``age``
  -- the age at capture binned into :data:`AGE_LABELS`.
* AI-Solutions-KK, from the estimated attributes (``attributes.csv``), propagated per
  identity: ``skin_tone`` -- the first non-empty Monk Skin Tone label of the
  identity; ``gender`` -- the per-identity mode; ``age`` -- the band of the
  per-identity median age. KK has no pose attribute.

Subgroup EER (:func:`subgroup_rows`)
------------------------------------
A pair belongs to subgroup ``g`` when both of its images carry ``g``. Per attribute
the rows are an ``__overall__`` row (all pairs whose two ends carry the same
non-null value) and one row per subgroup with at least ``min_mated`` (50) mated
pairs and one impostor pair. The disparity of an attribute (:func:`disparity_row`)
is the max - min (and the std) of the subgroup EERs over subgroups with at least
:data:`MIN_NEG_DISPARITY` (1000) impostor pairs.

Differential FMR (:func:`fmr_rows`)
-----------------------------------
A global threshold ``tau`` is set at a target FMR (1e-2, 1e-3) as the
``1 - target`` quantile of all within-subgroup impostor scores of the attribute;
FMR(g) = P(impostor score >= tau | both ends in g) for subgroups with at least
``min_neg`` (50) impostor pairs, and the spread is max - min in percentage points.
``tau`` is NaN (and every FMR 0) when a pair of the pool touches a crop without an
embedding, as in the published tables; ``nan_policy="omit"`` drops such pairs.

Disparity CIs (:func:`disparity_ci_rows`)
----------------------------------------
Max - min EER (pp) over the uniform Monk basis MST5/6/7/9/10 of AI-Solutions-KK, with
a cluster bootstrap over identities (500 replicates, seed 0, 400,000 non-mated
pairs): a pair stays in a replicate iff the identities of both of its ends were
drawn at least once. Here a mated pair belongs to ``g`` through its (shared)
identity and a non-mated pair through the skin tone of its first image (``idx1``)
only, so the impostors of a subgroup include cross-tone pairs; a subgroup needs 20
mated pairs and one impostor pair in a replicate. The amplification ratio divides a
codec's disparity by the median of the aligned bootstrap disparities.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from face1kb import config

from .embeddings import ANCHOR_MODELS, EMB_DIM, check_rows, list_sources
from .verification import (
    NONMATED_SAMPLE,
    eer,
    linear_quantile,
    make_scorer,
    pair_indices,
)

log = logging.getLogger(__name__)

#: Subgroup name of the all-groups row.
OVERALL = "__overall__"
#: Impostor floor for a subgroup to enter the max-min disparity.
MIN_NEG_DISPARITY = 1000
#: Default minimum mated pairs of a scored subgroup.
MIN_MATED = 50
#: Default minimum impostor pairs of a subgroup in the FMR analysis.
MIN_NEG_FMR = 50
#: Target FMRs of the differential-FMR analysis.
FMR_FAIRNESS_TARGETS: tuple[float, ...] = (0.01, 0.001)
#: Uniform Monk Skin Tone basis of the AI-Solutions-KK disparity (MST8 excluded).
MONK_UNIFORM_BASIS: tuple[str, ...] = ("MST5", "MST6", "MST7", "MST9", "MST10")
#: Bootstrap replicates of the disparity CIs.
DISPARITY_CI_REPS = 500
#: Minimum mated pairs of a subgroup inside a disparity-CI replicate.
DISPARITY_CI_MIN_MATED = 20
#: Codecs of the disparity-CI table (112 px, 1024 B), aligned reference first.
DISPARITY_CI_CODECS: tuple[str, ...] = (
    "aligned",
    "webp",
    "avif",
    "heif",
    "jpeg",
    "jpeg_xl",
    "jpeg_fzt",
    "jpeg2000",
    "ours_accurate",
    "ours_fast",
)

#: Columns of ``accuracy/fairness_<dataset>.csv``.
FAIRNESS_COLUMNS: tuple[str, ...] = (
    "dataset",
    "model",
    "res",
    "budget",
    "codec",
    "attribute",
    "subgroup",
    "n_pos",
    "n_neg",
    "eer",
)
#: Columns of ``accuracy/fairness_disparity_<dataset>.csv``.
DISPARITY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "model",
    "res",
    "budget",
    "codec",
    "attribute",
    "eer_min",
    "eer_max",
    "eer_std",
    "n_subgroups",
)
#: Columns of ``accuracy/fmr_fairness_<dataset>.csv``.
FMR_FAIRNESS_COLUMNS: tuple[str, ...] = (
    "dataset",
    "model",
    "res",
    "budget",
    "codec",
    "attribute",
    "subgroup",
    "n_neg",
    "tau",
    "target_fmr",
    "fmr",
    "fmr_min",
    "fmr_max",
    "fmr_disparity_pp",
    "n_subgroups",
)
#: Columns of ``accuracy/disparity_ci_kk.csv``.
DISPARITY_CI_COLUMNS: tuple[str, ...] = (
    "anchor",
    "codec",
    "disparity_pp",
    "disp_lo",
    "disp_hi",
    "n_boot",
    "ratio",
    "ratio_lo",
    "ratio_hi",
)

# coarse pose class from the Color FERET textual pose (first matching rule wins)
POSE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("profile", ("profile", "75 degree")),
    ("quarter", ("quarter", "45 degree")),
    ("half", ("half",)),
    ("frontal", ("frontal", "15 degree")),
)
#: Age-band edges (years, right-closed, lowest included) and labels.
AGE_BINS: tuple[int, ...] = (0, 25, 35, 50, 65, 200)
AGE_LABELS: tuple[str, ...] = ("<=25", "26-35", "36-50", "51-65", ">65")


# ------------------------------------------------------------------ attributes
def pose_class(pose) -> str | None:
    """Coarse pose class of a Color FERET textual pose, or ``None``."""
    if not isinstance(pose, str):
        return None
    low = pose.lower()
    for cls, keys in POSE_RULES:
        if any(k in low for k in keys):
            return cls
    return None


def age_band(age):
    """Bin ages (a pandas Series) into :data:`AGE_LABELS`."""
    import pandas as pd  # noqa: PLC0415

    return pd.cut(age, list(AGE_BINS), labels=list(AGE_LABELS), include_lowest=True)


def attributes_colorferet(index, labels) -> dict[str, np.ndarray]:
    """Per-row Color FERET attributes, aligned to the index order.

    Parameters
    ----------
    index : pandas.DataFrame
        Dataset index (``rel_path`` per row; :func:`face1kb.config.read_index`).
    labels : pandas.DataFrame
        NIST ground-truth labels with ``image`` (file stem), ``pose``, ``gender``,
        ``race`` and either ``age_from``/``age_to`` or ``age``.

    Returns
    -------
    dict of str to numpy.ndarray
        ``pose``, ``skin_tone``, ``gender``, ``age`` object arrays (``None``/NaN =
        unknown).
    """
    stems = [Path(p).stem for p in index["rel_path"]]
    left = index[[]].assign(image=stems)
    cols = ["image", "pose", "gender", "race"]
    cols += [c for c in ("age_from", "age_to", "age") if c in labels.columns]
    m = left.merge(labels[cols], on="image", how="left")
    if len(m) != len(index):
        raise ValueError(f"colorferet label join changed length: {len(m)}")
    if "age_from" in m.columns and "age_to" in m.columns:
        mid = (m["age_from"] + m["age_to"]) / 2.0
    else:
        mid = m["age"] * 1.0
    return {
        "pose": m["pose"].map(pose_class).to_numpy(dtype=object),
        "skin_tone": m["race"].astype(object).where(m["race"].notna()).to_numpy(),
        "gender": m["gender"].astype(object).where(m["gender"].notna()).to_numpy(),
        "age": age_band(mid).astype(object).where(mid.notna()).to_numpy(),
    }


def attributes_kk(index, attributes) -> dict[str, np.ndarray]:
    """Per-row AI-Solutions-KK attributes, propagated per identity.

    Parameters
    ----------
    index : pandas.DataFrame
        Dataset index with the identity in ``subject``.
    attributes : pandas.DataFrame
        Estimated attributes with ``identity`` (or ``subject``), ``mst_label``,
        ``gender`` and ``age`` (a subset of the images may be labelled).

    Returns
    -------
    dict of str to numpy.ndarray
        ``skin_tone`` (first non-empty MST label of the identity), ``gender``
        (per-identity mode) and ``age`` (band of the per-identity median).
    """
    import pandas as pd  # noqa: PLC0415

    key = "identity" if "identity" in attributes.columns else "subject"

    def _first(s):
        s = s.dropna()
        return s.iloc[0] if s.size else np.nan

    def _mode(s):
        s = s.dropna()
        return s.mode().iloc[0] if s.size else np.nan

    g = attributes.groupby(key)
    mst = g["mst_label"].agg(_first)
    gender = g["gender"].agg(_mode)
    age_med = g["age"].agg(lambda s: s.dropna().median() if s.dropna().size else np.nan)
    band = age_band(age_med)
    ident = index["subject"]
    return {
        "skin_tone": ident.map(mst).astype(object).to_numpy(),
        "gender": ident.map(gender).astype(object).to_numpy(),
        "age": ident.map(pd.Series(band.values, index=age_med.index))
        .astype(object)
        .to_numpy(),
    }


def load_attributes(dataset: str, n_rows: int | None = None) -> dict[str, np.ndarray]:
    """Per-row attributes of ``dataset`` from its index and label/attribute file.

    ``n_rows`` (the embedding row count) is checked against the index length.
    """
    import pandas as pd  # noqa: PLC0415

    index = config.read_index(dataset)
    if n_rows is not None and len(index) != n_rows:
        raise ValueError(f"index ({len(index)}) != embeddings ({n_rows})")
    if dataset == "colorferet":
        return attributes_colorferet(index, pd.read_csv(config.labels_csv(dataset)))
    if dataset == "kk":
        # Subjects are strings, as in config.read_index.
        attrs = pd.read_csv(
            config.attributes_csv(dataset), dtype={"subject": str, "identity": str}
        )
        return attributes_kk(index, attrs)
    raise ValueError(f"no attributes for dataset {dataset!r}")


# ----------------------------------------------------------- subgroup accuracy
def _same_group(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Pairs whose two ends carry the same non-null attribute value."""
    import pandas as pd  # noqa: PLC0415

    return (a1 == a2) & np.not_equal(a1, None) & pd.notna(a1)


def subgroup_rows(
    attr_name: str,
    attr_vals: np.ndarray,
    pos_idx,
    neg_idx,
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    min_mated: int = MIN_MATED,
) -> list[dict]:
    """EER rows of every subgroup of one attribute plus the ``__overall__`` row.

    A pair is in subgroup ``g`` iff both ends carry ``g``. Subgroups with fewer than
    ``min_mated`` mated pairs or without impostor pairs are skipped. Rows carry
    ``attribute, subgroup, n_pos, n_neg, eer``.
    """
    p1, p2 = pos_idx
    n1, n2 = neg_idx
    pa1, pa2 = attr_vals[p1], attr_vals[p2]
    na1, na2 = attr_vals[n1], attr_vals[n2]
    pos_def = _same_group(pa1, pa2)
    neg_def = _same_group(na1, na2)

    rows: list[dict] = []
    if pos_def.any() and neg_def.any():
        rows.append(
            {
                "attribute": attr_name,
                "subgroup": OVERALL,
                "n_pos": int(pos_def.sum()),
                "n_neg": int(neg_def.sum()),
                "eer": eer(pos_scores[pos_def], neg_scores[neg_def]),
            }
        )
    groups = sorted({v for v in pa1[pos_def].tolist() if v is not None})
    for g in groups:
        pm = pos_def & (pa1 == g)
        nm = neg_def & (na1 == g)
        n_pos = int(pm.sum())
        n_neg = int(nm.sum())
        if n_pos < min_mated or n_neg == 0:
            continue
        rows.append(
            {
                "attribute": attr_name,
                "subgroup": str(g),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "eer": eer(pos_scores[pm], neg_scores[nm]),
            }
        )
    return rows


def disparity_row(
    subgroup_eers: list[dict], min_neg: int = MIN_NEG_DISPARITY
) -> dict | None:
    """Max-min / std of subgroup EERs (``__overall__`` excluded; ``n_neg >= min_neg``).

    Returns ``eer_min, eer_max, eer_std, n_subgroups, n_subgroups_dropped,
    min_neg_required``, or ``None`` when no subgroup qualifies.
    """
    sub = [r for r in subgroup_eers if r["subgroup"] != OVERALL]
    kept = [r for r in sub if r["n_neg"] >= min_neg]
    eers = np.array([r["eer"] for r in kept], dtype=float)
    eers = eers[~np.isnan(eers)]
    if not eers.size:
        return None
    return {
        "eer_min": float(eers.min()),
        "eer_max": float(eers.max()),
        "eer_std": float(eers.std()),
        "n_subgroups": int(eers.size),
        "n_subgroups_dropped": int(len(sub) - len(kept)),
        "min_neg_required": min_neg,
    }


def _core_source(meta: dict, res: set, budgets: set) -> bool:
    """Core-grid source (empty suffix) that passes the res / budget filters."""
    if meta["suffix"]:
        return False
    if res and meta["res"] not in res:
        return False
    if meta["kind"] == "compressed" and budgets and meta["budget"] not in budgets:
        return False
    return True


def _attr_len(attrs: dict[str, np.ndarray]) -> int:
    """Row count of an attribute mapping (length of its first array)."""
    return len(next(iter(attrs.values())))


def fairness_rows(
    dataset: str,
    models: Iterable[str] = ANCHOR_MODELS,
    *,
    res: Iterable[int] = (),
    budgets: Iterable[int] = (),
    sample_nonmated: int = NONMATED_SAMPLE["fairness"],
    min_mated: int = MIN_MATED,
    seed: int = 0,
    device=None,
    cpu_fallback: bool = True,
    attributes: dict[str, np.ndarray] | None = None,
    pairs=None,
) -> tuple[list[dict], list[dict]]:
    """Subgroup EER rows and disparity rows of every core-grid source.

    Aligned sources are always kept (their budget is 0); ``budgets`` filters the
    compressed ones. Returns ``(fairness_rows, disparity_rows)``; select
    :data:`FAIRNESS_COLUMNS` / :data:`DISPARITY_COLUMNS` for the published CSVs.
    Raises ``ValueError`` if an embedding array does not have one row per
    attribute entry. ``cpu_fallback`` is passed to
    :func:`~face1kb.eval.verification.make_scorer` (the published tables used
    ``device="cuda"``; ``cpu_fallback=False`` guarantees CUDA-scorer numbers).
    """
    if pairs is None:
        pairs = pair_indices(dataset, sample_nonmated, seed)
    (p1, p2), (n1, n2) = pairs
    log.info("[%s] mated=%d sampled-nonmated=%d", dataset, p1.size, n1.size)
    score = make_scorer(device, cpu_fallback=cpu_fallback)
    res_f, bud_f = set(res), set(budgets)
    attrs = attributes
    fair: list[dict] = []
    disp: list[dict] = []
    for model in models:
        for path, meta in list_sources(dataset, model):
            if not _core_source(meta, res_f, bud_f):
                continue
            emb = np.load(path)
            if emb.shape[1] != EMB_DIM:
                continue
            if attrs is None:
                attrs = load_attributes(dataset, emb.shape[0])
            check_rows(emb, _attr_len(attrs), f"{model}/{path.stem}")
            pos_s = score(emb, p1, p2)
            neg_s = score(emb, n1, n2)
            base = {
                "dataset": dataset,
                "model": model,
                "res": meta["res"],
                "budget": meta["budget"],
                "codec": meta["codec"],
            }
            for attr_name, vals in attrs.items():
                grp = subgroup_rows(
                    attr_name, vals, (p1, p2), (n1, n2), pos_s, neg_s, min_mated
                )
                fair += [{**base, **r} for r in grp]
                d = disparity_row(grp)
                if d is not None:
                    disp.append({**base, "attribute": attr_name, **d})
            log.info("  %-18s %-24s done", model, path.stem)
    return fair, disp


# ----------------------------------------------------------- differential FMR
def fmr_rows(
    attr_name: str,
    attr_vals: np.ndarray,
    neg_idx,
    neg_scores: np.ndarray,
    target_fmr: float,
    min_neg: int = MIN_NEG_FMR,
    nan_policy: str = "propagate",
) -> list[dict]:
    """Per-subgroup FMR at a global threshold, plus the ``__overall__`` spread row.

    The overall row (only with two or more qualifying subgroups) carries the
    pooled FMR, ``fmr_min``, ``fmr_max``, ``fmr_disparity_pp`` and
    ``n_subgroups``.

    With ``nan_policy="propagate"`` (the published tables) a NaN score in the
    impostor pool -- a pair touching a crop without an embedding -- makes ``tau``
    NaN and every FMR 0. ``"omit"`` drops the NaN scores first.
    """
    if nan_policy not in ("propagate", "omit"):
        raise ValueError(
            f"nan_policy must be 'propagate' or 'omit', not {nan_policy!r}"
        )
    n1, n2 = neg_idx
    na1, na2 = attr_vals[n1], attr_vals[n2]
    neg_def = _same_group(na1, na2)
    if nan_policy == "omit":
        neg_def &= ~np.isnan(neg_scores)
    if neg_def.sum() < min_neg:
        return []
    pool = neg_scores[neg_def]
    tau = float(linear_quantile(pool, 1.0 - target_fmr))
    rows = []
    groups = sorted({v for v in na1[neg_def].tolist() if v is not None})
    fmrs = {}
    for g in groups:
        nm = neg_def & (na1 == g)
        if int(nm.sum()) < min_neg:
            continue
        fmr_g = float(np.mean(neg_scores[nm] >= tau))
        fmrs[g] = fmr_g
        rows.append(
            {
                "attribute": attr_name,
                "subgroup": str(g),
                "n_neg": int(nm.sum()),
                "tau": tau,
                "target_fmr": target_fmr,
                "fmr": fmr_g,
            }
        )
    if len(fmrs) >= 2:
        vals = np.array(list(fmrs.values()))
        rows.append(
            {
                "attribute": attr_name,
                "subgroup": OVERALL,
                "n_neg": int(neg_def.sum()),
                "tau": tau,
                "target_fmr": target_fmr,
                "fmr": float(np.mean(pool >= tau)),
                "fmr_min": float(vals.min()),
                "fmr_max": float(vals.max()),
                "fmr_disparity_pp": float((vals.max() - vals.min()) * 100),
                "n_subgroups": int(vals.size),
            }
        )
    return rows


def fmr_fairness_rows(
    dataset: str,
    models: Iterable[str] = ANCHOR_MODELS,
    *,
    res: Iterable[int] = (112,),
    budgets: Iterable[int] = (512, 1024),
    target_fmrs: Sequence[float] = FMR_FAIRNESS_TARGETS,
    sample_nonmated: int = NONMATED_SAMPLE["fmr_fairness"],
    min_neg: int = MIN_NEG_FMR,
    seed: int = 0,
    device=None,
    cpu_fallback: bool = True,
    attributes: dict[str, np.ndarray] | None = None,
    pairs=None,
    nan_policy: str = "propagate",
) -> list[dict]:
    """Differential-FMR rows (:func:`fmr_rows`) of every core-grid source.

    ``nan_policy`` is passed to :func:`fmr_rows`; the default reproduces the
    published tables. Raises ``ValueError`` if an embedding array does not have
    one row per attribute entry. ``cpu_fallback`` is passed to
    :func:`~face1kb.eval.verification.make_scorer`.
    """
    if pairs is None:
        pairs = pair_indices(dataset, sample_nonmated, seed)
    _, (n1, n2) = pairs
    score = make_scorer(device, cpu_fallback=cpu_fallback)
    res_f, bud_f = set(res), set(budgets)
    attrs = attributes
    out: list[dict] = []
    for model in models:
        for path, meta in list_sources(dataset, model):
            if not _core_source(meta, res_f, bud_f):
                continue
            emb = np.load(path)
            if emb.shape[1] != EMB_DIM:
                continue
            if attrs is None:
                attrs = load_attributes(dataset, emb.shape[0])
            check_rows(emb, _attr_len(attrs), f"{model}/{path.stem}")
            neg_s = score(emb, n1, n2)
            base = {
                "dataset": dataset,
                "model": model,
                "res": meta["res"],
                "budget": meta["budget"],
                "codec": meta["codec"],
            }
            for fmr_t in target_fmrs:
                for attr_name, vals in attrs.items():
                    for r in fmr_rows(
                        attr_name, vals, (n1, n2), neg_s, fmr_t, min_neg, nan_policy
                    ):
                        out.append({**base, **r})
            log.info("  %-18s %-26s done", model, path.stem)
    return out


# ------------------------------------------------------------ disparity CIs
def uniform_disparity(
    pos_s: np.ndarray,
    neg_s: np.ndarray,
    pos_gmask: Sequence[np.ndarray],
    neg_gmask: Sequence[np.ndarray],
    keep_pos: np.ndarray,
    keep_neg: np.ndarray,
    min_mated: int = DISPARITY_CI_MIN_MATED,
) -> float:
    """Max-min subgroup EER in percentage points over the kept pairs.

    ``pos_gmask`` / ``neg_gmask`` hold one boolean mask per subgroup of the basis.
    NaN when any subgroup has fewer than ``min_mated`` mated or no impostor pairs.
    """
    eers = []
    for pg, ng in zip(pos_gmask, neg_gmask):
        pm = keep_pos & pg
        nm = keep_neg & ng
        if pm.sum() < min_mated or nm.sum() == 0:
            return np.nan
        eers.append(eer(pos_s[pm], neg_s[nm]))
    eers = np.array(eers)
    if np.isnan(eers).any():
        return np.nan
    return float((eers.max() - eers.min()) * 100)


def cluster_bootstrap_masks(n_clusters: int, reps: int, rng) -> list[np.ndarray]:
    """``reps`` boolean masks of the clusters drawn at least once (with replacement)."""
    masks = []
    for _ in range(reps):
        drawn = rng.choice(n_clusters, size=n_clusters, replace=True)
        m = np.zeros(n_clusters, bool)
        m[drawn] = True
        masks.append(m)
    return masks


def disparity_ci_rows(  # noqa: C901 - anchors x codecs with the ratio bookkeeping
    dataset: str = "kk",
    models: Iterable[str] = ANCHOR_MODELS,
    *,
    res: int = 112,
    budget: int = 1024,
    codecs: Sequence[str] = DISPARITY_CI_CODECS,
    subgroups: Sequence[str] = MONK_UNIFORM_BASIS,
    attribute: str = "skin_tone",
    reps: int = DISPARITY_CI_REPS,
    sample_nonmated: int = NONMATED_SAMPLE["disparity_ci"],
    seed: int = 0,
    device=None,
    cpu_fallback: bool = True,
    attributes: dict[str, np.ndarray] | None = None,
    pairs=None,
) -> list[dict]:
    """Cluster-bootstrap CIs of the subgroup disparity and the amplification ratio.

    ``codecs`` must list ``"aligned"`` first (the ratio denominator). Returns rows
    with :data:`DISPARITY_CI_COLUMNS` (the aligned rows carry no ratio). Raises
    ``ValueError`` if the attributes or an embedding array do not have one row per
    index entry. ``cpu_fallback`` is passed to
    :func:`~face1kb.eval.verification.make_scorer`.
    """
    if pairs is None:
        pairs = pair_indices(dataset, sample_nonmated, seed)
    (p1, p2), (n1, n2) = pairs
    score = make_scorer(device, cpu_fallback=cpu_fallback)
    index = config.read_index(dataset)
    ident = index["subject"].to_numpy()
    attrs = (
        attributes if attributes is not None else load_attributes(dataset, len(index))
    )
    if _attr_len(attrs) != len(index):
        raise ValueError(
            f"attribute length {_attr_len(attrs)} != index length {len(index)}"
        )
    tone = attrs[attribute]
    pa = tone[p1]
    na = tone[n1]
    pos_gmask = [(pa == g) for g in subgroups]
    neg_gmask = [(na == g) for g in subgroups]
    clusters, ident_ix = np.unique(ident, return_inverse=True)
    ps1, ps2 = ident_ix[p1], ident_ix[p2]
    ns1, ns2 = ident_ix[n1], ident_ix[n2]
    log.info(
        "[%s] %d identities, mated=%d nonmated=%d",
        dataset,
        clusters.size,
        p1.size,
        n1.size,
    )
    rng = np.random.default_rng(seed)
    boot_masks = cluster_bootstrap_masks(clusters.size, reps, rng)
    all_pos = np.ones_like(pa, bool)
    all_neg = np.ones_like(na, bool)

    rows = []
    aligned_boot: dict[str, np.ndarray] = {}
    for model in models:
        for codec in codecs:
            tag = f"aligned_{res}" if codec == "aligned" else f"{codec}_{res}_{budget}"
            npy = config.embeddings_path(dataset, model, tag)
            if not npy.exists():
                continue
            emb = np.load(npy)
            if emb.shape[1] != EMB_DIM:
                continue
            check_rows(emb, len(index), f"{model}/{tag}")
            pos_s = score(emb, p1, p2)
            neg_s = score(emb, n1, n2)
            full = uniform_disparity(
                pos_s, neg_s, pos_gmask, neg_gmask, all_pos, all_neg
            )
            bvals = []
            for m in boot_masks:
                kp = m[ps1] & m[ps2]
                kn = m[ns1] & m[ns2]
                bvals.append(
                    uniform_disparity(pos_s, neg_s, pos_gmask, neg_gmask, kp, kn)
                )
            bvals = np.array([v for v in bvals if not np.isnan(v)])
            lo, hi = (
                np.percentile(bvals, [2.5, 97.5]) if bvals.size else (np.nan, np.nan)
            )
            row = {
                "anchor": model,
                "codec": codec,
                "disparity_pp": full,
                "disp_lo": lo,
                "disp_hi": hi,
                "n_boot": bvals.size,
            }
            if codec == "aligned":
                aligned_boot[model] = bvals
            else:
                ab = aligned_boot.get(model)
                if ab is not None and ab.size and bvals.size:
                    k = min(ab.size, bvals.size)
                    ratio = bvals[:k] / np.clip(ab[:k], 1e-6, None)
                    base = np.median(ab)
                    row["ratio"] = full / base if base > 0 else np.nan
                    row["ratio_lo"], row["ratio_hi"] = np.percentile(ratio, [2.5, 97.5])
            rows.append(row)
            log.info("  %-20s %-14s d=%.2f [%.2f, %.2f]", model, codec, full, lo, hi)
    return rows


# ------------------------------------------------------- table-level helpers
def subgroup_spread_pp(
    fairness,
    *,
    model: str,
    codec: str,
    budget: int,
    attribute: str = "skin_tone",
    subgroups: Sequence[str] = MONK_UNIFORM_BASIS,
    res: int = 112,
) -> float:
    """Max-min EER (pp) over ``subgroups`` from a ``fairness_<dataset>.csv`` frame.

    The aligned source is looked up with budget 0. NaN with fewer than two
    subgroup rows.
    """
    b = 0 if codec == "aligned" else budget
    d = fairness[
        (fairness.res == res)
        & (fairness.attribute == attribute)
        & (fairness.subgroup.isin(list(subgroups)))
        & (fairness.codec == codec)
        & (fairness.model == model)
        & (fairness.budget == b)
    ]
    if len(d) < 2:
        return np.nan
    return float((d.eer.max() - d.eer.min()) * 100)


def disparity_pp(
    disparity, *, model: str, codec: str, budget: int, attribute: str, res: int = 112
) -> tuple[float, int]:
    """``(eer_max - eer_min) * 100`` and ``n_subgroups`` from a disparity frame.

    Returns ``(nan, 0)`` when the cell is missing; the aligned source is looked up
    with budget 0.
    """
    b = 0 if codec == "aligned" else budget
    r = disparity[
        (disparity.res == res)
        & (disparity.attribute == attribute)
        & (disparity.codec == codec)
        & (disparity.model == model)
        & (disparity.budget == b)
    ]
    if not len(r):
        return np.nan, 0
    return float((r.eer_max.iloc[0] - r.eer_min.iloc[0]) * 100), int(
        r.n_subgroups.iloc[0]
    )
