# SPDX-License-Identifier: MIT
"""Straightforward reference implementations used as differential oracles.

They restate the metric formulas in their direct (loop) form; the tests require the
library to return identical values.
"""

from __future__ import annotations

import numpy as np

from face1kb.eval.verification import eer_grid

# The published numbers were computed under NumPy 2 (2.4.6); every NumPy 2 release
# evaluates np.linspace over float32 endpoints as a float32 grid. NumPy 1.x
# evaluates it in float64; there the oracle uses eer_grid (tested separately
# against golden values of NumPy 2).
_linspace = np.linspace if int(np.__version__.split(".")[0]) >= 2 else eer_grid


def metrics(pos, neg, fars=(1e-2, 1e-3, 1e-4)):
    pos = np.sort(pos[~np.isnan(pos)])
    neg = np.sort(neg[~np.isnan(neg)])
    out = {"n_pos": int(pos.size), "n_neg": int(neg.size)}
    if pos.size == 0 or neg.size == 0:
        out["eer"] = np.nan
        for f in fars:
            out[f"fnmr_{f:g}"] = np.nan
        return out
    thr = _linspace(min(pos[0], neg[0]), max(pos[-1], neg[-1]), 4000)
    fnmr = np.searchsorted(pos, thr, side="left") / pos.size
    fmr = (neg.size - np.searchsorted(neg, thr, side="left")) / neg.size
    j = int(np.argmin(np.abs(fnmr - fmr)))
    out["eer"] = float((fnmr[j] + fmr[j]) / 2)
    for f in fars:
        m = int(round(f * neg.size))
        if m < 10:
            out[f"fnmr_{f:g}"] = np.nan
            out[f"fmr_realised_{f:g}"] = np.nan
            continue
        t = neg[neg.size - m]
        out[f"fnmr_{f:g}"] = float(np.searchsorted(pos, t, side="left") / pos.size)
        out[f"fmr_realised_{f:g}"] = float(m / neg.size)
    return out


def eer_only(pos, neg):
    pos = np.sort(pos[~np.isnan(pos)])
    neg = np.sort(neg[~np.isnan(neg)])
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    thr = _linspace(min(pos[0], neg[0]), max(pos[-1], neg[-1]), 4000)
    fnmr = np.searchsorted(pos, thr, side="left") / pos.size
    fmr = (neg.size - np.searchsorted(neg, thr, side="left")) / neg.size
    j = int(np.argmin(np.abs(fnmr - fmr)))
    return float((fnmr[j] + fmr[j]) / 2)


def eer_threshold(pos, neg):
    sp = np.sort(pos)
    sn = np.sort(neg)
    thr = _linspace(min(sp[0], sn[0]), max(sp[-1], sn[-1]), 4000)
    fnmr = np.searchsorted(sp, thr, side="left") / sp.size
    fmr = (sn.size - np.searchsorted(sn, thr, side="left")) / sn.size
    return float(thr[int(np.argmin(np.abs(fnmr - fmr)))])


def bootstrap(
    pos, neg, fars, reps, neg_cap, seed=0, pos_subj=None, neg_subj=None, level="pair"
):
    if reps <= 0:
        return {}
    pm = ~np.isnan(pos)
    nm = ~np.isnan(neg)
    pos = pos[pm]
    neg = neg[nm]
    if pos.size == 0 or neg.size == 0:
        return {}
    rng = np.random.default_rng(seed)
    keys = ["eer"] + [f"fnmr_{f:g}" for f in fars]
    acc = {k: [] for k in keys}
    if level == "subject" and pos_subj is not None and neg_subj is not None:
        ps = np.asarray(pos_subj)[pm]
        ns = np.asarray(neg_subj)[nm]
        if neg.size > neg_cap:
            sel = rng.choice(neg.size, size=neg_cap, replace=False)
            neg, ns = neg[sel], ns[sel]
        uniq, inv = np.unique(np.concatenate([ps, ns]), return_inverse=True)
        k_subj = uniq.size
        ps_code, ns_code = inv[: ps.size], inv[ps.size :]
        for _ in range(reps):
            counts = np.bincount(rng.integers(0, k_subj, k_subj), minlength=k_subj)
            pb = np.repeat(pos, counts[ps_code])
            nb = np.repeat(neg, counts[ns_code])
            if pb.size == 0 or nb.size == 0:
                continue
            m = metrics(pb, nb, fars)
            for key in keys:
                acc[key].append(m[key])
    else:
        if neg.size > neg_cap:
            neg = rng.choice(neg, size=neg_cap, replace=False)
        for _ in range(reps):
            pb = pos[rng.integers(0, pos.size, pos.size)]
            nb = neg[rng.integers(0, neg.size, neg.size)]
            m = metrics(pb, nb, fars)
            for key in keys:
                acc[key].append(m[key])
    return {
        k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        for k, v in acc.items()
        if v
    }


def midrank_loop(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = x.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def cliffs_delta_loop(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))
