# SPDX-License-Identifier: MIT
r"""ISO/IEC 29794-5 annex study, stage 2: score every swept cell.

For every embedding array ``<cell_id>__<model>.npy`` written by ``sweep.py`` whose
cell has a statistics row, this reports

* ``self_sim``: the annexes' objective, the mean cosine between each compressed crop
  and the same crop uncompressed at 112 px (the ``none_r112_color_none_def_b1024``
  baseline cell of the same matcher), and
* verification ``eer`` and ``fnmr`` at FMR 1e-3 and 1e-4: all mated pairs of the
  subset plus 2,000,000 non-mated pairs drawn with ``numpy.random.default_rng(0)``
  (pairs of the same subject are dropped), for two protocols (``sym``: both sides
  compressed; ``asym``: compressed against the uncompressed baseline) and two
  populations (``all`` and ``frontal``).

The EER is taken at the mated score where FNMR and FMR are closest (FNMR at the
sorted mated scores, FMR by a sorted search in the non-mated scores), and FNMR at a
target FMR uses the ``1 - FMR`` quantile of the non-mated scores. Pairs touching a
NaN row (crops a cell did not run) are dropped; ``--pop-limit N`` also drops every
crop after the first ``N`` (like-for-like scoring of the 2000-crop grid).

When a cell was recorded by several runs, the statistics row of the most recent
statistics file (by modification time) is used. Output: one row per (cell, matcher)
in ``OUTPUT_ROOT/annex/<out>`` (default ``scores.csv``). ``--device cuda`` computes the
cosines on the GPU (the paper's scores were computed this way); on the CPU the
results can differ in the last float32 digits.

Examples
--------
    python experiments/annex/score.py --datasets colorferet --pop-limit 2000 \
        --out scores_pop2000.csv --device cuda
    python experiments/annex/score.py --datasets colorferet,kk --device cuda
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from face1kb import config
from face1kb.data.cli_utils import parse_list, setup_logging

log = logging.getLogger("annex.score")

#: The uncompressed 112 px reference cell.
BASELINE = "none_r112_color_none_def_b1024"
N_NONMATED = 2_000_000
FMRS = (1e-3, 1e-4)


def annex_dir(dataset: str) -> Path:
    """Annex work folder of ``dataset``: ``WORK_ROOT/<dataset>/annex``."""
    return config.work_dir(dataset) / "annex"


def cache_dir(dataset: str, cache_root: str | None = None) -> Path:
    """Crop cache of ``prep.py`` (``annex/cache`` or ``<cache-root>/<dataset>``)."""
    return Path(cache_root) / dataset if cache_root else annex_dir(dataset) / "cache"


def read_cells(dataset: str) -> pd.DataFrame:
    """Statistics rows of all sweep runs; the most recent file wins per cell."""
    files = sorted(
        annex_dir(dataset).glob("cells*.parquet"),
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    if not files:
        raise SystemExit(f"no cells*.parquet under {annex_dir(dataset)}")
    return pd.concat([pd.read_parquet(p) for p in files]).drop_duplicates(
        "cell_id", keep="last"
    )


def make_pairs(meta: pd.DataFrame, mask: np.ndarray, seed: int = 0):
    """All mated pairs plus a seeded non-mated sample, restricted to ``mask``."""
    idx = np.flatnonzero(mask)
    sid = meta.sid.to_numpy()[idx]
    order = np.argsort(sid, kind="stable")
    idx, sid = idx[order], sid[order]
    pos = [
        (a, b)
        for _, g in pd.Series(idx).groupby(sid)
        for a, b in itertools.combinations(g.to_numpy(), 2)
    ]
    pos = np.array(pos, np.int64).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    a, b = rng.choice(idx, N_NONMATED), rng.choice(idx, N_NONMATED)
    keep = meta.sid.to_numpy()[a] != meta.sid.to_numpy()[b]
    return pos, np.stack([a[keep], b[keep]], 1)


class Scorer:
    """Row-normalised embeddings and indexed pair cosines, on NumPy or torch."""

    def __init__(self, device: str | None = None):
        self.torch = None
        self.device = device
        if device and device != "cpu":
            import torch  # noqa: PLC0415

            self.torch = torch

    def unit(self, e: np.ndarray):
        e = e.astype(np.float32)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        if self.torch is not None:
            return self.torch.as_tensor(e, device=self.device)
        return e

    def finite(self, x) -> np.ndarray:
        arr = x.cpu().numpy() if self.torch is not None else x
        return np.isfinite(arr).all(1)

    def cos(self, x, y, ij: np.ndarray, chunk: int = 500_000) -> np.ndarray:
        """Cosines of the index pairs ``ij`` (rows of ``x`` against rows of ``y``)."""
        if self.torch is not None:
            torch = self.torch
            out = [
                (x[i[:, 0]] * y[i[:, 1]]).sum(1)
                for i in (
                    torch.as_tensor(ij[k : k + chunk], device=x.device)
                    for k in range(0, len(ij), chunk)
                )
            ]
            if not out:
                return np.zeros(0, np.float32)
            return torch.cat(out).float().cpu().numpy()
        return np.concatenate(
            [
                np.einsum("ij,ij->i", x[ij[k : k + chunk, 0]], y[ij[k : k + chunk, 1]])
                for k in range(0, len(ij), chunk)
            ]
        )


def metrics(pos: np.ndarray, neg: np.ndarray) -> dict:
    """EER and FNMR at the FMR operating points from raw score arrays."""
    pos, neg = np.sort(pos), np.sort(neg)
    fnmr = np.arange(len(pos)) / len(pos)
    fmr = 1.0 - np.searchsorted(neg, pos, "left") / len(neg)
    i = int(np.argmin(np.abs(fnmr - fmr)))
    out = {"eer": float((fnmr[i] + fmr[i]) / 2)}
    for f in FMRS:
        t = np.quantile(neg, 1.0 - f)
        out[f"fnmr_{f:g}"] = float((pos < t).mean())
    return out


def score(
    ds: str,
    scorer: Scorer,
    arms: list[str],
    pop_limit: int = 0,
    cache_root: str | None = None,
) -> pd.DataFrame:
    """Score every embedding array present for ``ds``."""
    meta = pd.read_parquet(cache_dir(ds, cache_root) / "meta.parquet")
    emb_dir = annex_dir(ds) / "emb"
    cells = read_cells(ds)
    if arms:
        cells = cells[cells.arm.isin(arms)]
    pops = {
        "all": np.ones(len(meta), bool),
        "frontal": meta.frontal.to_numpy().astype(bool),
    }
    pairs = {k: make_pairs(meta, m) for k, m in pops.items()}
    base = {
        p.name.split("__")[1][:-4]: scorer.unit(np.load(p))
        for p in emb_dir.glob(f"{BASELINE}__*.npy")
    }
    rows = []
    keep = set(cells.cell_id)
    files = [p for p in sorted(emb_dir.glob("*.npy")) if p.name.split("__")[0] in keep]
    for p in tqdm(files, desc=f"score {ds}"):
        cid, model = p.name[:-4].split("__")
        cell = cells[cells.cell_id == cid]
        if cell.empty or model not in base:
            continue
        e = scorer.unit(np.load(p))
        b = base[model]
        ok = scorer.finite(e) & scorer.finite(b)
        if pop_limit:
            ok[pop_limit:] = False
        diag = np.stack([np.flatnonzero(ok)] * 2, 1)
        row = {
            "dataset": ds,
            "model": model,
            **cell.iloc[0].to_dict(),
            "self_sim": float(scorer.cos(e, b, diag).mean()),
        }
        for pop, (pos, neg) in pairs.items():
            ps, ns = pos[ok[pos].all(1)], neg[ok[neg].all(1)]
            for proto, y in (("sym", e), ("asym", b)):
                m = metrics(scorer.cos(e, y, ps), scorer.cos(e, y, ns))
                row |= {f"{pop}_{proto}_{k}": v for k, v in m.items()}
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """Score one or both datasets."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument("--arms", default="", help="only cells of these arms")
    ap.add_argument("--out", default="scores.csv", help="file name or path")
    ap.add_argument("--out-dir", default=None, help="default OUTPUT_ROOT/annex")
    ap.add_argument("--device", default=None, help="cuda[:N] (default: CPU)")
    ap.add_argument("--pop-limit", type=int, default=0)
    ap.add_argument("--cache-root", default=None, help="prep.py --cache-root")
    args = ap.parse_args(argv)
    setup_logging()
    scorer = Scorer(args.device)
    df = pd.concat(
        [
            score(ds, scorer, parse_list(args.arms), args.pop_limit, args.cache_root)
            for ds in parse_list(args.datasets)
        ],
        ignore_index=True,
    )
    out_dir = Path(args.out_dir) if args.out_dir else config.output_dir("annex")
    out = out_dir / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("%d rows -> %s", len(df), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
