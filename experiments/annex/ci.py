# SPDX-License-Identifier: MIT
r"""ISO/IEC 29794-5 annex study: subject-level bootstrap of the head-to-head arms.

For each dataset and classical codec, the arms are our configuration (``ours``: the
recommended resolution, colour, no manipulation, default flags), the Annex E and
Annex F configurations and the confirmed sweep winner (``winner``). The score of a
pair is the mean cosine over the four anchor matchers (score-level fusion), on all
mated pairs of the evaluation subset plus 400,000 seeded non-mated pairs. For each
arm the cell with the lowest fused EER is reported, with

* a 95 % percentile interval of its EER over 200 resamples of subjects with
  replacement (a pair is kept when its first crop's subject is drawn), and
* the paired EER difference to the ``ours`` arm on the same resamples (mean and 95 %
  interval, ``d``, ``d_lo``, ``d_hi``).

Cells whose arrays contain NaN for some matcher (run on a subsample) are skipped. The
random draws (``numpy.random.default_rng(0)``: the non-mated pairs, then the 200
resamples) depend on the row order of the subset (``prep.py``).

Inputs: ``OUTPUT_ROOT/annex/scores.csv`` (``score.py``), the embedding arrays and
``meta.parquet``. Output: ``OUTPUT_ROOT/annex/ci.csv``. A few minutes on a GPU
(``--device cuda``, the default); the CPU works but is slow.

Examples
--------
    python experiments/annex/ci.py
    python experiments/annex/ci.py --device cpu --reps 20
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from face1kb import config
from face1kb.data.cli_utils import parse_list, setup_logging
from face1kb.eval.embeddings import ANCHOR_MODELS

log = logging.getLogger("annex.ci")

ANCHORS = ANCHOR_MODELS
#: Our recommended resolution per codec (colour, no manipulation, default flags).
OURS = {
    "jpeg": 96,
    "webp": 96,
    "avif": 112,
    "heif": 96,
    "jpeg_xl": 112,
    "jpeg2000": 224,
}
N_NEG = 400_000
REPS = 200
KINDS = ("ours", "annexE", "annexF", "winner")


def annex_dir(dataset: str) -> Path:
    """Annex work folder of ``dataset``: ``WORK_ROOT/<dataset>/annex``."""
    return config.work_dir(dataset) / "annex"


def cache_dir(dataset: str, cache_root: str | None = None) -> Path:
    """Crop cache of ``prep.py`` (``annex/cache`` or ``<cache-root>/<dataset>``)."""
    return Path(cache_root) / dataset if cache_root else annex_dir(dataset) / "cache"


def eer(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """EER from raw score tensors (at the mated score closest to FNMR = FMR)."""
    pos, _ = torch.sort(pos)
    neg, _ = torch.sort(neg)
    fnmr = torch.arange(len(pos), device=pos.device, dtype=torch.float32) / len(pos)
    fmr = 1.0 - torch.searchsorted(neg, pos).float() / len(neg)
    i = int(torch.argmin((fnmr - fmr).abs()))
    return float((fnmr[i] + fmr[i]) / 2)


def make_pairs(sid: np.ndarray, rng: np.random.Generator):
    """All mated pairs plus a fixed non-mated sample, as index arrays."""
    idx = np.arange(len(sid))
    pos = np.array(
        [
            p
            for _, g in pd.Series(idx).groupby(sid)
            for p in itertools.combinations(g.to_numpy(), 2)
        ],
        np.int64,
    ).reshape(-1, 2)
    a, b = rng.choice(idx, N_NEG), rng.choice(idx, N_NEG)
    return pos, np.stack([a, b], 1)[sid[a] != sid[b]]


def arm_kind(r) -> str:
    """Arm of a scores row: ``ours`` for our configuration, else its ``arm``."""
    if (
        r["flags"] == "{}"
        and r.color == "color"
        and r.manip == "none"
        and OURS.get(r.codec) == r.res
    ):
        return "ours"
    return r.arm


def arms(scores: pd.DataFrame, ds: str) -> dict:
    """Map (codec, arm) -> the cell ids of that arm."""
    x = scores[(scores.dataset == ds) & scores.model.isin(ANCHORS)]
    out: dict = {}
    for _, r in x.iterrows():
        kind = arm_kind(r)
        if kind in KINDS:
            out.setdefault((r.codec, kind), set()).add(r.cell_id)
    return out


def run_dataset(  # noqa: C901 - one bootstrap pass: arms, resamples, paired deltas
    ds: str, dev: str, scores: pd.DataFrame, reps: int, cache_root: str | None
) -> list[dict]:
    """Bootstrap every arm of one dataset; returns the rows of ``ci.csv``."""
    meta = pd.read_parquet(cache_dir(ds, cache_root) / "meta.parquet")
    sid = pd.factorize(meta.sid)[0]
    rng = np.random.default_rng(0)
    pos, neg = make_pairs(sid, rng)
    pos_t = torch.as_tensor(pos, device=dev)
    neg_t = torch.as_tensor(neg, device=dev)
    subj = np.unique(sid)
    # subject of each pair (its first crop), for identity-level resampling
    pos_s = torch.as_tensor(sid[pos[:, 0]], device=dev)
    neg_s = torch.as_tensor(sid[neg[:, 0]], device=dev)
    draws = [
        torch.as_tensor(rng.choice(subj, len(subj)), device=dev) for _ in range(reps)
    ]
    emb_dir = annex_dir(ds) / "emb"

    def fused_scores(cell_id: str) -> torch.Tensor | None:
        s = []
        for m in ANCHORS:
            p = emb_dir / f"{cell_id}__{m}.npy"
            if not p.exists():
                return None
            e = torch.as_tensor(np.load(p).astype(np.float32), device=dev)
            e = e / (e.norm(dim=1, keepdim=True) + 1e-12)
            s.append(
                torch.cat(
                    [
                        (e[pos_t[:, 0]] * e[pos_t[:, 1]]).sum(1),
                        (e[neg_t[:, 0]] * e[neg_t[:, 1]]).sum(1),
                    ]
                )
            )
        if any(bool(torch.isnan(t).any()) for t in s):
            return None  # a matcher of this cell ran on a subsample only
        return torch.stack(s).mean(0)

    rows: list[dict] = []
    cache: dict = {}
    n_pos = len(pos)
    for (codec, kind), ids in tqdm(sorted(arms(scores, ds).items()), desc=f"ci {ds}"):
        best, bid = None, None
        for cid in sorted(ids):  # the best cell of the arm, as reported
            sc = fused_scores(cid)
            if sc is None:
                continue
            e = eer(sc[:n_pos], sc[n_pos:])
            if best is None or e < best[0]:
                best, bid = (e, sc), cid
        if best is None:
            continue
        cache[(codec, kind)] = best[1]
        boot = []
        for d in draws:
            pm, nm = torch.isin(pos_s, d), torch.isin(neg_s, d)
            boot.append(eer(best[1][:n_pos][pm], best[1][n_pos:][nm]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(
            {
                "dataset": ds,
                "codec": codec,
                "kind": kind,
                "cell_id": bid,
                "eer": best[0],
                "lo": lo,
                "hi": hi,
            }
        )
    # paired differences against ours, on the same resamples
    for (codec, kind), sc in cache.items():
        if kind == "ours" or (codec, "ours") not in cache:
            continue
        ref = cache[(codec, "ours")]
        dif = []
        for d in draws:
            pm, nm = torch.isin(pos_s, d), torch.isin(neg_s, d)
            dif.append(
                eer(sc[:n_pos][pm], sc[n_pos:][nm])
                - eer(ref[:n_pos][pm], ref[n_pos:][nm])
            )
        lo, hi = np.percentile(dif, [2.5, 97.5])
        for r in rows:
            if (r["dataset"], r["codec"], r["kind"]) == (ds, codec, kind):
                r |= {"d_lo": lo, "d_hi": hi, "d": float(np.mean(dif))}
    return rows


def main(argv: list[str] | None = None) -> int:
    """Bootstrap each arm's EER and its paired difference against ours."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--scores", default=None, help="default OUTPUT_ROOT/annex/...")
    ap.add_argument("--out", default=None, help="default OUTPUT_ROOT/annex/ci.csv")
    ap.add_argument("--cache-root", default=None, help="prep.py --cache-root")
    args = ap.parse_args(argv)
    setup_logging()
    path = (
        Path(args.scores) if args.scores else config.output_dir("annex", "scores.csv")
    )
    scores = pd.read_csv(path)
    rows = [
        r
        for ds in parse_list(args.datasets)
        for r in run_dataset(ds, args.device, scores, args.reps, args.cache_root)
    ]
    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else config.output_dir("annex", "ci.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("%d rows -> %s\n%s", len(df), out, df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
