# SPDX-License-Identifier: MIT
r"""Differential false-match rate across demographic subgroups.

For each (dataset, model, source, attribute) a global threshold ``tau`` is set at a
target overall FMR (1e-2 and 1e-3) as the ``1 - target`` quantile of all
within-subgroup impostor scores of the attribute; each subgroup's FMR is the share of
its impostor scores at or above ``tau``, and the spread is ``max - min`` in
percentage points over the subgroups with at least 50 impostor pairs. The impostor
set is a seeded sample of 300,000 non-mated pairs (seed 0); the default grid is 112 px
at 512 and 1024 B plus the aligned source. The computation is
:func:`face1kb.eval.fairness.fmr_fairness_rows`.

A pair touching a crop without an embedding (NaN score) makes ``tau`` NaN and every
FMR 0 with the default ``--nan-policy propagate``, as in the published tables;
``--nan-policy omit`` drops such pairs first.

Output (under ``--out-dir``, default ``$FACE1KB_OUTPUT_ROOT/fairness``)::

    fmr_fairness_<dataset>.csv   dataset, model, res, budget, codec, attribute,
                                 subgroup, n_neg, tau, target_fmr, fmr, fmr_min,
                                 fmr_max, fmr_disparity_pp, n_subgroups
                                 (the last four only on the "__overall__" rows)

Examples
--------
    python experiments/fairness/fmr_fairness.py
    python experiments/fairness/fmr_fairness.py --datasets colorferet --models lvface_l
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from face1kb import config
from face1kb.data.cli_utils import parse_list, setup_logging
from face1kb.eval import fairness
from face1kb.eval.embeddings import resolve_models
from face1kb.eval.verification import NONMATED_SAMPLE

log = logging.getLogger("fairness.fmr")


def main(argv: list[str] | None = None) -> int:
    """Write the differential-FMR rows of every requested dataset."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default="colorferet,kk")
    ap.add_argument("--models", default="anchor", help="'anchor', 'all' or a list")
    ap.add_argument("--res", default="112")
    ap.add_argument("--budgets", default="512,1024")
    ap.add_argument(
        "--target-fmr",
        default=",".join(str(t) for t in fairness.FMR_FAIRNESS_TARGETS),
    )
    ap.add_argument(
        "--sample-nonmated", type=int, default=NONMATED_SAMPLE["fmr_fairness"]
    )
    ap.add_argument("--min-neg", type=int, default=fairness.MIN_NEG_FMR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nan-policy", choices=("propagate", "omit"), default="propagate")
    ap.add_argument(
        "--device", default="cuda", help="scoring device: cuda, cuda:N or cpu"
    )
    ap.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="score on the CPU when the GPU runs out of memory (NumPy-scorer numbers)",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    setup_logging()
    import pandas as pd  # noqa: PLC0415

    out_dir = args.out_dir or config.output_dir("fairness")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for ds in parse_list(args.datasets):
        models = resolve_models(args.models, ds)
        if not models:
            log.warning("[%s] no embeddings for the requested models: skipped", ds)
            continue
        rows = fairness.fmr_fairness_rows(
            ds,
            models,
            res=parse_list(args.res, int),
            budgets=parse_list(args.budgets, int),
            target_fmrs=parse_list(args.target_fmr, float),
            sample_nonmated=args.sample_nonmated,
            min_neg=args.min_neg,
            seed=args.seed,
            device=None if args.device == "cpu" else args.device,
            cpu_fallback=args.allow_cpu_fallback,
            nan_policy=args.nan_policy,
        )
        if not rows:
            log.warning("[%s] no rows", ds)
            continue
        out = out_dir / f"fmr_fairness_{ds}.csv"
        df = pd.DataFrame(rows).reindex(columns=list(fairness.FMR_FAIRNESS_COLUMNS))
        df.to_csv(out, index=False)
        log.info("[%s] wrote %s (%d rows)", ds, out, len(df))
        written += 1
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
