# SPDX-License-Identifier: MIT
r"""Subgroup verification accuracy: per-attribute EER and the max-min disparity.

For every core-grid embedding source (``aligned_<res>`` and ``<codec>_<res>_<budget>``,
no crop-variant suffix) of the requested FR models this scores all mated pairs plus
one seeded sample of non-mated pairs (5,000,000 by default, seed 0), restricts the
trials to each subgroup of an attribute (a pair belongs to subgroup ``g`` when both
of its images carry ``g``) and reports the within-subgroup EER. The disparity of an
attribute is ``eer_max - eer_min`` (and the std) over the subgroups with at least
1,000 impostor pairs. Color FERET attributes are pose class, ethnicity (the NIST
``race`` field, column ``skin_tone``), gender and age band from ``labels.csv``;
AI-Solutions-KK attributes are Monk skin tone, gender and age band from
``attributes.csv``, propagated per identity. The computation is
:func:`face1kb.eval.fairness.fairness_rows`.

Outputs (under ``--out-dir``, default ``$FACE1KB_OUTPUT_ROOT/fairness``)::

    fairness_<dataset>.csv            dataset, model, res, budget, codec, attribute,
                                      subgroup, n_pos, n_neg, eer
                                      (subgroup "__overall__" = all attributed pairs)
    fairness_disparity_<dataset>.csv  dataset, model, res, budget, codec, attribute,
                                      eer_min, eer_max, eer_std, n_subgroups

The published tables were scored with the CUDA scorer (``--device cuda``, the
default); ``--device cpu`` uses the NumPy scorer, which can move an EER by one
impostor trial.

Examples
--------
    python experiments/fairness/subgroup.py
    python experiments/fairness/subgroup.py --datasets colorferet --models edgeface_xs \
        --res 112 --budgets 1024
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

log = logging.getLogger("fairness.subgroup")


def _device(name: str):
    """``None`` (NumPy scorer) for ``cpu``, else the torch device string."""
    return None if name == "cpu" else name


def _frame(rows: list[dict], columns):
    """Rows as a DataFrame with exactly ``columns`` (also for an empty row list)."""
    import pandas as pd  # noqa: PLC0415

    return pd.DataFrame(rows).reindex(columns=list(columns))


def main(argv: list[str] | None = None) -> int:
    """Score subgroup EERs for every (dataset, model, source) and write the CSVs."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default="colorferet,kk")
    ap.add_argument(
        "--models", default="anchor", help="'anchor' (default), 'all' or a comma list"
    )
    ap.add_argument(
        "--res", default="", help="comma list of resolutions (default: all present)"
    )
    ap.add_argument(
        "--budgets",
        default="",
        help="comma list of budgets (default: all; aligned sources are always kept)",
    )
    ap.add_argument("--sample-nonmated", type=int, default=NONMATED_SAMPLE["fairness"])
    ap.add_argument("--min-mated", type=int, default=fairness.MIN_MATED)
    ap.add_argument("--seed", type=int, default=0)
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

    out_dir = args.out_dir or config.output_dir("fairness")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for ds in parse_list(args.datasets):
        models = resolve_models(args.models, ds)
        if not models:
            log.warning("[%s] no embeddings for the requested models: skipped", ds)
            continue
        fair, disp = fairness.fairness_rows(
            ds,
            models,
            res=parse_list(args.res, int),
            budgets=parse_list(args.budgets, int),
            sample_nonmated=args.sample_nonmated,
            min_mated=args.min_mated,
            seed=args.seed,
            device=_device(args.device),
            cpu_fallback=args.allow_cpu_fallback,
        )
        if not fair:
            log.warning("[%s] no sources matched the filters: skipped", ds)
            continue
        f_csv = out_dir / f"fairness_{ds}.csv"
        d_csv = out_dir / f"fairness_disparity_{ds}.csv"
        _frame(fair, fairness.FAIRNESS_COLUMNS).to_csv(f_csv, index=False)
        _frame(disp, fairness.DISPARITY_COLUMNS).to_csv(d_csv, index=False)
        log.info(
            "[%s] wrote %s (%d rows) and %s (%d rows)",
            ds,
            f_csv,
            len(fair),
            d_csv,
            len(disp),
        )
        written += 1
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
