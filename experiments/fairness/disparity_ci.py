# SPDX-License-Identifier: MIT
r"""Cluster-bootstrap CIs of the AI-Solutions-KK Monk skin-tone disparity.

For each FR anchor and codec at 112 px / 1024 B (the aligned reference first) this
scores all mated pairs and a seeded sample of 400,000 non-mated pairs (seed 0), and
computes the disparity ``EER_max - EER_min`` in percentage points over the uniform
Monk basis MST5/6/7/9/10. The identities are then resampled with replacement 500
times; a pair stays in a replicate when the identities of both of its images were
drawn. The 2.5/97.5 percentiles over the replicates give the CI of the disparity and
of the amplification ratio (codec disparity over the median aligned bootstrap
disparity). Mated pairs belong to a subgroup through their identity, non-mated pairs
through the skin tone of their first image. The computation is
:func:`face1kb.eval.fairness.disparity_ci_rows`.

Output (default ``$FACE1KB_OUTPUT_ROOT/fairness/disparity_ci_kk.csv``)::

    anchor, codec, disparity_pp, disp_lo, disp_hi, n_boot, ratio, ratio_lo, ratio_hi

Examples
--------
    python experiments/fairness/disparity_ci.py
    python experiments/fairness/disparity_ci.py --models arcface_antelopev2 --reps 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from face1kb import config
from face1kb.data.cli_utils import parse_list, setup_logging
from face1kb.eval import fairness
from face1kb.eval.embeddings import ANCHOR_MODELS
from face1kb.eval.verification import NONMATED_SAMPLE

log = logging.getLogger("fairness.disparity_ci")


def main(argv: list[str] | None = None) -> int:
    """Bootstrap the disparity and the amplification ratio per anchor and codec."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(ANCHOR_MODELS))
    ap.add_argument(
        "--codecs",
        default=",".join(fairness.DISPARITY_CI_CODECS),
        help="comma list; 'aligned' must come first (the ratio denominator)",
    )
    ap.add_argument("--res", type=int, default=112)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=fairness.DISPARITY_CI_REPS)
    ap.add_argument(
        "--sample-nonmated", type=int, default=NONMATED_SAMPLE["disparity_ci"]
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device", default="cuda", help="scoring device: cuda, cuda:N or cpu"
    )
    ap.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="score on the CPU when the GPU runs out of memory (NumPy-scorer numbers)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    setup_logging()
    import pandas as pd  # noqa: PLC0415

    codecs = parse_list(args.codecs)
    if not codecs or codecs[0] != "aligned":
        ap.error("--codecs must start with 'aligned'")
    rows = fairness.disparity_ci_rows(
        "kk",
        parse_list(args.models),
        res=args.res,
        budget=args.budget,
        codecs=codecs,
        reps=args.reps,
        sample_nonmated=args.sample_nonmated,
        seed=args.seed,
        device=None if args.device == "cpu" else args.device,
        cpu_fallback=args.allow_cpu_fallback,
    )
    if not rows:
        log.error("no embedding arrays found for the requested cells")
        return 1
    out = args.out or config.output_dir("fairness") / "disparity_ci_kk.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).reindex(columns=list(fairness.DISPARITY_CI_COLUMNS))
    df.to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
