# SPDX-License-Identifier: MIT
r"""Paired codec-vs-codec significance tests per cell (Sections 14.1-14.4).

For every requested ``(dataset, model, res, budget)`` cell, all sources of the cell
(each codec plus the aligned crops of that resolution) are scored on one shared trial
set -- all mated pairs plus a seeded sample of 2,000,000 non-mated pairs (seed 0) --
restricted to the trials valid for every source, and each source decides at its own
EER threshold. Every unordered pair of sources gets a McNemar test on the per-trial
decisions and DeLong's paired AUC test; McNemar p-values are BH-FDR adjusted within
the cell (55 tests for 11 sources) and ``significant = p_adj < 0.05``
(:mod:`face1kb.eval.significance`).

Writes ``significance_<dataset><suffix>.csv`` under ``--out-dir`` (default
``config.output_dir("accuracy")``) with
:func:`face1kb.eval.significance.write_significance_csv`, which reproduces the text
of the published files; cells are ordered by model name, resolution and budget.
The published tables cover the four anchors at 112 px / 1024 B on Color FERET and at
112 and 224 px / 1024 B on AI-Solutions-KK, scored with the NumPy scorer (the
default here; ``--device cuda`` changes the AUC columns in the last digits).

``--merge OUT IN [IN ...]`` concatenates shard CSVs of separate runs (e.g. one per
model with ``--out-suffix``) into one file without recomputing anything and without
changing the shards' text.

Examples
--------
    python experiments/accuracy/compute_significance.py --datasets colorferet \
        --models anchor --res 112 --budgets 1024
    python experiments/accuracy/compute_significance.py --datasets kk \
        --models edgeface_xs --res 112,224 --budgets 1024 --out-suffix _edgeface_xs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from face1kb import config

log = logging.getLogger("compute_significance")


def _ints(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def dataset_frames(dataset: str, models: list[str], args) -> list:
    """One finished (BH-corrected) frame per cell of ``dataset``, in output order."""
    from face1kb.eval import significance as sig  # noqa: PLC0415
    from face1kb.eval.verification import pair_indices  # noqa: PLC0415

    pairs = pair_indices(dataset, args.sample_nonmated, args.seed)
    frames = []
    for model in sorted(models):
        for res in _ints(args.res) or list(config.RESOLUTIONS):
            for budget in _ints(args.budgets) or list(config.BUDGETS):
                rows = sig.significance_rows(
                    dataset,
                    [model],
                    res=(res,),
                    budgets=(budget,),
                    sample_nonmated=args.sample_nonmated,
                    seed=args.seed,
                    min_coverage=args.min_coverage,
                    device=args.device,
                    pairs=pairs,
                )
                if rows:
                    frames.append(sig.significance_frame(rows))
    return frames


def merge(out: str, inputs: list[str]) -> int:
    """Concatenate shard CSVs as written (no recomputation).

    Shards already carry the single default-parser round trip of
    :func:`face1kb.eval.significance.write_significance_csv`; they are read back with
    the exact (``round_trip``) float parser so that the merge keeps their text and
    does not round the last digits a second time.
    """
    import pandas as pd  # noqa: PLC0415

    parts = [pd.read_csv(p, float_precision="round_trip") for p in inputs]
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    log.info("wrote %s (%d rows from %d files)", out, sum(map(len, parts)), len(parts))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the paired tests and write the per-dataset CSVs."""
    from face1kb.eval import significance as sig  # noqa: PLC0415
    from face1kb.eval.embeddings import resolve_models  # noqa: PLC0415
    from face1kb.eval.verification import NONMATED_SAMPLE  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument("--models", default="anchor", help="'anchor', 'all' or a list")
    ap.add_argument("--res", default="", help="resolutions (default: all)")
    ap.add_argument("--budgets", default="", help="budgets (default: 1024,512)")
    ap.add_argument(
        "--sample-nonmated", type=int, default=NONMATED_SAMPLE["significance"]
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-coverage", type=float, default=sig.MIN_COVERAGE)
    ap.add_argument(
        "--device", default=None, help="torch device for scoring (default: NumPy)"
    )
    ap.add_argument("--out-dir", default=None, help="default: outputs/accuracy")
    ap.add_argument("--out-suffix", default="", help="appended to the file name")
    ap.add_argument(
        "--merge",
        nargs="+",
        metavar=("OUT", "IN"),
        help="concatenate shard CSVs IN ... into OUT and exit",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.merge:
        if len(args.merge) < 2:
            ap.error("--merge needs an output and at least one input")
        return merge(args.merge[0], args.merge[1:])
    out_dir = Path(args.out_dir) if args.out_dir else config.output_dir("accuracy")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        models = resolve_models(args.models, ds)
        if not models:
            log.warning("[%s] no embeddings for the requested models; skipped", ds)
            continue
        frames = dataset_frames(ds, models, args)
        if not frames:
            log.warning("[%s] no cell with two usable sources; nothing written", ds)
            continue
        out = out_dir / f"significance_{ds}{args.out_suffix}.csv"
        sig.write_significance_csv(frames, out)
        n = sum(len(f) for f in frames)
        n_sig = sum(int(f.significant.sum()) for f in frames)
        log.info("[%s] wrote %s (%d rows, %d significant)", ds, out, n, n_sig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
