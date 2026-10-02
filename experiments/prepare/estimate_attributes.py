# SPDX-License-Identifier: MIT
"""Estimate age, gender and Monk Skin Tone of the AI-Solutions-KK crops.

Age and gender: insightface ``buffalo_l`` genderage on the aligned 112 px crops
(first 8 per identity), on the CPU by default, which reproduces the paper's
genderage outputs bit for bit given the same crops (the paper's estimates were made
on another set of 112 px crops than the released ones, see docs/datasets.md, Known
deviations; ``--ctx-id 0`` runs it on the first visible CUDA device, which
can move an age on a rounding boundary by one year). Monk Skin Tone: ``stone``
(skin-tone-classifier, GPL-3.0, optional; ``pip install skin-tone-classifier``) on
the aligned 224 px crop of the first image per identity. Writes ``attributes.csv``
(one row per processed crop). The file is per-image data derived from the dataset:
keep it local. Nothing is written if a crop cannot be read.

``--dir-112`` / ``--dir-224`` point the estimator at other crop folders than
``aligned_112`` / ``aligned_224`` of the dataset (for example another release of the
crops); the index still defines which images are used.

Cost: 840 genderage calls (seconds on the CPU) and 105 ``stone`` calls (about a
minute on the CPU).

Examples
--------
    python experiments/prepare/estimate_attributes.py --dataset kk
    python experiments/prepare/estimate_attributes.py --dataset kk --no-mst
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from face1kb import config
from face1kb.data import read_index_csv
from face1kb.data.attributes import (
    GA_PER_SUBJECT,
    MST_PER_SUBJECT,
    estimate_attributes,
    write_attributes,
)
from face1kb.data.cli_utils import output_path, parse_list, setup_logging
from face1kb.data.crops import MissingCropsError

log = logging.getLogger("estimate_attributes")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="kk", choices=config.DATASETS)
    ap.add_argument("--ga-per", type=int, default=GA_PER_SUBJECT)
    ap.add_argument("--mst-per", type=int, default=MST_PER_SUBJECT)
    ap.add_argument("--no-mst", action="store_true", help="skip the stone MST step")
    ap.add_argument("--subjects", default="", help="comma-separated subset")
    ap.add_argument(
        "--ctx-id",
        type=int,
        default=-1,
        help="-1: CPU (default; bit-exact genderage outputs for the same crops); "
        "N >= 0: CUDA device N among the visible ones (CUDA_VISIBLE_DEVICES)",
    )
    ap.add_argument("--dir-112", default=None, help="112 px crops (age, gender)")
    ap.add_argument("--dir-224", default=None, help="224 px crops (MST)")
    ap.add_argument("--out", default=None, help="output CSV (default: attributes)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()

    out = output_path(args.out, config.attributes_csv(args.dataset), args.overwrite)
    idx = read_index_csv(config.index_csv(args.dataset))
    dir_112 = Path(args.dir_112 or config.aligned_dir(args.dataset, 112))
    dir_224 = (
        None
        if args.no_mst
        else Path(args.dir_224 or config.aligned_dir(args.dataset, 224))
    )
    for d in (dir_112, dir_224):
        if d is not None and not d.is_dir():
            raise SystemExit(f"crop folder not found: {d}")
    try:
        df = estimate_attributes(
            idx,
            dir_112,
            dir_224,
            per_subject_ga=args.ga_per,
            per_subject_mst=0 if args.no_mst else args.mst_per,
            subjects=parse_list(args.subjects) or None,
            ctx_id=args.ctx_id,
        )
    except (MissingCropsError, ValueError) as exc:
        raise SystemExit(f"estimate_attributes: {exc}; nothing written") from exc
    write_attributes(df, out)
    log.info("wrote %s (%d rows, %d identities)", out, len(df), df.subject.nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
