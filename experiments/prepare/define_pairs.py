# SPDX-License-Identifier: MIT
"""Build the complete verification-pair matrix (pairs.parquet) of a dataset.

Enumerates every unordered pair of the index (``idx1 < idx2``, row-major), labels it
mated / non-mated and writes ``idx1, idx2, label`` as zstd parquet. With
``--sample`` it also writes a readable ``pairs_sample.csv`` (``name1, name2,
label``; Color FERET: all mated + 200k non-mated pairs, AI-Solutions-KK: 200k + 200k,
seed 0).

Memory: about 1 GB for Color FERET and 3 GB for AI-Solutions-KK.

Examples
--------
    python experiments/prepare/define_pairs.py --dataset colorferet
    python experiments/prepare/define_pairs.py --dataset kk --sample
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from face1kb import config
from face1kb.data import build_pairs, read_index_csv, write_pairs
from face1kb.data.cli_utils import output_path, setup_logging
from face1kb.data.pairs import SAMPLE_POLICIES, SAMPLE_SEED, sample_pairs

log = logging.getLogger("define_pairs")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=config.DATASETS)
    ap.add_argument("--index", default=None, help="index CSV (default: index.csv)")
    ap.add_argument("--out", default=None, help="output parquet (default: pairs)")
    ap.add_argument("--sample", action="store_true", help="also write pairs_sample.csv")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()

    index_path = Path(args.index) if args.index else config.index_csv(args.dataset)
    out = output_path(args.out, config.pairs_parquet(args.dataset), args.overwrite)
    sample_path = out.with_name("pairs_sample.csv")
    if args.sample and sample_path.exists() and not args.overwrite:
        raise SystemExit(f"{sample_path} exists; pass --overwrite to replace it")
    idx = read_index_csv(index_path)
    if not np.array_equal(idx["id"].to_numpy(), np.arange(len(idx))):
        raise SystemExit(f"{index_path}: ids must be 0..N-1 in row order")
    pairs = build_pairs(idx["subject"])
    n_mated = int(pairs["label"].sum())
    log.info(
        "%d images -> %d pairs (%d mated, %d non-mated)",
        len(idx),
        len(pairs),
        n_mated,
        len(pairs) - n_mated,
    )
    write_pairs(pairs, out)
    log.info("wrote %s", out)
    if args.sample:
        n_pos, n_neg = SAMPLE_POLICIES[args.dataset]
        sample = sample_pairs(pairs, idx["rel_path"], n_pos, n_neg, SAMPLE_SEED)
        sample.to_csv(sample_path, index=False)
        log.info("wrote %s (%d rows)", sample_path, len(sample))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
