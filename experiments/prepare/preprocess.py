# SPDX-License-Identifier: MIT
"""Write preprocessed copies of the aligned crops (operators A1-A4, B1, B2, C1, C2).

Each operator's output goes to ``aligned_<res>_<op>/`` with the relative paths of
the index. A1-A4 run on the CPU (OpenCV, multiprocessing); B1/C1/C2 run BiRefNet
(GPU recommended, one mask per crop shared by the three) and B2 the MediaPipe
selfie segmenter. Existing outputs are skipped, so an interrupted run resumes
(``--overwrite`` recomputes them). The script exits with an error if a source crop
cannot be read.

Cost for the 11,335 Color FERET crops at 112 px: A1-A4 a few CPU minutes;
B1/C1/C2 about 25 GPU minutes (RTX 2080 Ti); B2 a few CPU minutes.

Examples
--------
    python experiments/prepare/preprocess.py --dataset colorferet
    python experiments/prepare/preprocess.py --dataset colorferet --ops A1,A3 --limit 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from face1kb import config
from face1kb.data import read_index_csv, scan_crops
from face1kb.data.cli_utils import output_dir_guard, parse_list, setup_logging
from face1kb.data.crops import MissingCropsError
from face1kb.data.preprocess import CPU_OPERATORS, SEG_OPERATORS, preprocess_folder

log = logging.getLogger("preprocess")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=config.DATASETS)
    ap.add_argument("--ops", default=",".join(CPU_OPERATORS + SEG_OPERATORS))
    ap.add_argument("--res", type=int, default=112)
    ap.add_argument("--batch", type=int, default=8, help="BiRefNet batch size")
    ap.add_argument("--device", default=None, help="torch device for BiRefNet")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N crops only")
    ap.add_argument("--out-root", default=None, help="write the folders here")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()
    output_dir_guard(args.out_root is not None)

    ops = parse_list(args.ops)
    src = config.aligned_dir(args.dataset, args.res)
    if not src.is_dir():
        raise SystemExit(f"source crop folder not found: {src}")
    if config.index_csv(args.dataset).exists():
        rel = read_index_csv(config.index_csv(args.dataset))["rel_path"].tolist()
    else:
        rel = [f"{s}/{n}" for s, _, n in scan_crops(src)]
    rel = rel[: args.limit] if args.limit else rel
    if args.out_root:
        dst = {op: Path(args.out_root) / f"aligned_{args.res}_{op}" for op in ops}
    else:
        dst = {op: config.aligned_dir(args.dataset, args.res, f"_{op}") for op in ops}
    try:
        written = preprocess_folder(
            src,
            dst,
            rel,
            batch=args.batch,
            workers=args.workers,
            overwrite=args.overwrite,
            device=args.device,
        )
    except MissingCropsError as exc:
        raise SystemExit(f"preprocess: {exc}") from exc
    for op, n in written.items():
        log.info("%s: wrote %d of %d files to %s", op, n, len(rel), dst[op])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
