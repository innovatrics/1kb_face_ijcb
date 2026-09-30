# SPDX-License-Identifier: MIT
"""Write an aligned-crop release in Hugging Face datasets format to crop folders.

The AI-Solutions-KK aligned crops are available on request from the authors as a
Hugging Face dataset (records ``image``, ``identity``, ``file_name``,
``resolution``). This writes them to ``aligned_<res>/<file_name>`` under the dataset
folder, the layout the pipeline starts from. Requires the ``datasets`` package.

Examples
--------
    python experiments/prepare/export_release.py --source /path/to/release
    python experiments/prepare/export_release.py --source <hub-id> --resolutions 112
"""

from __future__ import annotations

import argparse
import logging

from face1kb import config
from face1kb.data.cli_utils import output_dir_guard, parse_list, setup_logging
from face1kb.data.release import export_release

log = logging.getLogger("export_release")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, help="hub id, repo or saved dataset")
    ap.add_argument("--dataset", default="kk", choices=config.DATASETS)
    ap.add_argument("--resolutions", default="", help="default: all in the release")
    ap.add_argument("--split", default="train")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()
    output_dir_guard(False)

    counts = export_release(
        args.source,
        lambda r: config.aligned_dir(args.dataset, r),
        resolutions=parse_list(args.resolutions, int) or None,
        split=args.split,
        overwrite=args.overwrite,
    )
    for res, n in counts.items():
        log.info("aligned_%d: %d crops", res, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
