# SPDX-License-Identifier: MIT
r"""Build the Color FERET labels table (labels.csv) from the NIST ground truth.

Reads the per-subject and per-recording ground truth of the NIST Color FERET
distribution -- an extracted copy (folder containing ``dvd1``/``dvd2``) or the
distribution tar archive -- and writes one row per recording with subject, image,
rel_path, pose, gender, race (7 classes), glasses, beard, mustache, age at capture
and the nominal pose angles. With an index it reports how many crops have labels.

Examples
--------
    python experiments/prepare/prepare_colorferet_labels.py --nist /data/colorferet
    python experiments/prepare/prepare_colorferet_labels.py \
        --nist colorferet.tar --format xml
"""

from __future__ import annotations

import argparse
import logging

from face1kb import config
from face1kb.data import read_index_csv
from face1kb.data.cli_utils import output_path, setup_logging
from face1kb.data.colorferet_labels import parse_ground_truth, write_labels

log = logging.getLogger("prepare_colorferet_labels")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--nist", required=True, help="NIST Color FERET folder or tar archive"
    )
    ap.add_argument("--format", default="auto", choices=("auto", "xml", "name_value"))
    ap.add_argument("--out", default=None, help="output CSV (default: labels.csv)")
    ap.add_argument(
        "--index", default=None, help="index CSV to check coverage (optional)"
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()

    out = output_path(args.out, config.labels_csv("colorferet"), args.overwrite)
    try:
        df = parse_ground_truth(args.nist, fmt=args.format)
    except FileNotFoundError as exc:
        raise SystemExit(f"prepare_colorferet_labels: {exc}") from exc
    log.info(
        "%d recordings, %d subjects; race: %s",
        len(df),
        df.subject.nunique(),
        df.race.value_counts().to_dict(),
    )
    write_labels(df, out)
    log.info("wrote %s", out)
    index_path = args.index
    if index_path is None and config.index_csv("colorferet").exists():
        index_path = config.index_csv("colorferet")
    if index_path is not None:
        idx = read_index_csv(index_path)
        have = set(df["rel_path"].str.rsplit(".", n=1).str[0])
        stems = idx["rel_path"].str.rsplit(".", n=1).str[0]
        missing = int((~stems.isin(have)).sum())
        log.info(
            "index %s: %d of %d crops have labels",
            index_path,
            len(idx) - missing,
            len(idx),
        )
        if missing:
            log.warning("%d crops have no NIST ground truth", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
