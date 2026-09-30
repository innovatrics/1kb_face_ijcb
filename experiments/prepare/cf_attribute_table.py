# SPDX-License-Identifier: MIT
"""Color FERET ground-truth attribute distribution as a LaTeX table.

Reads ``labels.csv`` (from ``prepare_colorferet_labels.py``) and writes
``tables/cf_attributes.tex``: gender, race, glasses and pose blocks, each category
with its image count and share of the 11,338 recordings (Table ``tab:cf-attributes``).
With ``--paper-header`` the first (comment) line is the one of the published table,
and the output is byte-identical to it.

Examples
--------
    python experiments/prepare/cf_attribute_table.py
    python experiments/prepare/cf_attribute_table.py --labels labels.csv --out t.tex
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from face1kb import config
from face1kb.data.cli_utils import setup_logging
from face1kb.data.colorferet_labels import read_labels
from face1kb.data.stats import (
    CF_TABLE_ATTRS,
    CF_TABLE_HEADER,
    CF_TABLE_PAPER_HEADER,
    cf_attribute_table,
)

log = logging.getLogger("cf_attribute_table")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", default=None, help="default: colorferet labels.csv")
    ap.add_argument("--out", default=None, help="default: outputs/tables/...")
    ap.add_argument(
        "--paper-header",
        action="store_true",
        help="use the comment line of the published table (byte-identical output)",
    )
    args = ap.parse_args(argv)
    setup_logging()

    labels_path = Path(args.labels) if args.labels else config.labels_csv("colorferet")
    out = (
        Path(args.out)
        if args.out
        else config.output_dir("tables") / "cf_attributes.tex"
    )
    df = read_labels(labels_path)
    header = CF_TABLE_PAPER_HEADER if args.paper_header else CF_TABLE_HEADER
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(cf_attribute_table(df, header=header))
    log.info("wrote %s (n=%d)", out, len(df))
    for col, head in CF_TABLE_ATTRS:
        if col in df.columns:
            log.info("  %s: %d categories", head, df[col].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
