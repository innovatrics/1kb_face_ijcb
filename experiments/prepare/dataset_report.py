# SPDX-License-Identifier: MIT
r"""Dataset statistics and the dataset figures of the paper (Section 3).

From the index (and pairs, when present) it writes an aggregate summary
``dataset_stats/<dataset>_summary.json``. For AI-Solutions-KK it renders the paper
figures into ``figures/``: images per identity (index), estimated age / gender /
Monk Skin Tone (``attributes.csv``) and, given the raw AI-Solutions-KK download
(``--raw-dir``, one folder per identity) or a cached statistics JSON
(``--raw-stats``), the source file-size and dimension histograms. ``--examples``
also renders a montage of aligned crops of 10 seeded identities; it shows dataset
faces and must not be committed.

Examples
--------
    python experiments/prepare/dataset_report.py --dataset kk \
        --raw-dir /data/face_recognition_dataset --save-raw-stats
    python experiments/prepare/dataset_report.py --dataset colorferet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from face1kb import config
from face1kb.data import read_index_csv, read_pairs
from face1kb.data.attributes import read_attributes
from face1kb.data.cli_utils import setup_logging
from face1kb.data.stats import (
    attribute_figures,
    fig_examples,
    fig_images_per_subject,
    index_summary,
    raw_figures,
    raw_file_stats,
    save_json,
)

log = logging.getLogger("dataset_report")
_PREFIX = {"kk": "ds_kk", "colorferet": "ds_cf"}
_EXAMPLE_TITLES = {
    "kk": "AI-Solutions-KK: aligned-112 examples (10 random identities)",
    "colorferet": "Color FERET: aligned-112 examples (10 random subjects)",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=config.DATASETS)
    ap.add_argument("--raw-dir", default=None, help="raw AI-Solutions-KK download")
    ap.add_argument("--raw-stats", default=None, help="cached raw statistics JSON")
    ap.add_argument(
        "--save-raw-stats",
        action="store_true",
        help="cache the raw statistics (per-image sizes; keep local)",
    )
    ap.add_argument("--attributes", default=None, help="attributes CSV (KK)")
    ap.add_argument("--examples", action="store_true", help="example montage")
    ap.add_argument("--fig-dir", default=None, help="default: outputs/figures")
    ap.add_argument("--stats-dir", default=None, help="default: outputs/dataset_stats")
    args = ap.parse_args(argv)
    setup_logging()

    ds = args.dataset
    fig_dir = Path(args.fig_dir) if args.fig_dir else config.output_dir("figures")
    stats_dir = (
        Path(args.stats_dir) if args.stats_dir else config.output_dir("dataset_stats")
    )
    idx = read_index_csv(config.index_csv(ds))
    pairs_path = config.pairs_parquet(ds)
    pairs = read_pairs(pairs_path) if pairs_path.exists() else None
    summary = index_summary(idx, pairs)
    log.info("%s: %s", ds, json.dumps({k: summary[k] for k in ("images", "subjects")}))

    prefix = _PREFIX[ds]
    written = []
    if ds == "kk":
        per_id = idx.groupby("subject", sort=True).size().tolist()
        written.append(fig_images_per_subject(per_id, fig_dir / f"{prefix}_imgs.png"))
        att_path = (
            Path(args.attributes) if args.attributes else config.attributes_csv(ds)
        )
        if att_path.exists():
            written += attribute_figures(read_attributes(att_path), fig_dir, prefix)
        else:
            log.info("no attributes at %s: skipping age/gender/MST figures", att_path)
        stats = None
        if args.raw_stats:
            stats = json.loads(Path(args.raw_stats).read_text())
        elif args.raw_dir:
            stats = raw_file_stats(args.raw_dir)
            if args.save_raw_stats:
                save_json(stats, stats_dir / f"{ds}_raw_stats.json")
        if stats is not None:
            written += raw_figures(stats, fig_dir, prefix)
            summary["raw"] = {
                "images": stats["n_images"],
                "identities": stats["n_identities"],
            }
            log.info("%s raw download: %s", ds, json.dumps(summary["raw"]))
    written.insert(0, save_json(summary, stats_dir / f"{ds}_summary.json"))
    if args.examples:
        written.append(
            fig_examples(
                config.aligned_dir(ds, 112),
                idx,
                fig_dir / f"{prefix}_examples.png",
                _EXAMPLE_TITLES[ds],
            )
        )
    for p in written:
        log.info("wrote %s", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
