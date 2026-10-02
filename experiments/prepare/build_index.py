# SPDX-License-Identifier: MIT
r"""Build the canonical image index (index.csv) of a dataset from its aligned crops.

Scans ``aligned_<res>/<subject>/<stem>.png`` (default 112 px), sorts by
``(subject, stem)`` and writes ``id, rel_path, subject[, pose]`` (``pose`` for
Color FERET, decoded from the file-name pose code). Optionally checks that other
resolutions hold the same crop set. When both ``aligned_112`` and ``aligned_224``
exist, a sample of crops is checked for ``aligned_112 == aligned_224[::2, ::2]``,
which holds for crops of one alignment; a mismatch is reported as a warning. The
check needs OpenCV and is skipped (with a log message) without it.

Examples
--------
    python experiments/prepare/build_index.py --dataset kk
    python experiments/prepare/build_index.py --dataset colorferet \
        --check-res 64,96,112,168,224
"""

from __future__ import annotations

import argparse
import logging

from face1kb import config
from face1kb.data import build_index, write_index
from face1kb.data.cli_utils import output_path, parse_list, setup_logging
from face1kb.data.crops import check_subsampling
from face1kb.data.index import compare_image_sets, subject_counts

log = logging.getLogger("build_index")


def _have_opencv() -> bool:
    """Return whether OpenCV (used by the subsampling check) can be imported."""
    try:
        import cv2  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=config.DATASETS)
    ap.add_argument("--res", type=int, default=112, help="crop folder to scan")
    ap.add_argument(
        "--crop-dir", default=None, help="scan this folder instead of aligned_<res>"
    )
    ap.add_argument(
        "--check-res",
        default="",
        help="comma-separated resolutions whose crop sets must match",
    )
    ap.add_argument("--out", default=None, help="output CSV (default: index.csv)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()

    root = args.crop_dir or config.aligned_dir(args.dataset, args.res)
    out = output_path(args.out, config.index_csv(args.dataset), args.overwrite)
    try:
        df = build_index(root, dataset=args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"{args.dataset}: {exc}") from exc
    per = subject_counts(df)
    log.info(
        "%s: %d crops, %d subjects (%d-%d per subject) in %s",
        args.dataset,
        len(df),
        per.size,
        per.min(),
        per.max(),
        root,
    )
    check = parse_list(args.check_res, int)
    if check:
        folders = [config.aligned_dir(args.dataset, r) for r in check]
        try:
            report = compare_image_sets([root, *folders])
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"{args.dataset}: {exc}") from exc
        bad = {k: v for k, v in report.items() if v["missing"] or v["extra"]}
        for k, v in report.items():
            log.info("  %s: %s", k, v)
        if bad:
            raise SystemExit(f"crop sets differ between resolutions: {bad}")
    d112, d224 = (config.aligned_dir(args.dataset, r) for r in (112, 224))
    if d112.is_dir() and d224.is_dir() and not _have_opencv():
        log.info(
            "OpenCV is not installed: skipping the aligned_112 == "
            "aligned_224[::2, ::2] check (the index does not need it)"
        )
    elif d112.is_dir() and d224.is_dir():
        n, same = check_subsampling(d224, d112, df["rel_path"].tolist())
        if same < n:
            log.warning(
                "aligned_112 differs from aligned_224[::2, ::2] for %d of %d checked "
                "crops: the two folders come from different alignments; to replace "
                "aligned_112 by aligned_224[::2, ::2], run make_crop_variants.py "
                "--dataset %s --variants '' --resolutions 112 --overwrite",
                n - same,
                n,
                args.dataset,
            )
        else:
            log.info("aligned_112 == aligned_224[::2, ::2] on %d checked crops", n)
    write_index(df, out)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
