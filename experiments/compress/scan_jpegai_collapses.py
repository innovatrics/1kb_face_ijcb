# SPDX-License-Identifier: MIT
r"""Scan the stored JPEG-AI bitstreams for implausibly small ("collapsed") streams.

A JPEG-AI stream far below its budget decodes to an unrecognisable image. Such files
can be a property of the codec on that input or a broken write (an interrupted
encode). This script counts, per compressed cell, the ``.jpegai`` files smaller than
``--frac`` x the cell budget (default
:data:`face1kb.baselines.jpeg_ai.MIN_PLAUSIBLE_FRAC`, the plausibility floor of the
encoder; :func:`face1kb.baselines.verify.scan_jpegai`).
With ``--reencode`` every flagged crop is encoded again with the benchmark's budget
fit; a crop is counted as *reproduced* when the fresh encode is again below the floor
or fails (``JpegAIError``), i.e. the collapse is a property of the codec; otherwise
the stored file was a broken write.

Output: one aggregate row per cell with at least one ``.jpegai`` file,
``dataset, cell, res, suffix, budget, n_files, median_bytes, min_bytes, n_below,
n_reencoded, n_reproduced``, written to
``OUTPUT_ROOT/codec_comparison/jpegai_collapses.csv``. No per-image data is written
unless ``--list FILE`` is given (a per-file list of the flagged streams; it names
dataset images and must not be published).

Examples
--------
    python experiments/compress/scan_jpegai_collapses.py
    python experiments/compress/scan_jpegai_collapses.py --datasets kk --reencode \
        --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
from pathlib import Path

import numpy as np

from face1kb import config

log = logging.getLogger("scan_jpegai_collapses")

_CELL = re.compile(r"^(\d+)(.*)px_(\d+)B$")
COLUMNS = (
    "dataset",
    "cell",
    "res",
    "suffix",
    "budget",
    "n_files",
    "median_bytes",
    "min_bytes",
    "n_below",
    "n_reencoded",
    "n_reproduced",
)


def parse_cell(name: str) -> tuple[int, str, int] | None:
    """``"112_adv_hfc_006px_512B"`` -> ``(112, "_adv_hfc_006", 512)``."""
    m = _CELL.match(name)
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else None


def jpegai_cells(dataset: str) -> list[tuple[str, int, str, int, Path]]:
    """``(cell, res, suffix, budget, jpeg_ai folder)`` of every compressed cell."""
    root = config.work_dir(dataset) / "compressed"
    out = []
    if not root.is_dir():
        return out
    for cell in sorted(root.iterdir()):
        parsed = parse_cell(cell.name)
        if parsed and (cell / "jpeg_ai").is_dir():
            out.append((cell.name, *parsed, cell / "jpeg_ai"))
    return out


def reencode_below(src: Path, budget: int, frac: float) -> bool:
    """Re-encode ``src``; True if the fresh stream is again below the floor."""
    from PIL import Image  # noqa: PLC0415

    from face1kb.baselines import get_codec  # noqa: PLC0415
    from face1kb.baselines.jpeg_ai import JpegAIError  # noqa: PLC0415

    with Image.open(src) as im:
        img = im.convert("RGB")
    try:
        data, _ = get_codec("jpeg_ai").encode_to_budget(
            img, budget, min_plausible_frac=0.0
        )
    except JpegAIError:
        return True
    return len(data) < frac * budget


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument("--frac", type=float, default=None, help="floor as budget fraction")
    ap.add_argument("--reencode", action="store_true", help="re-encode flagged crops")
    ap.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES")
    ap.add_argument("--out", default=None, help="aggregate CSV")
    ap.add_argument("--list", default=None, help="per-file list (not for publication)")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from face1kb.baselines.jpeg_ai import MIN_PLAUSIBLE_FRAC  # noqa: PLC0415
    from face1kb.baselines.verify import scan_jpegai  # noqa: PLC0415

    frac = MIN_PLAUSIBLE_FRAC if args.frac is None else args.frac
    rows, listed = [], []
    for ds in (d for d in args.datasets.split(",") if d):
        for cell, res, suffix, budget, folder in jpegai_cells(ds):
            sizes = [p.stat().st_size for p in folder.rglob("*.jpegai")]
            if not sizes:
                continue
            bad = scan_jpegai(folder, frac)
            n_re = n_rep = 0
            for path, size in bad:
                again = None
                if args.reencode:
                    rel = path.relative_to(folder).with_suffix(".png")
                    src = config.aligned_dir(ds, res, suffix) / rel
                    if src.is_file():
                        again = reencode_below(src, budget, frac)
                        n_re += 1
                        n_rep += int(again)
                listed.append((ds, cell, str(path.relative_to(folder)), size, again))
            rows.append(
                {
                    "dataset": ds,
                    "cell": cell,
                    "res": res,
                    "suffix": suffix,
                    "budget": budget,
                    "n_files": len(sizes),
                    "median_bytes": float(np.median(sizes)),
                    "min_bytes": int(min(sizes)),
                    "n_below": len(bad),
                    "n_reencoded": n_re,
                    "n_reproduced": n_rep,
                }
            )
            if bad:
                log.info(
                    "%s %s: %d of %d below %.0f B",
                    ds,
                    cell,
                    len(bad),
                    len(sizes),
                    frac * budget,
                )
    out = (
        Path(args.out)
        if args.out
        else config.output_dir("codec_comparison") / "jpegai_collapses.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUMNS))
        w.writeheader()
        w.writerows(rows)
    total = sum(r["n_below"] for r in rows)
    log.info(
        "wrote %s: %d cell(s), %d stream(s) below the floor", out, len(rows), total
    )
    if args.list:
        with open(args.list, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "cell", "file", "bytes", "reencode_below"])
            w.writerows(listed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
