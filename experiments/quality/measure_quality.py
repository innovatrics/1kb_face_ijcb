# SPDX-License-Identifier: MIT
r"""Full-reference image quality of every compressed crop against its aligned original.

For each requested (dataset, codec, res, budget) cell, every bitstream under
``config.compressed_dir(dataset, res, budget, codec)`` is decoded with
:func:`face1kb.baselines.decode_file` (decoded PNG caches are used when present,
which is how the JPEG-AI and face1kb cells were decoded for the paper) and scored
against the lossless crop ``config.aligned_dir(dataset, res)/<subject>/<stem>.png``
with PSNR, SSIM, MS-SSIM, LPIPS and DISTS (:mod:`face1kb.eval.quality`, batches of
64). Files that fail to decode, decode to another size or have no reference crop are
skipped.

Outputs:

* per-image records, one parquet shard per cell,
  ``<shard-dir>/<codec>_<res>_<budget>.parquet`` (default shard dir
  ``$FACE1KB_WORK_ROOT/<dataset>/quality``). They are derived from the dataset images
  and stay local; existing shards are skipped unless ``--overwrite``;
* ``<out-dir>/quality_<dataset>.csv`` -- per-cell medians over all shards of the
  dataset (``dataset, codec, res, budget, n, psnr, ssim, ms_ssim, lpips, dists``);
* ``<out-dir>/quality_summary.csv`` -- the per-dataset CSVs present in ``out-dir``,
  concatenated.

``--aggregate-only`` rebuilds the CSVs from existing shards without scoring.
Decoding the face1kb, CompressAI and JPEG-AI bitstreams needs a CUDA GPU; the
metrics run on ``--device``.

Examples
--------
    python experiments/quality/measure_quality.py --datasets colorferet
    python experiments/quality/measure_quality.py --datasets kk --codecs webp \
        --resolutions 112 --budgets 1024 --limit 200
    python experiments/quality/measure_quality.py --datasets kk --aggregate-only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from face1kb import config
from face1kb.baselines import decode_file, extension
from face1kb.data.cli_utils import parse_list, setup_logging
from face1kb.eval import quality
from face1kb.eval.embeddings import CODECS

log = logging.getLogger("quality.measure")


def shard_dir(dataset: str, explicit: Path | None) -> Path:
    """Folder of the per-image shards of ``dataset``."""
    if explicit is not None:
        return explicit
    if config.LAYOUT == "legacy":
        raise SystemExit(
            "FACE1KB_LAYOUT=legacy: pass --shard-dir (the default location would be "
            "inside the legacy data store)"
        )
    return config.work_dir(dataset) / "quality"


def _read_rgb(path: Path) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def cell_items(dataset: str, codec: str, res: int, budget: int, *, limit: int, device):
    """Yield ``(record, decoded, reference)`` for the bitstreams of one cell."""
    cell = config.compressed_dir(dataset, res, budget, codec)
    files = sorted(cell.rglob(f"*{extension(codec)}"))
    if limit:
        files = files[:limit]
    ref_root = config.aligned_dir(dataset, res)
    skipped = 0
    for f in files:
        ref_path = ref_root / f.relative_to(cell).with_suffix(".png")
        if not ref_path.exists():
            skipped += 1
            continue
        try:
            dec = decode_file(f, res=res, device=device, cache=True)
            if dec is None or dec.shape[:2] != (res, res):
                skipped += 1
                continue
            ref = _read_rgb(ref_path)
        except Exception as exc:  # noqa: BLE001 - one broken file must not stop a cell
            log.debug("skip %s: %s", f, exc)
            skipped += 1
            continue
        meta = {
            "image": f.stem,
            "subject": f.parent.name,
            "codec": codec,
            "res": res,
            "budget": budget,
        }
        yield meta, dec, ref
    if skipped:
        log.info("  %s %d/%d: %d files skipped", codec, res, budget, skipped)


def aggregate(dataset: str, shards: Path, out_dir: Path) -> None:
    """Rewrite ``quality_<dataset>.csv`` and ``quality_summary.csv`` from shards."""
    import pandas as pd  # noqa: PLC0415

    files = sorted(shards.glob("*.parquet"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not files:
        log.warning("[%s] no quality shards under %s", dataset, shards)
    else:
        df = pd.concat((pd.read_parquet(s) for s in files), ignore_index=True)
        summary = quality.aggregate_quality(df, dataset, codecs=CODECS)
        out = out_dir / f"quality_{dataset}.csv"
        summary.to_csv(out, index=False)
        log.info("[%s] wrote %s (%d rows)", dataset, out, len(summary))
    # the combined summary is re-read from the per-dataset files on disk
    parts = [
        pd.read_csv(out_dir / f"quality_{ds}.csv")
        for ds in config.DATASETS
        if (out_dir / f"quality_{ds}.csv").exists()
    ]
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined.to_csv(out_dir / "quality_summary.csv", index=False)
        log.info("wrote %s (%d rows)", out_dir / "quality_summary.csv", len(combined))


def main(argv: list[str] | None = None) -> int:
    """Score the requested cells, then rebuild the median CSVs."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default="colorferet,kk")
    ap.add_argument("--codecs", default=",".join(CODECS))
    ap.add_argument("--resolutions", default=",".join(map(str, config.RESOLUTIONS)))
    ap.add_argument("--budgets", default=",".join(map(str, config.BUDGETS)))
    ap.add_argument("--device", default="cuda", help="decode and metric device")
    ap.add_argument("--batch", type=int, default=quality.BATCH_SIZE)
    ap.add_argument("--limit", type=int, default=0, help="first N files per cell")
    ap.add_argument("--overwrite", action="store_true", help="rescore existing shards")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--shard-dir", type=Path, default=None)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: $FACE1KB_OUTPUT_ROOT/quality",
    )
    args = ap.parse_args(argv)
    setup_logging()
    import pandas as pd  # noqa: PLC0415

    out_dir = args.out_dir or config.output_dir("quality")
    datasets = parse_list(args.datasets)
    if len(datasets) > 1 and args.shard_dir is not None:
        ap.error("--shard-dir takes one dataset")
    metrics = None
    for ds in datasets:
        shards = shard_dir(ds, args.shard_dir)
        if not args.aggregate_only:
            shards.mkdir(parents=True, exist_ok=True)
            cells = [
                (c, r, b)
                for c in parse_list(args.codecs)
                for r in parse_list(args.resolutions, int)
                for b in parse_list(args.budgets, int)
            ]
            for codec, res, budget in cells:
                shard = shards / f"{codec}_{res}_{budget}.parquet"
                if shard.exists() and not args.overwrite:
                    continue
                if not config.compressed_dir(ds, res, budget, codec).is_dir():
                    continue
                if metrics is None:
                    metrics = quality.make_metrics(args.device)
                recs = quality.score_images(
                    cell_items(
                        ds, codec, res, budget, limit=args.limit, device=args.device
                    ),
                    metrics,
                    batch=args.batch,
                    device=args.device,
                )
                if recs:
                    pd.DataFrame(recs).to_parquet(shard, index=False)
                    log.info("[%s] %s: %d crops", ds, shard.name, len(recs))
        aggregate(ds, shards, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
