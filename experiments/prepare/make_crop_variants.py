# SPDX-License-Identifier: MIT
r"""Render crop-tightness variants and derived resolutions from aligned crops.

* ``--variants tight,mid,fill``: the alignment-tightness presets of the ablation
  (IOD / width 0.40, 0.46, 0.52; eye line at 0.34, 0.27, 0.20 of the height),
  rendered at 112 px from the standard 224 px crops into ``aligned_112_<variant>``.
  They differ from crops warped directly from the source photographs by the second
  interpolation (about 1.5/255 mean absolute difference).
* ``--resolutions 56``: further resolutions of the standard crop. A resolution that
  divides an available one is an exact subsampling (``aligned_56 =
  aligned_112[::2, ::2]``); any other one is resampled from 224 px and is only an
  approximation of a direct warp (a warning is logged).

Existing output files are kept unless ``--overwrite`` is given (for example to
replace an ``aligned_112`` of another alignment by ``aligned_224[::2, ::2]``). The
script exits with an error if a source crop cannot be read.

Examples
--------
    python experiments/prepare/make_crop_variants.py --dataset colorferet
    python experiments/prepare/make_crop_variants.py --dataset colorferet \
        --variants "" --resolutions 56
    python experiments/prepare/make_crop_variants.py --dataset kk \
        --variants "" --resolutions 112 --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from face1kb import config
from face1kb.data import read_index_csv, scan_crops
from face1kb.data.alignment import (
    CROP_VARIANTS,
    TEMPLATE_SIZE,
    is_exact_subsampling,
    variants_metadata,
)
from face1kb.data.cli_utils import output_dir_guard, parse_list, setup_logging
from face1kb.data.crops import MissingCropsError, make_resolution, make_variants

log = logging.getLogger("make_crop_variants")


def _crop_dir(dataset: str, res: int, suffix: str, out_root: Path | None) -> Path:
    if out_root is not None:
        return out_root / f"aligned_{res}{suffix}"
    return config.aligned_dir(dataset, res, suffix)


def _rel_paths(dataset: str, src: Path, limit: int) -> list[str]:
    if config.index_csv(dataset).exists():
        rel = read_index_csv(config.index_csv(dataset))["rel_path"].tolist()
    else:
        rel = [f"{s}/{n}" for s, _, n in scan_crops(src)]
    return rel[:limit] if limit else rel


def _derive_source(dataset: str, res: int) -> int:
    for src in sorted(config.RESOLUTIONS, reverse=True):
        if (
            src != res
            and is_exact_subsampling(src, res)
            and config.aligned_dir(dataset, src).is_dir()
        ):
            return src
    return max(config.RESOLUTIONS)


def _require_dir(path: Path) -> None:
    if not path.is_dir():
        raise SystemExit(f"source crop folder not found: {path}")


def _report(what: str, n: int, total: int, detail: str = "") -> None:
    """Log what was written; say so explicitly when existing files were kept."""
    kept = total - n
    if n == 0:
        log.info(
            "%s: nothing written, all %d outputs exist (pass --overwrite to "
            "recompute them)",
            what,
            total,
        )
        return
    log.info(
        "%s: wrote %d files%s%s",
        what,
        n,
        f" ({detail})" if detail else "",
        f", kept {kept} existing (pass --overwrite to recompute them)" if kept else "",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=config.DATASETS)
    ap.add_argument("--variants", default=",".join(CROP_VARIANTS))
    ap.add_argument("--src-res", type=int, default=224, help="source of the variants")
    ap.add_argument("--out-res", type=int, default=TEMPLATE_SIZE)
    ap.add_argument(
        "--resolutions", default="", help="derived resolutions, e.g. 56 or 56,80"
    )
    ap.add_argument("--out-root", default=None, help="write the folders here")
    ap.add_argument("--limit", type=int, default=0, help="first N crops only")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()
    output_dir_guard(args.out_root is not None)
    out_root = Path(args.out_root) if args.out_root else None

    variants = parse_list(args.variants)
    if variants:
        src = config.aligned_dir(args.dataset, args.src_res)
        _require_dir(src)
        rel = _rel_paths(args.dataset, src, args.limit)
        dst = {
            v: _crop_dir(args.dataset, args.out_res, f"_{v}", out_root)
            for v in variants
        }
        try:
            n = make_variants(src, dst, rel, args.out_res, args.workers, args.overwrite)
        except MissingCropsError as exc:
            raise SystemExit(f"variants {variants}: {exc}") from exc
        _report(f"variants {variants} from {src}", n, len(rel) * len(variants))
        meta = variants_metadata({v: CROP_VARIANTS[v] for v in variants})
        meta["source_resolution"] = args.src_res
        meta["output_resolution"] = args.out_res
        meta_dir = (
            out_root if out_root is not None else config.dataset_dir(args.dataset)
        )
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "variants_templates.json").write_text(json.dumps(meta, indent=2))

    for res in parse_list(args.resolutions, int):
        src_res = _derive_source(args.dataset, res)
        src = config.aligned_dir(args.dataset, src_res)
        _require_dir(src)
        rel = _rel_paths(args.dataset, src, args.limit)
        dst = _crop_dir(args.dataset, res, "", out_root)
        try:
            n, exact = make_resolution(src, dst, res, rel, args.workers, args.overwrite)
        except MissingCropsError as exc:
            raise SystemExit(f"aligned_{res}: {exc}") from exc
        kind = "exact subsampling" if exact == n else "resampled, approximate"
        _report(f"aligned_{res} from aligned_{src_res}", n, len(rel), kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
