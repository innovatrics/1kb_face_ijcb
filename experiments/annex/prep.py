# SPDX-License-Identifier: MIT
r"""ISO/IEC 29794-5 annex study, stage 0: evaluation subsets and their crop cache.

For each dataset this builds, once:

* the **evaluation subset** (``meta.parquet``): at most 4 crops per Color FERET
  subject and 30 per AI-Solutions-KK identity, frontal crops first. Rows are sorted
  by (subject, frontal first, image stem) and the first crops of each subject are
  kept, which gives 3,976 Color FERET crops (994 subjects, 2,483 frontal) and 3,150
  AI-Solutions-KK crops (105 identities). A Color FERET crop is frontal when its pose
  code is ``fa`` or ``fb``; every AI-Solutions-KK crop counts as frontal. The row
  order matters downstream: the impostor samples of ``score.py`` and ``ci.py`` are
  seeded draws over row indices;
* the **crop cache**: one ``(N, r, r, 3)`` uint8 RGB array per annex resolution
  ``r`` in 56, 64, 80, 96, 112, 168, 224 (``crops_<r>.npy``), read from the aligned
  crops ``aligned_<r>``. A resolution without a crop folder is derived from a larger
  one with :func:`face1kb.data.alignment.derive_resolution`: exactly by subsampling
  when the size divides (56 px = ``aligned_112[::2, ::2]``, identical to a crop warped
  at 56 px), otherwise approximately by resampling the 224 px crop (80 px; a warning
  is logged). ``crops.json`` records the source of each resolution;
* the **106-point landmarks** (``lmk106.npy``, ``(N, 106, 2)`` float32 in 224 px crop
  coordinates) of InsightFace ``2d106det`` run on the 224 px crops with the whole
  crop as the face box. They build the face-contour mask of the ``ofiq_landmarks``
  manipulation (OFIQ's own landmarker is not used).

The paper warped every annex resolution directly from the source photographs. For
Color FERET the 64-224 px crops and the 56 px crops are identical to those; the 80
px crops of this cache are an approximation. For AI-Solutions-KK the paper's annex
crops came from a separate alignment and differ from the released crops at every
resolution (``docs/reproduce_annex.md``).

Output: ``WORK_ROOT/<dataset>/annex/cache/`` (``--cache-root DIR`` writes
``DIR/<dataset>/`` instead). Existing files are kept unless ``--overwrite`` is given.
CPU for the cache (about a minute), one GPU or the CPU for the landmarks (about a
minute).

Examples
--------
    python experiments/annex/prep.py
    python experiments/annex/prep.py --datasets kk --ctx-id -1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from face1kb import config
from face1kb.data import cf_pose_code
from face1kb.data.alignment import derive_resolution, is_exact_subsampling
from face1kb.data.cli_utils import parse_list, setup_logging
from face1kb.data.index import CF_FRONTAL_CODES

log = logging.getLogger("annex.prep")

#: Annex resolution ladder (px).
RESOLUTIONS: tuple[int, ...] = (56, 64, 80, 96, 112, 168, 224)
#: Crops kept per subject.
PER_SUBJECT: dict[str, int] = {"colorferet": 4, "kk": 30}
#: Resolution the 106-point landmarks are computed at.
LANDMARK_RES = 224
META_COLUMNS = ("sid", "image", "rel_path", "frontal", "id")


def cache_dir(dataset: str, cache_root: str | None = None) -> Path:
    """Cache folder of ``dataset``: ``WORK_ROOT/<dataset>/annex/cache`` by default."""
    if cache_root:
        return Path(cache_root) / dataset
    return config.work_dir(dataset) / "annex" / "cache"


def build_meta(dataset: str, index: pd.DataFrame) -> pd.DataFrame:
    """Select the evaluation subset of ``dataset`` from its image index.

    Parameters
    ----------
    dataset
        ``colorferet`` or ``kk``.
    index
        The image index (``id, rel_path, ...``; :func:`face1kb.config.read_index`).

    Returns
    -------
    DataFrame
        ``sid, image, rel_path, frontal, id``: subject folder, image stem, crop path
        relative to ``aligned_<res>``, frontal flag and index row id.
    """
    rel = index["rel_path"].astype(str)
    df = pd.DataFrame(
        {
            "sid": [Path(p).parent.name for p in rel],
            "image": [Path(p).stem for p in rel],
            "rel_path": rel.to_numpy(),
            "id": index["id"].to_numpy(),
        }
    )
    if dataset == "colorferet":
        df["frontal"] = [cf_pose_code(s) in CF_FRONTAL_CODES for s in df.image]
    else:
        df["frontal"] = True  # no pose labels: the whole subset is the population
    df = df.sort_values(["sid", "frontal", "image"], ascending=[True, False, True])
    keep = df.groupby("sid").head(PER_SUBJECT[dataset]).reset_index(drop=True)
    return keep[list(META_COLUMNS)]


def _read(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def crop_source(dataset: str, res: int) -> tuple[int, bool]:
    """Resolution to read for ``res`` and whether the result is exact.

    ``aligned_<res>`` itself when it exists, else the smallest available crop folder
    whose size is a multiple of ``res`` (exact subsampling), else ``aligned_224``
    (approximate resampling).
    """
    if config.aligned_dir(dataset, res).is_dir():
        return res, True
    for src in sorted(r for r in (*config.RESOLUTIONS, 224) if r > res):
        if is_exact_subsampling(src, res) and config.aligned_dir(dataset, src).is_dir():
            return src, True
    if config.aligned_dir(dataset, 224).is_dir():
        return 224, False
    raise SystemExit(f"{dataset}: no aligned crops to build the {res} px cache from")


def build_crops(
    dataset: str, meta: pd.DataFrame, out: Path, overwrite: bool
) -> dict[str, str]:
    """Write ``crops_<res>.npy`` for every annex resolution; return their sources."""
    sources = {}
    for res in RESOLUTIONS:
        path = out / f"crops_{res}.npy"
        src_res, exact = crop_source(dataset, res)
        how = f"aligned_{src_res}"
        if src_res != res:
            how += " subsampled" if exact else " resampled (approximate)"
        sources[str(res)] = how
        if path.exists() and not overwrite:
            continue
        if not exact:
            log.warning(
                "%s: no aligned_%d crops; resampling aligned_%d (an approximation of "
                "a direct warp)",
                dataset,
                res,
                src_res,
            )
        src = config.aligned_dir(dataset, src_res)
        arr = np.lib.format.open_memmap(
            path.with_suffix(".tmp.npy"), "w+", np.uint8, (len(meta), res, res, 3)
        )
        for i, rel in enumerate(tqdm(meta.rel_path, desc=f"{dataset} {res} px")):
            img = _read(src / rel)
            arr[i] = img if src_res == res else derive_resolution(img, res)[0]
        arr.flush()
        del arr
        path.with_suffix(".tmp.npy").replace(path)
    return sources


def build_landmarks(out: Path, ctx_id: int, overwrite: bool) -> None:
    """Run InsightFace ``2d106det`` on the 224 px crops (the crop is the face box)."""
    path = out / "lmk106.npy"
    if path.exists() and not overwrite:
        return
    from insightface.app.common import Face  # noqa: PLC0415
    from insightface.model_zoo import get_model  # noqa: PLC0415

    from face1kb.data.fetch import insightface_model_path  # noqa: PLC0415

    model = get_model(str(insightface_model_path("2d106det.onnx")))
    model.prepare(ctx_id=ctx_id)
    crops = np.load(out / f"crops_{LANDMARK_RES}.npy", mmap_mode="r")
    box = np.array([0, 0, LANDMARK_RES - 1, LANDMARK_RES - 1], np.float32)
    lm = np.zeros((len(crops), 106, 2), np.float32)
    for i in tqdm(range(len(crops)), desc="2d106det"):
        img = np.ascontiguousarray(crops[i][:, :, ::-1])  # RGB -> BGR
        lm[i] = model.get(img, Face(bbox=box))
    np.save(path, lm)


def main(argv: list[str] | None = None) -> int:
    """Build the subsets, crop caches and landmark caches."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument("--cache-root", default=None, help="write DIR/<dataset>/ instead")
    ap.add_argument("--ctx-id", type=int, default=0, help="GPU of 2d106det (-1: CPU)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()
    if config.LAYOUT == "legacy" and not args.cache_root:
        raise SystemExit("FACE1KB_LAYOUT=legacy: pass --cache-root")

    for ds in parse_list(args.datasets):
        out = cache_dir(ds, args.cache_root)
        out.mkdir(parents=True, exist_ok=True)
        meta_path = out / "meta.parquet"
        if meta_path.exists() and not args.overwrite:
            meta = pd.read_parquet(meta_path)
        else:
            meta = build_meta(ds, config.read_index(ds))
            meta.to_parquet(meta_path)
        log.info(
            "%s: %d crops, %d subjects, %d frontal",
            ds,
            len(meta),
            meta.sid.nunique(),
            int(meta.frontal.sum()),
        )
        sources = build_crops(ds, meta, out, args.overwrite)
        (out / "crops.json").write_text(json.dumps(sources, indent=2) + "\n")
        build_landmarks(out, args.ctx_id, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
