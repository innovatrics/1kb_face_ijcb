# SPDX-License-Identifier: MIT
"""Derived crop sets rendered from existing aligned crops.

* Crop-tightness variants (``aligned_112_tight``, ``_mid``, ``_fill``) rendered from
  the standard 224 px crops (:func:`face1kb.data.alignment.render_variant`).
* Other resolutions from an existing one
  (:func:`face1kb.data.alignment.derive_resolution`): exact for integer ratios
  (``aligned_56`` from ``aligned_112`` or ``aligned_224``), approximate otherwise.

Crops are read and written with OpenCV (channel order is preserved), and the output
keeps the relative paths of the input. Existing outputs are skipped unless
``overwrite`` is set (the number of skipped files is logged). A source crop that
cannot be read raises :class:`MissingCropsError` once all other crops are done,
unless ``strict=False``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from face1kb.data import alignment

log = logging.getLogger(__name__)


class MissingCropsError(FileNotFoundError):
    """Some source crops could not be read."""


def check_missing(missing: Sequence[str], total: int, what: str = "crops") -> None:
    """Raise :class:`MissingCropsError` if any of ``total`` inputs is ``missing``.

    Parameters
    ----------
    missing : sequence of str
        Paths of the inputs that could not be read.
    total : int
        Number of inputs.
    what : str
        What the inputs are, for the message.
    """
    if missing:
        raise MissingCropsError(
            f"{len(missing)} of {total} source {what} could not be read "
            f"(e.g. {missing[0]}); check the crop folder and the index"
        )


def _log_skipped(skipped: int, what: str) -> None:
    if skipped:
        log.info(
            "%s: skipped %d existing outputs (pass overwrite to recompute them)",
            what,
            skipped,
        )


def _variant_task(task) -> tuple[int, int, str | None]:
    """Render the variants of one crop: ``(written, skipped, unreadable source)``."""
    import cv2  # noqa: PLC0415

    src, dsts, out_res, overwrite = task
    todo = {v: d for v, d in dsts.items() if overwrite or not os.path.exists(d)}
    skipped = len(dsts) - len(todo)
    if not todo:
        return 0, skipped, None
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        log.warning("cannot read %s", src)
        return 0, skipped, src
    for variant, dst in todo.items():
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, alignment.render_variant(img, variant, out_res=out_res))
    return len(todo), skipped, None


def make_variants(
    src_dir: str | os.PathLike,
    dst_dirs: dict[str, str | os.PathLike],
    rel_paths: Sequence[str],
    out_res: int = alignment.TEMPLATE_SIZE,
    workers: int | None = None,
    overwrite: bool = False,
    strict: bool = True,
) -> int:
    """Render crop-tightness variants of every crop in ``rel_paths``.

    Parameters
    ----------
    src_dir : path
        Standard aligned crops (224 px recommended).
    dst_dirs : dict
        ``{variant: output folder}`` with variants from
        :data:`face1kb.data.alignment.CROP_VARIANTS`.
    rel_paths : sequence of str
        Crops to render.
    out_res : int
        Output resolution (112 px in the study).
    workers : int, optional
        Worker processes (default: CPU count minus 4).
    overwrite : bool
        Recompute outputs that already exist (otherwise they are skipped).
    strict : bool
        Raise :class:`MissingCropsError` if a source crop cannot be read (after
        rendering all others); otherwise only log it.

    Returns
    -------
    int
        Number of files written.
    """
    from tqdm import tqdm  # noqa: PLC0415

    for v in dst_dirs:
        alignment.variant_template(v)  # validates the name
    src_dir = Path(src_dir)
    tasks = [
        (
            str(src_dir / rp),
            {v: str(Path(d) / rp) for v, d in dst_dirs.items()},
            out_res,
            overwrite,
        )
        for rp in rel_paths
    ]
    n_workers = workers or max(1, (os.cpu_count() or 8) - 4)
    written = skipped = 0
    missing: list[str] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for n, k, bad in tqdm(
            ex.map(_variant_task, tasks, chunksize=32),
            total=len(tasks),
            desc="crop variants",
        ):
            written += n
            skipped += k
            if bad is not None:
                missing.append(bad)
    _log_skipped(skipped, "crop variants")
    if strict:
        check_missing(missing, len(tasks))
    return written


def _resolution_task(task) -> tuple[int, int, int, str | None]:
    """Derive one crop: ``(written, exact, skipped, unreadable source)``."""
    import cv2  # noqa: PLC0415

    src, dst, dst_res, overwrite = task
    if not overwrite and os.path.exists(dst):
        return 0, 0, 1, None
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        log.warning("cannot read %s", src)
        return 0, 0, 0, src
    out, exact = alignment.derive_resolution(img, dst_res)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, out)
    return 1, int(exact), 0, None


def make_resolution(
    src_dir: str | os.PathLike,
    dst_dir: str | os.PathLike,
    dst_res: int,
    rel_paths: Sequence[str],
    workers: int | None = None,
    overwrite: bool = False,
    strict: bool = True,
) -> tuple[int, int]:
    """Derive ``dst_res`` px crops from an aligned crop folder.

    Parameters
    ----------
    src_dir, dst_dir : path
        Source crop folder and output folder.
    dst_res : int
        Output resolution.
    rel_paths : sequence of str
        Crops to derive.
    workers : int, optional
        Worker processes (default: CPU count minus 4).
    overwrite : bool
        Recompute outputs that already exist (otherwise they are skipped, e.g. an
        existing ``aligned_112`` of another alignment stays in place).
    strict : bool
        Raise :class:`MissingCropsError` if a source crop cannot be read (after
        deriving all others); otherwise only log it.

    Returns
    -------
    tuple[int, int]
        Files written and how many of them are exact subsamplings.
    """
    from tqdm import tqdm  # noqa: PLC0415

    src_dir = Path(src_dir)
    tasks = [
        (str(src_dir / rp), str(Path(dst_dir) / rp), dst_res, overwrite)
        for rp in rel_paths
    ]
    n_workers = workers or max(1, (os.cpu_count() or 8) - 4)
    written = exact = skipped = 0
    missing: list[str] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for w, e, k, bad in tqdm(
            ex.map(_resolution_task, tasks, chunksize=64),
            total=len(tasks),
            desc=f"aligned_{dst_res}",
        ):
            written += w
            exact += e
            skipped += k
            if bad is not None:
                missing.append(bad)
    _log_skipped(skipped, f"aligned_{dst_res}")
    if strict:
        check_missing(missing, len(tasks))
    if written and exact < written:
        log.warning(
            "aligned_%d: %d of %d crops are resampled approximations "
            "(not an integer ratio of the source resolution)",
            dst_res,
            written - exact,
            written,
        )
    return written, exact


def check_subsampling(
    hi_dir: str | os.PathLike,
    lo_dir: str | os.PathLike,
    rel_paths: Sequence[str],
    n: int = 20,
) -> tuple[int, int]:
    """Check that the crops of ``lo_dir`` are exact subsamplings of ``hi_dir``.

    Crops of one alignment satisfy ``lo == hi[::k, ::k]`` whenever the side of the
    ``hi`` crops is ``k`` times that of the ``lo`` crops (e.g. ``aligned_112`` and
    ``aligned_224``). A mismatch means that the two folders come from different
    alignments.

    Parameters
    ----------
    hi_dir, lo_dir : path
        Crop folders of the larger and the smaller resolution.
    rel_paths : sequence of str
        Crops present in both folders; ``n`` of them, evenly spaced, are checked.
    n : int
        Number of crops to check.

    Returns
    -------
    tuple[int, int]
        Crops checked and crops that are exact subsamplings.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if not rel_paths or n <= 0:
        return 0, 0
    step = max(1, len(rel_paths) // n)
    checked = equal = 0
    for rp in list(rel_paths)[::step][:n]:
        hi = cv2.imread(str(Path(hi_dir) / rp), cv2.IMREAD_COLOR)
        lo = cv2.imread(str(Path(lo_dir) / rp), cv2.IMREAD_COLOR)
        if hi is None or lo is None:
            continue
        checked += 1
        k = hi.shape[0] // lo.shape[0]
        if k * lo.shape[0] == hi.shape[0] and np.array_equal(hi[::k, ::k], lo):
            equal += 1
    return checked, equal
