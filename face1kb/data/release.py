# SPDX-License-Identifier: MIT
"""Export an aligned-crop release in Hugging Face ``datasets`` format to folders.

The AI-Solutions-KK aligned crops are distributed on request as a Hugging Face
dataset with one record per crop: ``image`` (PNG), ``identity``, ``file_name``
(``<identity folder>/<stem>.png``) and ``resolution`` (px). :func:`export_release`
writes the records to the directory layout the pipeline starts from::

    <out_root>/aligned_<resolution>/<file_name>

Existing files are kept. Requires the ``datasets`` package.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

log = logging.getLogger(__name__)


def export_release(
    source: str | os.PathLike,
    out_dir: Callable[[int], Path],
    resolutions: Sequence[int] | None = None,
    split: str = "train",
    overwrite: bool = False,
) -> dict[int, int]:
    """Write the crops of an aligned-crop dataset to ``aligned_<res>`` folders.

    Parameters
    ----------
    source : str or path
        A Hugging Face dataset id, a local dataset repository, or a folder saved
        with ``Dataset.save_to_disk``.
    out_dir : callable
        Maps a resolution to its output folder (e.g.
        ``lambda r: config.aligned_dir("kk", r)``).
    resolutions : sequence of int, optional
        Resolutions to export (default: all in the dataset).
    split : str
        Dataset split.
    overwrite : bool
        Rewrite files that exist.

    Returns
    -------
    dict
        Number of crops per resolution (written or already present).
    """
    import datasets  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    src = Path(source)
    if src.is_dir() and (src / "dataset_info.json").exists():
        ds = datasets.load_from_disk(str(src))
        if isinstance(ds, datasets.DatasetDict):
            ds = ds[split]
    else:
        ds = datasets.load_dataset(str(source), split=split)
    res_col = ds["resolution"] if "resolution" in ds.column_names else None
    wanted = sorted(set(res_col)) if resolutions is None else list(resolutions)
    counts: dict[int, int] = {}
    for res in wanted:
        sub = ds.filter(lambda r, res=res: r == res, input_columns="resolution")
        root = Path(out_dir(int(res)))
        n = 0
        for rec in tqdm(sub, desc=f"aligned_{res}"):
            target = root / rec["file_name"]
            if overwrite or not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                rec["image"].save(target, "PNG")
            n += 1
        counts[int(res)] = n
        log.info("aligned_%d: %d crops in %s", res, n, root)
    return counts
