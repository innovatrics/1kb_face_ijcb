# SPDX-License-Identifier: MIT
"""Canonical image index of a dataset, built from its aligned crops.

The index (``index.csv``, :func:`face1kb.config.index_csv`) fixes the row order that
every later stage uses: embeddings row ``i`` is image ``id == i``, and the
verification pairs refer to these ids. It is a directory scan of one aligned crop
folder, ``aligned_<res>/<subject>/<stem>.png``, sorted by ``(subject, stem)`` as
strings, and has the columns

* ``id`` -- 0..N-1 in that order,
* ``rel_path`` -- ``<subject>/<stem>.png``, relative to any ``aligned_<res>`` folder,
* ``subject`` -- the subject folder name (Color FERET: the zero-padded NIST subject
  number, e.g. ``00001``; AI-Solutions-KK: the identity folder, e.g.
  ``pins_Adriana Lima``),
* ``pose`` -- Color FERET only: the pose description decoded from the pose code
  in the file name (:data:`CF_POSE_NAMES`).

For the paper's crop sets of both datasets this order is the order of the paper's
image index, so the verification pairs and the seeded impostor sample of the
accuracy stage are reproduced exactly from those crops (for AI-Solutions-KK, the
crops available from the authors). Color FERET crops aligned by the user from the
NIST distribution may contain a different set of images; the index, the pairs and
the impostor sample then differ accordingly.

``subject`` is read as a string (:func:`read_index_csv`, and likewise
:func:`face1kb.config.read_index`), so that it joins with the ``subject`` column of
``labels.csv`` (:func:`face1kb.data.read_labels`).

Tables keyed by image (labels, attributes, per-image results) should be joined on
``rel_path`` with :func:`align_to_index`, which checks that every index row is
matched exactly once.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

import pandas as pd

log = logging.getLogger(__name__)

#: File extensions accepted as crops by :func:`scan_crops`.
IMAGE_EXTS: tuple[str, ...] = (".png",)

#: Color FERET pose codes (third ``_``-separated token of the file name, e.g.
#: ``00001_930831_fa_a``) and the pose descriptions used in the paper tables.
CF_POSE_NAMES: dict[str, str] = {
    "fa": "frontal image",
    "fb": "frontal image taken shortly after the previous image",
    "pl": "profile left",
    "hl": "half left",
    "ql": "quarter left",
    "pr": "profile right",
    "hr": "half right",
    "qr": "quarter right",
    "ra": "head turned about 45 degree left",
    "rb": "head turned about 15 degree left",
    "rc": "head turned about 15 degree right",
    "rd": "head turned about 45 degree right",
    "re": "head turned about 75 degree right",
}
#: Nominal yaw angle of each Color FERET pose code as given in the NIST ground truth
#: (degrees; positive for the "left" codes ``hl``, ``pl``, ``ql``, ``ra``, ``rb``).
CF_POSE_YAW: dict[str, float] = {
    "fa": 0.0,
    "fb": 0.0,
    "pl": 90.0,
    "hl": 67.5,
    "ql": 22.5,
    "pr": -90.0,
    "hr": -67.5,
    "qr": -22.5,
    "ra": 45.0,
    "rb": 15.0,
    "rc": -15.0,
    "rd": -45.0,
    "re": -75.0,
}
#: Frontal Color FERET pose codes.
CF_FRONTAL_CODES: tuple[str, ...] = ("fa", "fb")

#: Datasets whose index carries a ``pose`` column.
_POSE_DATASETS = ("colorferet",)


def cf_pose_code(stem: str) -> str:
    """Pose code of a Color FERET image stem (``00001_930831_fa_a`` -> ``fa``)."""
    parts = Path(stem).stem.split("_")
    if len(parts) < 3 or parts[2] not in CF_POSE_NAMES:
        raise ValueError(f"not a Color FERET image name: {stem!r}")
    return parts[2]


def cf_pose(stem: str) -> str:
    """Pose description of a Color FERET image stem (see :data:`CF_POSE_NAMES`)."""
    return CF_POSE_NAMES[cf_pose_code(stem)]


def sort_key(subject: str, stem: str) -> tuple[str, str]:
    """Key of the canonical order: ``(subject, stem)`` compared as strings."""
    return (str(subject), str(stem))


def scan_crops(
    root: str | os.PathLike, exts: Sequence[str] = IMAGE_EXTS
) -> list[tuple[str, str, str]]:
    """List the crops of an ``aligned_<res>`` folder in canonical order.

    Parameters
    ----------
    root : path
        Folder with one sub-folder per subject holding ``<stem><ext>`` crops.
    exts : sequence of str
        Accepted file extensions (lower case). Crop names must use them exactly as
        given: every reader of the index resolves ``<subject>/<stem>.png``.

    Returns
    -------
    list of (subject, stem, file name)
        Sorted by ``(subject, stem)``. Hidden files and folders are ignored.

    Raises
    ------
    FileNotFoundError
        If ``root`` does not exist.
    ValueError
        If two files of one subject share a stem (e.g. ``a.png`` and ``a.jpg``), or
        a crop has an accepted extension in another letter case (``a.PNG``).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"aligned crop folder not found: {root}")
    exts = tuple(e.lower() for e in exts)
    out: list[tuple[str, str, str]] = []
    with os.scandir(root) as subjects:
        for sub in subjects:
            if sub.name.startswith(".") or not sub.is_dir():
                continue
            stems: dict[str, str] = {}
            with os.scandir(sub.path) as files:
                for f in files:
                    if f.name.startswith(".") or not f.is_file():
                        continue
                    stem, ext = os.path.splitext(f.name)
                    if ext.lower() not in exts:
                        continue
                    if ext not in exts:
                        raise ValueError(
                            f"crop {os.path.join(sub.path, f.name)!r}: rename the "
                            f"crops to the lower-case extension {ext.lower()!r}; "
                            "the index and its readers use lower-case names"
                        )
                    if stem in stems:
                        raise ValueError(
                            f"duplicate stem {stem!r} in {sub.path}: "
                            f"{stems[stem]!r} and {f.name!r}"
                        )
                    stems[stem] = f.name
            out.extend((sub.name, s, n) for s, n in stems.items())
    out.sort(key=lambda t: sort_key(t[0], t[1]))
    return out


def build_index(
    root: str | os.PathLike,
    dataset: str | None = None,
    pose: bool | None = None,
    exts: Sequence[str] = IMAGE_EXTS,
) -> pd.DataFrame:
    """Build the canonical index table from an aligned crop folder.

    Parameters
    ----------
    root : path
        An ``aligned_<res>`` folder (any resolution; all resolutions of a dataset
        hold the same images).
    dataset : str, optional
        Dataset name; ``colorferet`` adds the ``pose`` column by default.
    pose : bool, optional
        Force (``True``) or suppress (``False``) the Color FERET ``pose`` column.
    exts : sequence of str
        Accepted crop extensions.

    Returns
    -------
    DataFrame
        ``id, rel_path, subject[, pose]`` in canonical order, ``rel_path`` with the
        ``.png`` extension of the crops as found.
    """
    rows = scan_crops(root, exts)
    if not rows:
        raise ValueError(f"no crops found under {root}")
    df = pd.DataFrame(
        {
            "id": range(len(rows)),
            "rel_path": [f"{s}/{name}" for s, _, name in rows],
            "subject": [s for s, _, _ in rows],
        }
    )
    if pose is None:
        pose = dataset in _POSE_DATASETS
    if pose:
        df["pose"] = [cf_pose(stem) for _, stem, _ in rows]
    df["id"] = df["id"].astype("int64")
    return df


def write_index(df: pd.DataFrame, path: str | os.PathLike) -> Path:
    """Write an index table as CSV (no pandas index column)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def crop_rel_path(rel_path: str) -> str:
    """Return the ``.png`` crop path of an index entry (``a/b.jpg``: ``a/b.png``).

    Crops are always PNG; some index files list the source file names instead.
    """
    return PurePosixPath(str(rel_path)).with_suffix(".png").as_posix()


def read_index_csv(path: str | os.PathLike) -> pd.DataFrame:
    """Read an index CSV with the public column names.

    ``subject`` stays a string (e.g. ``00001``, as in
    :func:`face1kb.config.read_index`); an ``identity`` column is renamed to
    ``subject``; ``rel_path`` names the ``.png`` crop (:func:`crop_rel_path`), also
    for index files that list the source file names (e.g. ``.jpg``).
    """
    df = pd.read_csv(path, dtype={"subject": str, "identity": str, "rel_path": str})
    if "subject" not in df.columns and "identity" in df.columns:
        df = df.rename(columns={"identity": "subject"})
    if "rel_path" in df.columns:
        df["rel_path"] = [crop_rel_path(p) for p in df["rel_path"]]
    return df


def subject_counts(index: pd.DataFrame) -> pd.Series:
    """Count the images per subject (subjects in sorted order)."""
    return index.groupby("subject", sort=True).size()


def compare_image_sets(
    folders: Iterable[str | os.PathLike], exts: Sequence[str] = IMAGE_EXTS
) -> dict[str, dict[str, int]]:
    """Compare the crop sets of several ``aligned_<res>`` folders.

    Returns
    -------
    dict
        ``{folder: {"n": count, "missing": n_missing_vs_union,
        "extra": n_not_in_first}}``; all ``missing`` / ``extra`` counts are zero
        when the folders hold the same ``(subject, stem)`` set.
    """
    sets: dict[str, set[tuple[str, str]]] = {}
    for f in folders:
        sets[str(f)] = {(s, st) for s, st, _ in scan_crops(f, exts)}
    union = set().union(*sets.values()) if sets else set()
    first = next(iter(sets.values()), set())
    return {
        k: {"n": len(v), "missing": len(union - v), "extra": len(v - first)}
        for k, v in sets.items()
    }


def align_to_index(
    index: pd.DataFrame,
    table: pd.DataFrame,
    on: str = "rel_path",
    strict: bool = True,
) -> pd.DataFrame:
    """Reorder a per-image ``table`` to the row order of ``index``.

    Keys are compared without file extension (``a/b.png`` matches ``a/b.jpg``),
    since several tables list the source file names.

    Parameters
    ----------
    index : DataFrame
        Canonical index with column ``on``.
    table : DataFrame
        Per-image table with column ``on`` (unique keys).
    on : str
        Join column (default ``rel_path``).
    strict : bool
        Raise if an index row has no match; otherwise its fields are missing.

    Returns
    -------
    DataFrame
        ``table`` rows in index order (``len == len(index)``), with a fresh
        ``RangeIndex``; the ``on`` column holds the index values.
    """

    def _key(s: pd.Series) -> pd.Series:
        return s.astype(str).map(lambda p: os.path.splitext(p)[0])

    tkey = _key(table[on])
    if tkey.duplicated().any():
        dup = tkey[tkey.duplicated()].iloc[0]
        raise ValueError(f"duplicate key {dup!r} in the table")
    ikey = _key(index[on])
    if ikey.duplicated().any():
        raise ValueError(f"duplicate {on!r} values in the index")
    left = pd.DataFrame({"_key": ikey.to_numpy(), on: index[on].to_numpy()})
    right = table.drop(columns=[on]).reset_index(drop=True)
    right.insert(0, "_key", tkey.to_numpy())
    out = left.merge(right, on="_key", how="left", indicator="_merge")
    missing = out["_merge"] != "both"
    if missing.any() and strict:
        first = out.loc[missing, "_key"].iloc[0]
        raise KeyError(
            f"{int(missing.sum())} index rows have no match in the table, e.g. {first}"
        )
    return out.drop(columns=["_key", "_merge"]).reset_index(drop=True)
