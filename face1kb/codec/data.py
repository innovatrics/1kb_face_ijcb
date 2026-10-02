# SPDX-License-Identifier: MIT
"""Training data for the codecs: uniform random sampling of aligned face crops.

The shipped codecs were trained on WebFace42M (Zhu et al., CVPR 2021), used for
research purposes only; the dataset is not distributed with this package. Any
collection of aligned 112 x 112 RGB face crops can be used instead, in one of two
forms:

* a HuggingFace ``datasets`` folder written by ``Dataset.save_to_disk`` with an
  ``image`` column (``datasets.Image`` feature or ``{"bytes", "path"}`` struct) and a
  string ``subject_id`` column, or
* an image folder ``<root>/<subject>/<image>.{jpg,jpeg,png}`` (the layout of the
  official WebFace42M release); the subject id is the folder path relative to
  ``root``.

Sampling is step-based: every draw is a uniformly random index into the whole
source (with replacement), rejected if it belongs to the other split. There are no
epochs, no shuffle buffer and no augmentation. Train and validation splits are
disjoint in identity: a subject belongs to the validation split when
``crc32(subject_id) % val_mod == 0``. The split therefore depends on the subject id
strings; the shipped weights were trained on a ``save_to_disk`` copy of WebFace42M
whose subject ids are the consecutive integers ``"0", "1", ...``.
"""

from __future__ import annotations

import io
import logging
import os
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
#: Consecutive unreadable rows after which :class:`RandomFaceDataset` gives up.
MAX_CONSECUTIVE_FAILURES = 1000
#: Consecutive draws from the other split after which the split counts as empty.
MAX_CONSECUTIVE_REJECTIONS = 100_000
#: Entries of a cached file list that are checked for existence when it is read.
LIST_CHECK_ENTRIES = 8


def _is_val(subject_id: str, val_mod: int) -> bool:
    """Return True if this subject belongs to the validation (held-out) split."""
    return zlib.crc32(subject_id.encode()) % val_mod == 0


def _subject_int(sid: str) -> int:
    """Integer label of a subject id (the integer itself, else its crc32)."""
    try:
        return int(sid)
    except ValueError:
        return zlib.crc32(sid.encode())


def to_tensor01(img, size: int | None = 112) -> torch.Tensor:
    """Convert an image to a ``(3, H, W)`` float32 RGB tensor in ``[0, 1]``.

    Accepts a ``PIL.Image``, the HuggingFace ``Image`` storage struct
    ``{"bytes", "path"}``, encoded ``bytes`` or a file path. Images whose side is not
    ``size`` are resized (bicubic) to ``size x size``; aligned 112 px crops, such as
    WebFace42M, pass through unchanged. ``size=None`` disables the resize.
    """
    from PIL import Image as PILImage  # noqa: PLC0415

    if isinstance(img, dict):
        raw = img.get("bytes")
        img = PILImage.open(io.BytesIO(raw)) if raw else PILImage.open(img["path"])
    elif isinstance(img, (bytes, bytearray)):
        img = PILImage.open(io.BytesIO(img))
    elif isinstance(img, (str, os.PathLike)):
        img = PILImage.open(img)
    img = img.convert("RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), PILImage.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class ImageFolderSource:
    """Map-style source over ``<root>/<subject>/<image>`` files.

    The file list is sorted (by relative path) so that indices are reproducible. A
    scan of a large tree is slow; pass ``list_file`` to cache it (one relative path
    per line; written on first use, read afterwards). When a cached list is read,
    its first entries must exist under ``root``; otherwise the list is stale or
    belongs to another root and a ``FileNotFoundError`` is raised.
    """

    def __init__(self, root: str | Path, list_file: str | Path | None = None):
        self.root = Path(root)
        lf = Path(list_file) if list_file else None
        if lf is not None and lf.exists():
            rels = [ln for ln in lf.read_text().splitlines() if ln]
            missing = [
                r for r in rels[:LIST_CHECK_ENTRIES] if not (self.root / r).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"file list {lf} does not match {self.root}: e.g. {missing[0]!r} "
                    "does not exist; delete the list to rescan the folder"
                )
        else:
            rels = []
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames.sort()
                for fn in sorted(filenames):
                    if fn.lower().endswith(IMAGE_EXTS):
                        rel = Path(dirpath, fn).relative_to(self.root)
                        rels.append(rel.as_posix())
            rels.sort()
            if lf is not None:
                lf.write_text("\n".join(rels) + "\n")
        if not rels:
            raise FileNotFoundError(f"no images found under {self.root}")
        self.rels = rels

    def __len__(self) -> int:
        return len(self.rels)

    def __getitem__(self, idx: int) -> dict:
        rel = self.rels[idx]
        return {
            "image": str(self.root / rel),
            "subject_id": str(Path(rel).parent.as_posix()),
        }


def open_source(root: str | Path, list_file: str | Path | None = None):
    """Open a training source: a ``save_to_disk`` folder or an image folder."""
    root = Path(root)
    if (root / "dataset_info.json").exists() or (root / "dataset_dict.json").exists():
        from datasets import DatasetDict, load_from_disk  # noqa: PLC0415

        ds = load_from_disk(str(root))
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
        return ds
    if not root.is_dir():
        raise FileNotFoundError(f"training data not found: {root}")
    return ImageFolderSource(root, list_file=list_file)


class RandomFaceDataset(IterableDataset):
    """Infinite uniform-random sampler over a source, filtered to one split.

    Each worker of each rank draws from its own fixed stream
    ``default_rng(seed + 7919 * rank + 104729 * worker + 1)``. Rows that fail to
    decode are skipped (logged as warnings, rate-limited); after
    :data:`MAX_CONSECUTIVE_FAILURES` consecutive failures, or
    :data:`MAX_CONSECUTIVE_REJECTIONS` consecutive draws from the other split (an
    empty split), a ``RuntimeError`` is raised instead of looping forever. Neither
    check changes the sample stream.

    Parameters
    ----------
    root
        HuggingFace ``save_to_disk`` folder or image folder.
    split
        ``"train"`` or ``"val"``.
    val_mod
        Hold out the subjects with ``crc32(subject_id) % val_mod == 0``.
    seed, rank, world_size
        Sampling stream (``world_size`` is informational; ranks differ by ``rank``).
    image_size
        Side length the crops are resized to if necessary (see :func:`to_tensor01`).
    list_file
        Optional cached file list for image folders.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        val_mod: int = 64,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        image_size: int | None = 112,
        list_file: str | Path | None = None,
    ):
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(split)
        self.root = root
        self.list_file = list_file
        self.ds = open_source(root, list_file=list_file)
        self.n = len(self.ds)
        self.split = split
        self.val_mod = val_mod
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.image_size = image_size

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        # Distinct, fixed stream per (rank, worker): coprime multipliers.
        rng = np.random.default_rng(self.seed + 7919 * self.rank + 104729 * wid + 1)
        want_val = self.split == "val"
        fails = rejects = total_fails = 0
        while True:
            idx = int(rng.integers(self.n))
            other_split = False
            try:
                ex = self.ds[idx]
                sid = str(ex["subject_id"])
                other_split = _is_val(sid, self.val_mod) != want_val
                if not other_split:
                    rejects = 0
                    t = to_tensor01(ex["image"], size=self.image_size)
            except Exception as e:  # noqa: BLE001 - skip rare corrupt rows
                fails += 1
                total_fails += 1
                if total_fails <= 10 or total_fails % 1000 == 0:
                    logger.warning(
                        "skipping unreadable row %d of %s (%d skipped so far): %r",
                        idx,
                        self.root,
                        total_fails,
                        e,
                    )
                if fails >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"{fails} consecutive rows of {self.root} could not be "
                        f"loaded (list file: {self.list_file}); last error: {e!r}"
                    ) from e
                continue
            if other_split:
                rejects += 1
                if rejects >= MAX_CONSECUTIVE_REJECTIONS:
                    raise RuntimeError(
                        f"no sample of the {self.split!r} split in {rejects} "
                        f"consecutive draws from {self.root}; is the split empty "
                        f"(val_mod={self.val_mod})?"
                    )
                continue
            fails = 0
            yield t, _subject_int(sid)


def make_random_loader(
    root: str | Path,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 8,
    val_mod: int = 64,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
    image_size: int | None = 112,
    list_file: str | Path | None = None,
) -> DataLoader:
    """DataLoader over :class:`RandomFaceDataset`: random, infinite, no augmentation."""
    ds = RandomFaceDataset(
        root,
        split=split,
        val_mod=val_mod,
        seed=seed,
        rank=rank,
        world_size=world_size,
        image_size=image_size,
        list_file=list_file,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def collect_crops(
    root: str | Path,
    n: int,
    split: str = "val",
    val_mod: int = 64,
    image_size: int | None = 112,
    list_file: str | Path | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return the first ``n`` crops of a split in source order: ``((n,3,H,W), ids)``.

    Deterministic; intended for fixed validation sets.
    """
    ds = open_source(root, list_file=list_file)
    want_val = split == "val"
    imgs, ids = [], []
    for i in range(len(ds)):
        ex = ds[i]
        sid = str(ex["subject_id"])
        if _is_val(sid, val_mod) != want_val:
            continue
        imgs.append(to_tensor01(ex["image"], size=image_size))
        ids.append(_subject_int(sid))
        if len(imgs) >= n:
            break
    if not imgs:
        side = image_size or 112
        return torch.empty(0, 3, side, side), np.empty(0, dtype=np.int64)
    return torch.stack(imgs), np.asarray(ids)
