# SPDX-License-Identifier: MIT
"""Training recipe of the Li-AE proxy auto-encoder.

The proxy (:class:`face1kb.adversarial.proxy.ProxyAE`) is trained on the attacker's
own face crops (the threat model assumes identities disjoint from the evaluation
sets; for the released proxy this was not checked, see ``docs/adversarial.md``) with a
*prototypical* objective in the spirit of Li et al. 2020: the decoder reconstructs
the **mean crop of the input's identity** (its prototype) instead of the input, so
the encoder has to learn identity-related features from few images.

Recipe of the released proxy (the defaults below):

* data: 400 identities with up to 12 crops each (identities with fewer than 4
  crops are skipped), 112 x 112 RGB crops; the released proxy saw 3,854 crops of a
  small WebFace42M sample, used for research purposes only;
* target: per-identity mean crop, loss: MSE between the sigmoid reconstruction and
  the target;
* optimisation: 60 epochs, batch 128 over a seeded permutation per epoch, Adam
  (lr 2e-3) with cosine annealing over the epochs, seed 0, the whole training set
  kept on the device.

:func:`select_identities` implements the identity selection on any stream of
``(subject, image)`` records, :func:`iter_image_folder` provides such a stream for a
folder-per-identity dataset (the layout of the official WebFace42M release), and
:func:`train_proxy` runs the optimisation. The released proxy was selected from an
identity-sorted re-packing of WebFace42M whose record order cannot be recreated from
the official release, and GPU training is not bit-deterministic, so a retrained
proxy reproduces the recipe, not the released weights.

Command line::

    python -m face1kb.adversarial.train --images-root /path/to/webface42m \
        --out runs/liae_proxy.safetensors
"""

from __future__ import annotations

import argparse
import io
import logging
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nnf

from .proxy import DEFAULT_WIDTH, RES, ProxyAE, save_proxy

logger = logging.getLogger(__name__)

#: Identities of the released proxy.
N_IDS = 400
#: Maximum crops kept per identity.
PER_ID = 12
#: Identities with fewer crops are skipped.
MIN_PER_ID = 4
#: Maximum number of records scanned by the released run.
MAX_ROWS = 120_000
#: Training epochs.
EPOCHS = 60
#: Mini-batch size.
BATCH_SIZE = 128
#: Adam learning rate (cosine-annealed over the epochs).
LR = 2e-3
#: Seed of the weight initialisation and of the epoch permutations.
SEED = 0

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def decode_crop(image, res: int = RES) -> np.ndarray:
    """Return an image as a ``(res, res, 3)`` uint8 RGB array.

    ``image`` may be encoded bytes, a file path, a PIL image or an ``(H, W, 3)``
    uint8 array. Images of another size are resized with PIL's bilinear filter.
    """
    return _resize(_decode_raw(image), res)


def _decode_raw(image) -> np.ndarray:
    """Decode ``image`` to an ``(H, W, 3)`` uint8 RGB array without resizing."""
    from PIL import Image  # noqa: PLC0415

    if isinstance(image, (bytes, bytearray, memoryview)):
        return np.asarray(Image.open(io.BytesIO(bytes(image))).convert("RGB"), np.uint8)
    if isinstance(image, (str, Path)):
        return np.asarray(Image.open(image).convert("RGB"), np.uint8)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), np.uint8)
    a = np.asarray(image)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 array, got {a.shape}")
    return a


def _resize(a: np.ndarray, res: int) -> np.ndarray:
    """Resize an RGB array to ``res x res`` with PIL's bilinear filter (if needed)."""
    from PIL import Image  # noqa: PLC0415

    if a.shape[:2] == (res, res):
        return a
    return np.asarray(Image.fromarray(a).resize((res, res), Image.BILINEAR), np.uint8)


def select_identities(
    records: Iterable[tuple[Hashable, object]],
    n_ids: int = N_IDS,
    per_id: int = PER_ID,
    min_per_id: int = MIN_PER_ID,
    max_rows: int | None = MAX_ROWS,
    res: int = RES,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Gather whole identities from a stream of ``(subject, image)`` records.

    Records are read in order. A record whose subject already has ``per_id`` crops
    is skipped without decoding; an image that fails to decode is skipped. Reading
    stops as soon as ``n_ids`` subjects have ``per_id`` crops, or after
    ``max_rows`` records. The result keeps, in order of first appearance, the first
    ``n_ids`` subjects with at least ``min_per_id`` crops, which includes subjects
    with fewer than ``per_id`` crops that appeared before the stop.

    Parameters
    ----------
    records
        Iterable of ``(subject, image)``; ``image`` is anything
        :func:`decode_crop` accepts.
    n_ids, per_id, min_per_id
        Selection sizes (defaults: the released proxy, 400 / 12 / 4).
    max_rows
        Maximum number of records read (``None``: no limit).
    res
        Crop size (images of another size are resized).

    Returns
    -------
    images : np.ndarray
        ``(N, res, res, 3)`` uint8 crops, grouped by subject.
    labels : np.ndarray
        ``(N,)`` int64 class index ``0 .. n - 1`` of each crop.
    subjects : list
        The selected subject keys; ``subjects[label]`` is the subject of a crop.
    """
    by_id: dict[Hashable, list[np.ndarray]] = defaultdict(list)
    n_full = 0
    for row, (sid, image) in enumerate(records):
        if max_rows is not None and row >= max_rows:
            break
        if len(by_id[sid]) >= per_id:
            continue
        try:
            a = _decode_raw(image)
        except Exception:  # noqa: BLE001 - skip an undecodable image
            logger.debug("skipping an undecodable image of subject %s", sid)
            continue
        by_id[sid].append(_resize(a, res))
        if len(by_id[sid]) == per_id:
            n_full += 1
        if n_full >= n_ids:
            break
    subjects = [s for s, v in by_id.items() if len(v) >= min_per_id][:n_ids]
    images, labels = [], []
    for li, s in enumerate(subjects):
        images.extend(by_id[s])
        labels.extend([li] * len(by_id[s]))
    if not images:
        raise ValueError("no subject has at least min_per_id decodable images")
    logger.info("selected %d identities, %d crops", len(subjects), len(images))
    return np.stack(images), np.asarray(labels, np.int64), subjects


def iter_image_folder(
    root: str | Path, subjects: Iterable[str] | None = None
) -> Iterator[tuple[str, Path]]:
    """Yield ``(subject, image_path)`` for a folder-per-identity dataset.

    Subject folders and the image files inside them are visited in sorted order.

    Parameters
    ----------
    root
        Dataset root with one sub-folder per identity.
    subjects
        Optional list of subject folder names to visit (in the given order);
        default: all sub-folders of ``root``.
    """
    root = Path(root)
    if subjects is None:
        names = sorted(p.name for p in root.iterdir() if p.is_dir())
    else:
        names = list(subjects)
    for name in names:
        folder = root / name
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in _IMAGE_SUFFIXES:
                yield name, p


def prototypes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the per-class mean image ``(C, 3, H, W)`` of ``x`` with labels ``y``."""
    n_cls = int(y.max().item()) + 1
    proto = torch.zeros(n_cls, *x.shape[1:], device=x.device, dtype=x.dtype)
    for c in range(n_cls):
        proto[c] = x[y == c].mean(0)
    return proto


def train_proxy(
    images: np.ndarray,
    labels: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    seed: int = SEED,
    width: int = DEFAULT_WIDTH,
    device: str | torch.device | None = None,
    log_every: int = 5,
) -> tuple[ProxyAE, list[float]]:
    """Train a :class:`ProxyAE` with the prototypical objective.

    Parameters
    ----------
    images
        ``(N, 112, 112, 3)`` uint8 RGB crops.
    labels
        ``(N,)`` integer class indices ``0 .. C - 1``.
    epochs, batch_size, lr, seed, width
        Recipe (defaults: the released proxy).
    device
        Torch device (default CUDA if available). The whole set is moved to it.
    log_every
        Log the epoch loss every ``log_every`` epochs (and after the last one).

    Returns
    -------
    net : ProxyAE
        The trained proxy in eval mode.
    history : list of float
        Mean training loss (prototype MSE) of every epoch.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    x = torch.from_numpy(images.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)
    y = torch.from_numpy(np.asarray(labels, np.int64)).to(dev)
    tgt = prototypes(x, y)[y]

    # Seeded initialisation without touching the caller's global RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        net = ProxyAE(width=width)
    net = net.to(dev).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = x.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    history: list[float] = []
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen).to(dev)
        tot = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            recon, _ = net(x[idx])
            loss = nnf.mse_loss(recon, tgt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        history.append(tot / n)
        if ep % log_every == 0 or ep == epochs - 1:
            logger.info("epoch %3d  proto-mse=%.5f", ep, tot / n)
    net.eval()
    return net, history


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m face1kb.adversarial.train",
        description="Train the Li-AE proxy auto-encoder on held-out face crops "
        "(folder-per-identity dataset of aligned 112 px crops).",
    )
    ap.add_argument("--images-root", required=True, help="one sub-folder per identity")
    ap.add_argument(
        "--subjects",
        default=None,
        help="optional text file with one subject folder name per line (visit order)",
    )
    ap.add_argument("--out", required=True, help="output .safetensors file")
    ap.add_argument("--ids", type=int, default=N_IDS, help="identities to keep")
    ap.add_argument("--per-id", type=int, default=PER_ID, help="max crops per identity")
    ap.add_argument(
        "--min-per-id", type=int, default=MIN_PER_ID, help="skip smaller identities"
    )
    ap.add_argument(
        "--max-rows", type=int, default=MAX_ROWS, help="max images read (0: no limit)"
    )
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument(
        "--source", default="", help="free-text training-data note for the metadata"
    )
    return ap.parse_args(argv)


def main(argv=None) -> None:
    """Command-line entry point (see the module docstring)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse_args(argv)
    subjects = None
    if args.subjects:
        lines = Path(args.subjects).read_text().splitlines()
        subjects = [s.strip() for s in lines if s.strip()]
    images, labels, chosen = select_identities(
        iter_image_folder(args.images_root, subjects),
        n_ids=args.ids,
        per_id=args.per_id,
        min_per_id=args.min_per_id,
        max_rows=args.max_rows or None,
    )
    net, history = train_proxy(
        images,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        width=args.width,
        device=args.device,
    )
    meta = {
        "n_ids": len(chosen),
        "n_imgs": len(images),
        "epochs": args.epochs,
        "final_proto_mse": f"{history[-1]:.6f}",
    }
    if args.source:
        meta["source"] = args.source
    path = save_proxy(net, args.out, meta)
    logger.info("wrote %s (%d identities, %d crops)", path, len(chosen), len(images))


if __name__ == "__main__":
    main()
