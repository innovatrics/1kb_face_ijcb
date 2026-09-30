# SPDX-License-Identifier: MIT
r"""Craft the no-box adversarial crop sets of the compression-as-defence study.

For every ``eps`` of the sweep the clean 112 px aligned crops are perturbed with one
attack of :mod:`face1kb.adversarial` (``hfc``, ``clip`` or ``liae``) and written as an
ordinary aligned-crop variant::

    aligned_dir(dataset, 112, adv_suffix(attack, eps))/<subject>/<stem>.png
    # e.g. aligned_112_adv_hfc_006/00001/00001_930831_fa_a.png

The compression and embedding pipelines consume these folders like any other crop
set (``--align-suffix _adv_hfc_006`` / ``--suffixes _adv_hfc_006``).

The crops are taken in ``index.csv`` order, starting from the first crop; a crop whose
clean file is missing is skipped, so the attack batches are formed over the crops
that exist. ``--limit N`` keeps the first ``N`` index rows (the paper uses 2000 for the
CLIP attack on Color FERET and for every attack on AI-Solutions-KK). The defaults are
the paper settings: ``eps`` 0.03/0.06/0.10, seed 0, batches of 64, 10 I-FGSM steps
(and 10 ILA steps for Li-AE), the released Li-AE proxy, and
``--random-starts paper``, which recomputes the random starts of the paper crop sets
on any device (:func:`face1kb.adversarial.paper_cuda_blocks`). With
``--random-starts native`` the starts come from the generator of the attack device.

HFC runs on the CPU and reproduces the paper crops bit for bit. CLIP and Li-AE run
on a GPU; they reproduce the paper crops only up to run-to-run variation of the GPU
kernels (see ``docs/adversarial.md``).

Existing output files are kept unless ``--overwrite`` is given. With
``FACE1KB_LAYOUT=legacy`` an explicit ``--out-root`` is required.

Examples
--------
    python experiments/adversarial/craft.py --dataset colorferet --attack hfc
    python experiments/adversarial/craft.py --dataset colorferet --attack clip \
        --limit 2000 --device cuda
    python experiments/adversarial/craft.py --dataset kk --attack liae --limit 2000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from face1kb import adversarial as adv
from face1kb import config
from face1kb.data.cli_utils import output_dir_guard, parse_list, setup_logging

log = logging.getLogger("adversarial.craft")


def load_clean(dataset: str, limit: int = 0) -> tuple[list[str], np.ndarray]:
    """Clean 112 px crops in index order, skipping crops whose file is missing.

    Returns
    -------
    tuple[list[str], ndarray]
        The relative paths of the loaded crops and their ``(N, 112, 112, 3)`` uint8
        RGB pixels.
    """
    index = config.read_index(dataset).sort_values("id").reset_index(drop=True)
    rels = index["rel_path"].tolist()
    if limit:
        rels = rels[:limit]
    src = config.aligned_dir(dataset, adv.ATTACK_RES)
    kept, imgs = [], []
    for rel in rels:
        path = src / rel
        if path.exists():
            with Image.open(path) as im:
                imgs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
            kept.append(rel)
    if len(kept) < len(rels):
        log.warning(
            "%d of %d crops missing under %s", len(rels) - len(kept), len(rels), src
        )
    if not kept:
        raise SystemExit(f"no clean {adv.ATTACK_RES} px crops found under {src}")
    return kept, np.stack(imgs)


def attack_options(args: argparse.Namespace) -> dict:
    """Keyword arguments of :func:`face1kb.adversarial.craft` for ``args.attack``."""
    if args.attack == "hfc":
        return {"tile": args.tile, "suppress": args.suppress, "seed": args.seed}
    opts = {
        "device": args.device,
        "steps": args.steps,
        "seed": args.seed,
        "batch_size": args.batch_size,
    }
    if args.random_starts == "paper":
        opts["cuda_blocks"] = adv.paper_cuda_blocks(args.dataset, args.attack)
    if args.attack == "clip":
        opts["model"] = adv.load_clip(device=args.device)
    else:  # liae
        opts["ila_steps"] = args.ila_steps
        opts["proxy"] = adv.load_proxy(
            args.proxy or None, device=args.device or _cuda()
        )
    return opts


def _cuda() -> str:
    import torch  # noqa: PLC0415

    return "cuda" if torch.cuda.is_available() else "cpu"


def out_dir(dataset: str, suffix: str, out_root: str | None) -> Path:
    """Destination folder of one crop set."""
    folder = config.aligned_dir(dataset, adv.ATTACK_RES, suffix)
    return Path(out_root) / folder.name if out_root else folder


def write_crops(root: Path, rels: list[str], crops: np.ndarray, overwrite: bool) -> int:
    """Write ``crops[i]`` to ``root / rels[i]`` as PNG; return the number written."""
    n = 0
    for rel, img in zip(tqdm(rels, desc=root.name, unit="img"), crops):
        dst = root / rel
        if dst.exists() and not overwrite:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(dst)
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    """Craft one attack over the eps sweep."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=config.DATASETS)
    ap.add_argument("--attack", required=True, choices=sorted(adv.ATTACKS))
    ap.add_argument("--eps", default=",".join(str(e) for e in adv.DEFAULT_EPS))
    ap.add_argument("--limit", type=int, default=0, help="first N index rows (0: all)")
    ap.add_argument("--device", default=None, help="torch device of clip/liae")
    ap.add_argument("--random-starts", choices=("paper", "native"), default="paper")
    ap.add_argument("--steps", type=int, default=10, help="I-FGSM steps (clip, liae)")
    ap.add_argument("--ila-steps", type=int, default=10, help="ILA steps (liae)")
    ap.add_argument("--batch-size", type=int, default=64, help="clip, liae")
    ap.add_argument("--proxy", default="", help="Li-AE proxy (default: released)")
    ap.add_argument("--tile", type=int, default=4, help="HFC noise block size (px)")
    ap.add_argument("--suppress", type=float, default=0.5, help="HFC own-HF blend")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default=None, help="write the crop sets here")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()
    output_dir_guard(args.out_root is not None)

    epses = parse_list(args.eps, float)
    rels, clean = load_clean(args.dataset, args.limit)
    log.info(
        "%s: %s at eps %s on %d crops", args.dataset, args.attack, epses, len(rels)
    )
    opts = attack_options(args)
    for eps in epses:
        suffix = adv.adv_suffix(args.attack, eps)
        crops = adv.craft(args.attack, clean, eps, **opts)
        linf = int(np.abs(crops.astype(np.int16) - clean.astype(np.int16)).max())
        root = out_dir(args.dataset, suffix, args.out_root)
        n = write_crops(root, rels, crops, args.overwrite)
        log.info(
            "%s: wrote %d/%d crops (l-inf %d/255, budget %.3f) -> %s",
            suffix,
            n,
            len(rels),
            linf,
            eps,
            root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
