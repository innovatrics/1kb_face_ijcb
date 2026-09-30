# SPDX-License-Identifier: MIT
r"""Command line: ``python -m face1kb.fr {list,verify,validate}``.

``list``
    The registered evaluators and whether their files are present.
``verify``
    Check the SHA-256 of the files of the given (default: all built-in) evaluators.
``validate``
    Embed three aligned crops -- two of one subject and one of another -- and print
    the self, mate and impostor cosine similarities::

        python -m face1kb.fr validate --model arcface_antelopev2 \
            --images a1.png a2.png b.png

Fetching is done by ``scripts/fetch_models.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from . import download as _fetch
from .registry import REGISTRY, ROSTER, available, load


def _read_crop(path: str) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415

    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if arr.shape[:2] != (112, 112):
        from .embedders import resize_batch  # noqa: PLC0415

        logging.getLogger(__name__).warning(
            "%s is %dx%d; resizing to 112 (bilinear)", path, arr.shape[1], arr.shape[0]
        )
        arr = resize_batch(arr[None], 112)[0]
    return arr


def _cmd_list(args) -> int:
    for name in available():
        spec = REGISTRY[name]
        state = (
            "user"
            if not spec.builtin
            else ("present" if _fetch.present(name) else "missing")
        )
        print(f"{name:20s} {spec.family:9s} {spec.label:16s} {state}")
    return 0


def _cmd_verify(args) -> int:
    names = args.models or list(ROSTER)
    unknown = [n for n in names if n not in ROSTER]
    if unknown:
        print(f"unknown built-in evaluator(s): {unknown}; choose from {list(ROSTER)}")
        return 2
    bad = 0
    for name in names:
        problems = _fetch.verify(name)
        print(f"{name:20s} {'OK' if not problems else 'FAILED'}")
        for p in problems:
            print(f"    {p}")
        bad += bool(problems)
    return 1 if bad else 0


def _cmd_validate(args) -> int:
    crops = [_read_crop(p) for p in args.images]
    emb = load(args.model, device=args.device, download=not args.offline).embed(crops)
    n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    self_cos = float(n[0] @ n[0])
    mate = float(n[0] @ n[1])
    impostor = float(n[0] @ n[2])
    ok = emb.shape[1] > 0 and np.isfinite(emb).all() and mate > impostor
    print(
        f"{args.model}: shape={emb.shape} | self-cos={self_cos:.3f} | "
        f"mate-cos={mate:.3f} | impostor-cos={impostor:.3f} | "
        f"{'OK' if ok else 'CHECK'}"
    )
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``python -m face1kb.fr``."""
    ap = argparse.ArgumentParser(
        prog="python -m face1kb.fr",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the registered evaluators")
    p = sub.add_parser("verify", help="check the SHA-256 of the model files")
    p.add_argument("models", nargs="*", metavar="MODEL", help="default: all 14")
    p = sub.add_parser("validate", help="self / mate / impostor cosine on 3 crops")
    p.add_argument("--model", required=True)
    p.add_argument(
        "--images",
        nargs=3,
        required=True,
        metavar=("A1", "A2", "B"),
        help="two crops of one subject, one of another (112x112 aligned)",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--offline", action="store_true", help="never download files")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return {"list": _cmd_list, "verify": _cmd_verify, "validate": _cmd_validate}[
        args.cmd
    ](args)


if __name__ == "__main__":
    sys.exit(main())
