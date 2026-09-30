# SPDX-License-Identifier: MIT
"""Command-line interface: ``python -m face1kb.codec {encode,decode,info}``.

Examples
--------
Encode an aligned 112 px crop to at most 1024 bytes and decode it again::

    python -m face1kb.codec encode face.png face.f1k --variant accurate --budget 1024
    python -m face1kb.codec info face.f1k
    python -m face1kb.codec decode face.f1k face_decoded.png

The ``.f1k`` file is the raw container (:mod:`face1kb.codec.container`); the variant
is read from its header at decode time.

Default output paths: ``encode face.png`` writes ``face.f1k`` and ``decode face.f1k``
writes ``face_decoded.png``, so an encode / decode round trip never replaces the
source crop. Neither command overwrites its own input file. The output path may
also follow the options (``encode face.png --budget 512 out.f1k``).

``encode`` prints a warning on stderr whenever the written container is larger than
the budget (``--overflow floor``, or ``--paper-compat`` at resolutions without a
bucket code such as 96 and 168 px, where the paper accounting ignores the 4-byte
geometry trailer) and when it is the identity-only fallback (no spatial stream fits;
it decodes to a black frame). The JSON record's ``fitted`` means that a spatial
stream fit the search target; ``within_budget`` states whether the file holds the
budget.

``decode`` reports the model loading time (``load_ms``) separately from the decoding
time (``decode_ms``). Both are single cold calls (CUDA context creation, entropy-coder
table construction), so ``decode_ms`` is far above the warm per-crop decode time of a
loaded :class:`~face1kb.codec.api.Codec`; identity-only containers are decoded
without a model or torch.

Missing or unreadable input files, unwritable outputs, an output path equal to the
input, malformed containers and impossible budgets end with an ``error:`` line on
stderr and exit code 1 (``-v`` shows the traceback).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path


def _read_rgb(path: Path):
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _check_output(out: Path, src: Path) -> None:
    if out.resolve() == src.resolve():
        raise ValueError(f"refusing to overwrite the input file {src}")


def _cmd_encode(args) -> int:
    import face1kb  # noqa: PLC0415
    from face1kb.codec.budget import IdentityOnlyWarning  # noqa: PLC0415

    out = args.output or args.input.with_suffix(".f1k")
    _check_output(out, args.input)
    img = _read_rgb(args.input)
    codec = face1kb.load(args.variant, device=args.device, weights_dir=args.weights_dir)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        # Reported below with a CLI-specific hint instead.
        warnings.simplefilter("ignore", IdentityOnlyWarning)
        data, info = codec.encode(
            img,
            budget=args.budget,
            paper_compat=args.paper_compat,
            return_info=True,
            overflow=args.overflow,
        )
    ms = (time.perf_counter() - t0) * 1000.0
    out.write_bytes(data)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(out),
                "variant": args.variant,
                "res": int(img.shape[0]),
                "budget": args.budget,
                **{k: v for k, v in info.items()},
                "encode_ms": round(ms, 1),
            }
        )
    )
    if len(data) > args.budget:
        if info["over_budget"]:
            why = "the lowest-gain stream does not fit"
        elif info["fitted"]:
            why = "the paper accounting does not reserve the 4-byte geometry trailer"
        else:
            why = "the budget is below the identity-only container size"
        print(
            f"warning: {out} ({len(data)} B) exceeds the {args.budget} B budget: {why}",
            file=sys.stderr,
        )
    if info["identity_only"]:
        print(
            f"warning: no spatial stream fits {args.budget} B at {img.shape[0]} px; "
            f"{out} is the identity-only container ({len(data)} B), which decodes to "
            "a black frame. Use a larger --budget or --overflow floor.",
            file=sys.stderr,
        )
    return 0


def _cmd_decode(args) -> int:
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    import face1kb  # noqa: PLC0415
    from face1kb.codec import container  # noqa: PLC0415

    out = args.output or args.input.with_name(args.input.stem + "_decoded.png")
    _check_output(out, args.input)
    data = args.input.read_bytes()
    h = container.unpack(data)
    if h.variant_id == container.VARIANT_IDENTITY_ONLY:
        # Black frame of the stored size; no model (and no torch) needed.
        load_ms = 0.0
        t0 = time.perf_counter()
        img = np.zeros((h.res, h.res, 3), dtype=np.uint8)
    elif h.variant_id in (container.VARIANT_FAST, container.VARIANT_ACCURATE):
        t0 = time.perf_counter()
        codec = face1kb.load(
            container.VARIANT_NAMES[h.variant_id],
            device=args.device,
            weights_dir=args.weights_dir,
        )
        load_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        img = codec.decode(data)  # returns a host array, so the GPU work is done
    else:
        raise ValueError(f"unsupported container variant id {h.variant_id}")
    ms = (time.perf_counter() - t0) * 1000.0
    Image.fromarray(img).save(out)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(out),
                "variant": container.VARIANT_NAMES.get(h.variant_id, h.variant_id),
                "load_ms": round(load_ms, 1),
                "decode_ms": round(ms, 1),
            }
        )
    )
    return 0


def _cmd_info(args) -> int:
    from face1kb.codec.container import describe  # noqa: PLC0415

    for p in args.inputs:
        print(json.dumps({"file": str(p), **describe(p.read_bytes())}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the CLI."""
    ap = argparse.ArgumentParser(
        prog="python -m face1kb.codec",
        description="Encode / decode aligned face crops with the face1kb codecs.",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="log to stderr")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--device", default="cuda", help="torch device (default cuda)")
        p.add_argument(
            "--weights-dir",
            type=Path,
            default=None,
            help="folder with face1kb_*.safetensors (default FACE1KB_WEIGHTS_DIR)",
        )

    enc = sub.add_parser("encode", help="aligned RGB crop -> container file")
    enc.add_argument("input", type=Path, help="square aligned crop (PNG, ...)")
    enc.add_argument("output", type=Path, nargs="?", help="default <input>.f1k")
    enc.add_argument("--variant", choices=("fast", "accurate"), default="fast")
    enc.add_argument("--budget", type=int, default=1024, help="bytes (default 1024)")
    enc.add_argument(
        "--paper-compat",
        action="store_true",
        help="paper budget accounting (reproduces the paper bitstreams)",
    )
    enc.add_argument(
        "--overflow",
        choices=("identity", "floor", "error"),
        default=None,
        help="behaviour when nothing fits (default: identity, floor with "
        "--paper-compat)",
    )
    common(enc)
    enc.set_defaults(func=_cmd_encode)

    dec = sub.add_parser("decode", help="container file -> PNG")
    dec.add_argument("input", type=Path)
    dec.add_argument(
        "output", type=Path, nargs="?", help="default <input stem>_decoded.png"
    )
    common(dec)
    dec.set_defaults(func=_cmd_decode)

    inf = sub.add_parser("info", help="print container header fields")
    inf.add_argument("inputs", type=Path, nargs="+")
    inf.set_defaults(func=_cmd_info)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    ap = build_parser()
    args, extra = ap.parse_known_args(argv)
    # argparse binds the optional ``output`` positional (empty) together with
    # ``input``, so an output path given after the options arrives as an extra.
    if (
        len(extra) == 1
        and not extra[0].startswith("-")
        and getattr(args, "output", False) is None
    ):
        args.output = Path(extra[0])
        extra = []
    if extra:
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return int(args.func(args))
    except Exception as e:
        # Missing / unreadable files (OSError, which includes PIL's
        # UnidentifiedImageError), malformed containers, impossible budgets and
        # overflow="error" are reported without a traceback (unless --verbose). The
        # budget module is looked up lazily so that `info` does not import torch.
        budget_mod = sys.modules.get("face1kb.codec.budget")
        handled = (ValueError, OSError) + (
            (budget_mod.BudgetError,) if budget_mod else ()
        )
        if args.verbose or not isinstance(e, handled):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
