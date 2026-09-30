# SPDX-License-Identifier: MIT
"""Check the baseline-codec installation, and scan stored JPEG-AI bitstreams.

Round trip (default)::

    python -m face1kb.baselines.verify [--res 112] [--budget 1024] [--device cuda]

encodes one synthetic ``res`` px image to ``budget`` bytes with every baseline,
decodes it and checks the decoded shape. A codec whose backend is not installed is
reported as ``missing``; the exit status is non-zero only if an installed codec
fails (or, with ``--strict``, if any codec is missing).

Truncated-stream scan::

    python -m face1kb.baselines.verify --scan-jpegai WORK_ROOT/colorferet/compressed

lists ``.jpegai`` files smaller than ``MIN_PLAUSIBLE_FRAC`` (25 %) of the budget of
their cell (``<res>[<suffix>]px_<budget>B``), which indicates a truncated write.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_CELL = re.compile(r"px_(\d+)B$")


def synthetic_image(res: int = 112, seed: int = 0) -> np.ndarray:
    """Deterministic smooth test image with mild noise (``res x res x 3`` uint8)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32) / max(res - 1, 1)
    r = 0.5 + 0.4 * np.sin(2 * np.pi * xx)
    g = 0.5 + 0.4 * np.cos(2 * np.pi * yy)
    b = 1.0 - ((xx - 0.5) ** 2 + (yy - 0.5) ** 2) * 2
    img = np.stack([r, g, b], axis=-1) * 255 + rng.normal(0, 4, (res, res, 3))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def _backend_missing(codec: str, device: str) -> str | None:
    """Explain why ``codec`` cannot run here, or None if its backend is present."""
    from . import registry  # noqa: PLC0415

    entry = registry.get_codec(codec)
    if entry.family == "classical":
        from PIL import Image  # noqa: PLC0415

        from . import classical  # noqa: PLC0415

        classical._ensure_plugins()
        Image.init()
        fmt = classical.PIL_FORMATS[codec]
        if fmt not in Image.SAVE:
            return f"Pillow cannot write {fmt} (install the 'codecs' extra)"
        return None
    if entry.family in ("compressai", "jpeg_ai"):
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return "torch is not installed"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            return "no CUDA device"
    if entry.family == "compressai":
        try:
            import compressai  # noqa: F401, PLC0415
        except ImportError:
            return "compressai is not installed"
    if entry.family == "jpeg_ai":
        from . import jpeg_ai  # noqa: PLC0415

        problems = jpeg_ai.check_setup()
        if problems:
            return "; ".join(problems)
    return None


def round_trip(codecs, res: int, budget: int, device: str) -> list[dict]:
    """Encode + decode the synthetic image with each codec; return report rows."""
    from . import registry  # noqa: PLC0415

    img = synthetic_image(res)
    rows = []
    for name in codecs:
        entry = registry.get_codec(name)
        row = {"codec": name, "status": "ok", "bytes": None, "fitted": None}
        why = _backend_missing(name, device)
        if why:
            row.update(status="missing", detail=why)
            rows.append(row)
            continue
        try:
            kw = {"device": device} if entry.family == "compressai" else {}
            data, info = entry.encode_to_budget(img, budget, **kw)
            out = entry.decode(data, res=res, device=device)
            if out.shape != img.shape or out.dtype != np.uint8:
                raise RuntimeError(f"decoded shape {out.shape} {out.dtype}")
            row.update(
                bytes=len(data),
                fitted=info["fitted"],
                setting=info["setting"],
                psnr=round(_psnr(img, out), 2),
            )
        except Exception as exc:  # noqa: BLE001 - report every failure
            row.update(status="FAIL", detail=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def scan_jpegai(root: str | Path, frac: float | None = None) -> list[tuple[Path, int]]:
    """``.jpegai`` files under ``root`` smaller than ``frac`` x their cell budget."""
    from .jpeg_ai import MIN_PLAUSIBLE_FRAC  # noqa: PLC0415

    frac = MIN_PLAUSIBLE_FRAC if frac is None else frac
    bad = []
    for p in sorted(Path(root).rglob("*.jpegai")):
        budget = next(
            (int(m.group(1)) for part in p.parts if (m := _CELL.search(part))), None
        )
        if budget is None:
            continue
        size = p.stat().st_size
        if size < frac * budget:
            bad.append((p, size))
    return bad


def main(argv=None) -> int:
    """Command-line entry point."""
    from . import registry  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--codecs", default=",".join(registry.CODECS))
    ap.add_argument("--res", type=int, default=112)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--strict", action="store_true", help="fail on missing codecs")
    ap.add_argument(
        "--scan-jpegai",
        metavar="DIR",
        help="list truncated .jpegai files under DIR instead of the round trip",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if args.scan_jpegai:
        bad = scan_jpegai(args.scan_jpegai)
        for p, size in bad:
            print(f"{size:6d} B  {p}")
        print(f"{len(bad)} implausibly small JPEG-AI bitstream(s)")
        return 1 if bad else 0

    codecs = [c for c in args.codecs.split(",") if c]
    rows = round_trip(codecs, args.res, args.budget, args.device)
    for r in rows:
        if r["status"] == "ok":
            print(
                f"{r['codec']:22s} ok       {r['bytes']:5d} B  "
                f"fitted={r['fitted']!s:5s}  setting={r['setting']}  "
                f"PSNR={r['psnr']} dB"
            )
        else:
            print(f"{r['codec']:22s} {r['status']:8s} {r.get('detail', '')}")
    failed = [r for r in rows if r["status"] == "FAIL"]
    missing = [r for r in rows if r["status"] == "missing"]
    if failed or (args.strict and missing):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
