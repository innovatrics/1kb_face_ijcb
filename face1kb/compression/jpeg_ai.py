"""Wrapper around the JPEG-AI reference software (ITU-T T.840.1 | ISO 6048).

The paper uses the official reference implementation with the High Operating
Point (HOP) profile. This module keeps one encoder and one decoder process
alive (model weights are loaded only once) and exposes:

- :func:`encode_at_bpp` - encode a PNG at a fixed ``--set_target_bpp``,
- :func:`find_max_bpp_within` - bisect the highest bpp whose bitstream fits
  a byte budget (the paper's recursive size search),
- :func:`decode` - decode a stored bitstream back to PNG.

The reference software must be checked out and configured separately; see
``docs/JPEG_AI.md``. Its location is taken from ``config.JPEGAI_REPO_DIR``
(environment variable ``JPEGAI_REPO_DIR``).
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

from face1kb import config

#: Encoder configuration used by the paper: all optional tools off,
#: High Operating Point profile (best quality).
ENCODER_CFG = ("cfg/tools_off.json", "cfg/profiles/high.json")

_encoder = None
_decoder = None


def available() -> bool:
    """Return True when the reference-software checkout is present."""
    return config.JPEGAI_REPO_DIR.is_dir()


@contextmanager
def _inside_repo():
    """Run with CWD and ``sys.path`` pointing into the reference software.

    The reference software resolves its configs and ``src.*`` imports
    relative to its repository root, so both must be adjusted temporarily.
    """
    repo = str(config.JPEGAI_REPO_DIR)
    prev_cwd = os.getcwd()
    os.chdir(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@contextmanager
def _quiet():
    """Silence stdout (the reference software prints copiously)."""
    orig = sys.stdout
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = orig


def _require_available() -> None:
    if not available():
        raise RuntimeError(
            f"JPEG-AI reference software not found at "
            f"{config.JPEGAI_REPO_DIR}. See docs/JPEG_AI.md and set the "
            f"JPEGAI_REPO_DIR environment variable."
        )


def _get_encoder():
    """Lazily create the persistent encoder process (models load once)."""
    global _encoder
    if _encoder is None:
        _require_available()
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        with _inside_repo():
            from src.reco.coders import RecoEncoderProcess

            with _quiet():
                _encoder = RecoEncoderProcess(None)
    return _encoder


def _get_decoder():
    """Lazily create the persistent decoder process (models load once)."""
    global _decoder
    if _decoder is None:
        _require_available()
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        with _inside_repo():
            from src.reco.coders import RecoDecoderProcess

            with _quiet():
                _decoder = RecoDecoderProcess(None)
    return _decoder


def encode_at_bpp(
    png_path: str | Path,
    bin_path: str | Path,
    target_bpp_x100: int,
    rec_path: str | Path | None = None,
) -> bool:
    """Encode one image at a fixed target bitrate.

    Parameters
    ----------
    png_path : str or Path
        Input PNG image.
    bin_path : str or Path
        Output bitstream path.
    target_bpp_x100 : int
        Target bits-per-pixel multiplied by 100 (16 means 0.16 bpp).
    rec_path : str or Path, optional
        When given, the encoder also writes the reconstruction here.

    Returns
    -------
    bool
        True when the bitstream was produced.
    """
    encoder = _get_encoder()
    cmd = [
        str(png_path),
        str(bin_path),
        "--set_target_bpp",
        str(target_bpp_x100),
        "--cfg",
        *ENCODER_CFG,
    ]
    if rec_path is not None:
        cmd.extend(["-r", str(rec_path)])
    try:
        with _inside_repo(), _quiet():
            encoder.process(cmd)
    except Exception:
        return False
    return Path(bin_path).exists()


def find_max_bpp_within(
    png_path: str | Path,
    bin_path: str | Path,
    max_size_bytes: int,
    bpp_range: tuple[int, int] = (10, 30),
) -> tuple[int, int] | None:
    """Bisect the highest target bpp whose bitstream fits the byte budget.

    Returns
    -------
    tuple of (int, int) or None
        ``(best_bpp_x100, file_size)``, or None when even the lowest
        candidate exceeds the budget.
    """
    lo, hi = bpp_range
    best = None

    while lo <= hi:
        mid = (lo + hi) // 2
        Path(bin_path).unlink(missing_ok=True)

        if not encode_at_bpp(png_path, bin_path, mid):
            hi = mid - 2
            continue

        size = Path(bin_path).stat().st_size
        if size <= max_size_bytes:
            best = (mid, size)
            lo = mid + 2
        else:
            hi = mid - 2

    return best


def decode(bin_path: str | Path, png_path: str | Path) -> None:
    """Decode a stored JPEG-AI bitstream to a PNG file."""
    decoder = _get_decoder()
    with _inside_repo(), _quiet():
        decoder.process([str(bin_path), str(png_path)])
