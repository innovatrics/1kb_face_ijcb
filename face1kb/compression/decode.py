"""Decode any stored bitstream back to an RGB pixel array.

Dispatch is based on the file extension used by the compression stage:

- ``.fzt``   - JPEG-FzT (inverse F-transform upsampling),
- ``.jxl``   - JPEG XL (Pillow plugin when available, ``djxl`` otherwise),
- ``.jpegai``- JPEG-AI reference-software decoder,
- everything else (``.jpg``, ``.jp2``, ``.webp``, ``.png``) via Pillow.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from face1kb.compression import jpeg_ai, jpeg_fzt

DJXL_AVAILABLE = shutil.which("djxl") is not None


def _decode_pillow(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _decode_jxl(path: Path) -> np.ndarray:
    try:
        return _decode_pillow(path)
    except Exception:
        if not DJXL_AVAILABLE:
            raise RuntimeError("djxl not found; install libjxl tools")
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            subprocess.run(
                ["djxl", str(path), tmp.name], capture_output=True, check=True
            )
            return _decode_pillow(Path(tmp.name))


def _decode_jpegai(path: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        jpeg_ai.decode(path, tmp.name)
        return _decode_pillow(Path(tmp.name))


def decode_to_rgb(path: str | Path, resolution: int) -> np.ndarray:
    """Decode a stored bitstream to a uint8 RGB array.

    Parameters
    ----------
    path : str or Path
        Bitstream produced by the compression stage.
    resolution : int
        Full resolution of the encoded image; required by JPEG-FzT, whose
        bitstream stores only the low-resolution F-transform stage.

    Returns
    -------
    numpy.ndarray
        Decoded image, uint8 ``(H, W, 3)``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".fzt":
        return jpeg_fzt.decompress(path, (resolution, resolution))
    if suffix == ".jxl":
        return _decode_jxl(path)
    if suffix == ".jpegai":
        return _decode_jpegai(path)
    return _decode_pillow(path)
