"""Unified 1 kB target-size compression for all six evaluated codecs.

Every codec exposes the same interface::

    compress_<codec>(image, max_size_bytes, out_path) -> (size, quality)

The compressed bitstream is written to *out_path* and its final size plus the
quality parameter that produced it are returned. Since none of the codecs
accepts a target file size directly, the paper's recursive search is used:
start at maximum acceptable quality and decrease it until the bitstream fits
*max_size_bytes* (JPEG-AI uses bisection over the target bitrate instead).
When even the lowest setting does not fit, the smallest achievable bitstream
is kept - such images are retained in the evaluation (see Sec. 2.5 of the
paper).

JPEG, JPEG2000 and WebP are produced with Pillow; JPEG-XL with the ``cjxl``
command-line tool (libjxl); JPEG-FzT with our implementation in
:mod:`face1kb.compression.jpeg_fzt`; JPEG-AI with the reference software
(see ``docs/JPEG_AI.md``).
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from face1kb.compression import jpeg_ai, jpeg_fzt

CJXL_AVAILABLE = shutil.which("cjxl") is not None


def compress_jpeg(
    image: Image.Image, max_size_bytes: int, out_path: Path
) -> tuple[int, int]:
    """Compress with standard JPEG, lowering quality until the size fits."""
    quality = 40
    while quality > 0:
        image.save(out_path, "JPEG", quality=quality, optimize=True)
        size = out_path.stat().st_size
        if size <= max_size_bytes:
            return size, quality
        quality -= 2
    return out_path.stat().st_size, quality


def compress_webp(
    image: Image.Image, max_size_bytes: int, out_path: Path
) -> tuple[int, int]:
    """Compress with WebP, lowering quality until the size fits."""
    quality = 45
    while quality > 0:
        image.save(out_path, "WEBP", quality=quality, method=6)
        size = out_path.stat().st_size
        if size <= max_size_bytes:
            return size, quality
        quality -= 2
    return out_path.stat().st_size, quality


def compress_jpeg2000(
    image: Image.Image, max_size_bytes: int, out_path: Path
) -> tuple[int, int]:
    """Compress with JPEG 2000, raising the quality layer until it fits.

    Pillow's ``quality_layers`` acts as a compression-ratio parameter, i.e.
    an inverse quality scale where higher values mean stronger compression.
    """
    quality = 80
    while quality < 100:
        image.save(out_path, "JPEG2000", quality_layers=[quality])
        size = out_path.stat().st_size
        if size <= max_size_bytes:
            return size, quality
        quality += 2
    return out_path.stat().st_size, quality


def compress_jpeg_xl(
    image: Image.Image, max_size_bytes: int, out_path: Path
) -> tuple[int, int]:
    """Compress with JPEG XL via ``cjxl``, lowering quality until it fits."""
    if not CJXL_AVAILABLE:
        raise RuntimeError("cjxl not found; install libjxl tools")

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        image.save(tmp.name, "PNG")
        quality = 60
        while quality > 0:
            subprocess.run(
                ["cjxl", tmp.name, str(out_path), "-q", str(quality)],
                capture_output=True,
                check=False,
            )
            if out_path.exists() and out_path.stat().st_size <= max_size_bytes:
                return out_path.stat().st_size, quality
            quality -= 2

    if not out_path.exists():
        raise RuntimeError("JPEG XL compression produced no output")
    return out_path.stat().st_size, quality


def compress_jpeg_fzt(
    image: Image.Image, max_size_bytes: int, out_path: Path
) -> tuple[int, int]:
    """Compress with JPEG-FzT, lowering the JPEG quality until it fits.

    The stored ``.fzt`` bitstream is the JPEG-encoded low-resolution
    F-transform stage; the F-transform itself is quality-independent and is
    therefore computed only once.
    """
    stage, _ = jpeg_fzt.compress_with_quality(image, quality=51)
    for quality in range(51, 0, -2):
        stage.save(out_path, "JPEG", quality=quality, optimize=True)
        size = out_path.stat().st_size
        if size <= max_size_bytes:
            return size, quality
    return out_path.stat().st_size, 1


def compress_jpeg_ai(
    image: Image.Image, max_size_bytes: int, out_path: Path
) -> tuple[int, int]:
    """Compress with JPEG-AI, bisecting the target bitrate until it fits.

    Uses the reference-software encoder in the High Operating Point profile;
    the ``--set_target_bpp`` parameter (bpp x 100) is searched via bisection
    for the highest bitrate whose bitstream fits *max_size_bytes*.
    """
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        image.save(tmp.name, "PNG")

        result = jpeg_ai.find_max_bpp_within(
            tmp.name, out_path, max_size_bytes
        )
        if result is not None:
            best_bpp, _ = result
            out_path.unlink(missing_ok=True)
            jpeg_ai.encode_at_bpp(tmp.name, out_path, best_bpp)
            return out_path.stat().st_size, best_bpp

        # Even the lowest searched bitrate did not fit: keep the smallest
        # achievable bitstream.
        out_path.unlink(missing_ok=True)
        if not jpeg_ai.encode_at_bpp(tmp.name, out_path, 1):
            raise RuntimeError("JPEG-AI compression produced no output")
        return out_path.stat().st_size, 1


#: Codec name -> compression function with the unified interface.
COMPRESSORS = {
    "jpeg": compress_jpeg,
    "jpeg2000": compress_jpeg2000,
    "jpeg_xl": compress_jpeg_xl,
    "webp": compress_webp,
    "jpeg_fzt": compress_jpeg_fzt,
    "jpeg_ai": compress_jpeg_ai,
}
