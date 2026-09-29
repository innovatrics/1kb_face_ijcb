"""Unified 1 kB target-size compression for all six evaluated codecs.

Every codec exposes the same interface::

    compress_<codec>(image, max_size_bytes, out_path) -> (size, quality)

The compressed bitstream is written to *out_path*. Since none of the codecs
accepts a target file size directly, the paper's recursive search is used:
start at maximum acceptable quality and decrease it until the bitstream fits
*max_size_bytes* (JPEG-AI uses bisection over the target bitrate instead).
When even the lowest setting does not fit, the smallest achievable bitstream
is kept - such images are retained in the evaluation (see Sec. 2.5 of the
paper).

Return value
------------
``size`` is the size in bytes of the bitstream left in *out_path* and
``quality`` is the codec setting that produced exactly that bitstream - also
when the budget is missed, in which case it is the last (most aggressive)
setting of the search:

- JPEG: Pillow quality 40, 38, ..., 2 (2 on a miss),
- WebP: Pillow quality 45, 43, ..., 1 (1 on a miss),
- JPEG2000: Pillow ``quality_layers`` compression ratio 80, 82, ..., 98
  (inverse scale, 98 on a miss),
- JPEG-XL: ``cjxl -q`` 60, 58, ..., 2; settings that ``cjxl`` rejects are
  skipped (libjxl 0.7 refuses values below 5), so on a miss this is the
  lowest accepted setting (6 with libjxl 0.7),
- JPEG-FzT: JPEG quality of the F-transform stage 51, 49, ..., 1 (1 on a
  miss),
- JPEG-AI: target bpp x 100, bisected within 10-30; on a miss the image is
  encoded at 1 (0.01 bpp) and 1 is returned.

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
    qualities = range(40, 0, -2)
    for quality in qualities:
        image.save(out_path, "JPEG", quality=quality, optimize=True)
        size = out_path.stat().st_size
        if size <= max_size_bytes:
            return size, quality
    return out_path.stat().st_size, qualities[-1]


def compress_webp(
    image: Image.Image, max_size_bytes: int, out_path: Path
) -> tuple[int, int]:
    """Compress with WebP, lowering quality until the size fits."""
    qualities = range(45, 0, -2)
    for quality in qualities:
        image.save(out_path, "WEBP", quality=quality, method=6)
        size = out_path.stat().st_size
        if size <= max_size_bytes:
            return size, quality
    return out_path.stat().st_size, qualities[-1]


def compress_jpeg2000(
    image: Image.Image, max_size_bytes: int, out_path: Path
) -> tuple[int, int]:
    """Compress with JPEG 2000, raising the quality layer until it fits.

    Pillow's ``quality_layers`` acts as a compression-ratio parameter, i.e.
    an inverse quality scale where higher values mean stronger compression.
    """
    qualities = range(80, 100, 2)
    for quality in qualities:
        image.save(out_path, "JPEG2000", quality_layers=[quality])
        size = out_path.stat().st_size
        if size <= max_size_bytes:
            return size, quality
    return out_path.stat().st_size, qualities[-1]


def compress_jpeg_xl(
    image: Image.Image, max_size_bytes: int, out_path: Path
) -> tuple[int, int]:
    """Compress with JPEG XL via ``cjxl``, lowering quality until it fits.

    A setting counts as tried only when ``cjxl`` succeeds, so the returned
    quality always matches the bitstream left in *out_path*.
    """
    if not CJXL_AVAILABLE:
        raise RuntimeError("cjxl not found; install libjxl tools")

    last_quality = None
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        image.save(tmp.name, "PNG")
        for quality in range(60, 0, -2):
            result = subprocess.run(
                ["cjxl", tmp.name, str(out_path), "-q", str(quality)],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0 or not out_path.exists():
                continue
            last_quality = quality
            size = out_path.stat().st_size
            if size <= max_size_bytes:
                return size, quality

    if last_quality is None:
        raise RuntimeError("JPEG XL compression produced no output")
    return out_path.stat().st_size, last_quality


def compress_jpeg_fzt(
    image: Image.Image, max_size_bytes: int, out_path: Path
) -> tuple[int, int]:
    """Compress with JPEG-FzT, lowering the JPEG quality until it fits.

    The stored ``.fzt`` bitstream is the JPEG-encoded low-resolution
    F-transform stage; the F-transform itself is quality-independent and is
    therefore computed only once.
    """
    stage, _ = jpeg_fzt.compress_with_quality(image, quality=51)
    qualities = range(51, 0, -2)
    for quality in qualities:
        stage.save(out_path, "JPEG", quality=quality, optimize=True)
        size = out_path.stat().st_size
        if size <= max_size_bytes:
            return size, quality
    return out_path.stat().st_size, qualities[-1]


def compress_jpeg_ai(
    image: Image.Image, max_size_bytes: int, out_path: Path
) -> tuple[int, int]:
    """Compress with JPEG-AI, bisecting the target bitrate until it fits.

    Uses the reference-software encoder in the High Operating Point profile;
    the ``--set_target_bpp`` parameter (bpp x 100) is searched via bisection
    for the highest bitrate whose bitstream fits *max_size_bytes*. When even
    the lowest searched bitrate does not fit, the image is encoded at 1
    (0.01 bpp) and 1 is returned.
    """
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        image.save(tmp.name, "PNG")

        result = jpeg_ai.find_max_bpp_within(tmp.name, out_path, max_size_bytes)
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
