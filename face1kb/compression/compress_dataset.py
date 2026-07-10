"""Compress every aligned image with the selected codecs at 1 kB budget.

Reads the aligned originals from ``data/aligned_<resolution>/`` and writes
one bitstream per image and codec to::

    data/compressed_<resolution>/<codec>/<identity_dir>/<image><ext>

Usage
-----
    python -m face1kb.compression.compress_dataset --resolution 112
    python -m face1kb.compression.compress_dataset --resolution 224 \\
        --codecs jpeg webp
"""

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from face1kb import config
from face1kb.compression import jpeg_ai
from face1kb.compression.codecs import CJXL_AVAILABLE, COMPRESSORS


def codec_available(codec: str) -> bool:
    """Return False for codecs whose external tooling is missing."""
    if codec == "jpeg_xl" and not CJXL_AVAILABLE:
        print("skipping jpeg_xl: cjxl not installed (see setup_env.sh)")
        return False
    if codec == "jpeg_ai" and not jpeg_ai.available():
        print(
            "skipping jpeg_ai: reference software not found "
            "(see docs/JPEG_AI.md)"
        )
        return False
    return True


def iter_images(base_dir: Path):
    """Yield ``(absolute_path, relative_path)`` for all PNGs under a dir."""
    for path in sorted(base_dir.rglob("*.png")):
        yield path, path.relative_to(base_dir)


def compress_codec(
    resolution: int, codec: str, skip_existing: bool
) -> tuple[int, int]:
    """Compress all images at *resolution* with *codec*.

    Returns
    -------
    tuple of (int, int)
        ``(n_compressed, n_failed)``.
    """
    source_dir = config.aligned_dir(resolution)
    out_base = config.compressed_dir(resolution) / codec
    extension = config.CODEC_EXTENSIONS[codec]
    compress = COMPRESSORS[codec]

    n_done = 0
    n_failed = 0
    images = list(iter_images(source_dir))
    for img_path, rel_path in tqdm(images, desc=f"{codec}@{resolution}"):
        out_path = out_base / rel_path.with_suffix(extension)
        if skip_existing and out_path.exists():
            n_done += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(img_path) as image:
                compress(image, config.MAX_SIZE_BYTES, out_path)
            n_done += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            n_failed += 1
            tqdm.write(f"FAILED {codec} {rel_path}: {exc}")

    return n_done, n_failed


def main() -> None:
    """Run the requested codec x resolution compression jobs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        choices=config.RESOLUTIONS,
    )
    parser.add_argument(
        "--codecs",
        nargs="+",
        default=list(config.CODECS),
        choices=config.CODECS,
        help="Subset of codecs to run (default: all six).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose output bitstream already exists.",
    )
    args = parser.parse_args()

    for codec in args.codecs:
        if not codec_available(codec):
            continue
        n_done, n_failed = compress_codec(
            args.resolution, codec, args.skip_existing
        )
        print(f"{codec}@{args.resolution}: {n_done} compressed, "
              f"{n_failed} failed")


if __name__ == "__main__":
    main()
