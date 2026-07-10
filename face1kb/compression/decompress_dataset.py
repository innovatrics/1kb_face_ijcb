"""Decode all compressed bitstreams to PNG for the downstream stages.

Embedding extraction and image-quality measurement both operate on decoded
pixels. Decoding once into ``data/decompressed_<resolution>/<codec>/`` avoids
running the expensive JPEG-FzT and JPEG-AI decoders repeatedly.

Usage
-----
    python -m face1kb.compression.decompress_dataset --resolution 112
"""

import argparse

from PIL import Image
from tqdm import tqdm

from face1kb import config
from face1kb.compression.decode import decode_to_rgb


def decompress_codec(
    resolution: int, codec: str, skip_existing: bool
) -> tuple[int, int]:
    """Decode all bitstreams of one codec at one resolution to PNGs."""
    source_dir = config.compressed_dir(resolution) / codec
    out_base = config.decompressed_dir(resolution) / codec
    if not source_dir.is_dir():
        print(f"skipping {codec}@{resolution}: {source_dir} does not exist")
        return 0, 0

    n_done = 0
    n_failed = 0
    files = sorted(p for p in source_dir.rglob("*") if p.is_file())
    for bitstream in tqdm(files, desc=f"{codec}@{resolution}"):
        rel_path = bitstream.relative_to(source_dir)
        out_path = out_base / rel_path.with_suffix(".png")
        if skip_existing and out_path.exists():
            n_done += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            rgb = decode_to_rgb(bitstream, resolution)
            Image.fromarray(rgb).save(out_path, "PNG")
            n_done += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            n_failed += 1
            tqdm.write(f"FAILED {codec} {rel_path}: {exc}")

    return n_done, n_failed


def main() -> None:
    """Run the requested codec x resolution decompression jobs."""
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
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    for codec in args.codecs:
        n_done, n_failed = decompress_codec(
            args.resolution, codec, args.skip_existing
        )
        print(f"{codec}@{args.resolution}: {n_done} decoded, "
              f"{n_failed} failed")


if __name__ == "__main__":
    main()
