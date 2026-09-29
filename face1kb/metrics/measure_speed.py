"""Compression and decompression speed per codec (Tables 1 and 2).

Times single-image compression and decompression for every codec on a
sample of aligned images, sweeping the codec quality parameter as in the
paper, and reports mean +- half-range (``(max - min) / 2``) in ms/image::

    outputs/metrics/speed_<resolution>.csv

Pillow-based codecs and JPEG-FzT run on CPU; JPEG-XL is measured through
the ``cjxl``/``djxl`` command-line tools (process-spawn overhead included);
JPEG-AI uses the reference software and runs per-image on the GPU without
model caching between measurements amortised (models are loaded once).

Usage
-----
    python -m face1kb.metrics.measure_speed --resolution 112 --n-images 20
"""

import argparse
import csv
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image

from face1kb import config
from face1kb.compression import jpeg_ai, jpeg_fzt
from face1kb.compression.codecs import CJXL_AVAILABLE
from face1kb.compression.decode import decode_to_rgb

#: Quality parameters swept per codec (JPEG-AI sweeps target bpp x 100).
QUALITY_SWEEPS = {
    "jpeg": range(1, 101, 10),
    "jpeg2000": range(1, 101, 10),
    "webp": range(1, 101, 10),
    "jpeg_xl": range(1, 101, 10),
    "jpeg_fzt": range(1, 101, 10),
    "jpeg_ai": range(10, 31, 4),
}


def _timed(func) -> float:
    """Run *func* and return the elapsed wall-clock time in ms."""
    start = time.perf_counter()
    func()
    return (time.perf_counter() - start) * 1000.0


def measure_codec(
    codec: str, image_paths: list[Path], resolution: int
) -> tuple[list[float], list[float]]:
    """Return per-run compression and decompression times for one codec."""
    extension = config.CODEC_EXTENSIONS[codec]
    compress_times = []
    decompress_times = []

    for image_path in image_paths:
        with Image.open(image_path) as image:
            image.load()
            for quality in QUALITY_SWEEPS[codec]:
                with tempfile.NamedTemporaryFile(suffix=extension) as tmp:
                    out = Path(tmp.name)
                    if codec == "jpeg":
                        compress_times.append(_timed(
                            lambda: image.save(
                                out, "JPEG", quality=quality, optimize=True
                            )
                        ))
                    elif codec == "jpeg2000":
                        compress_times.append(_timed(
                            lambda: image.save(
                                out, "JPEG2000", quality_layers=[quality]
                            )
                        ))
                    elif codec == "webp":
                        compress_times.append(_timed(
                            lambda: image.save(
                                out, "WEBP", quality=quality, method=6
                            )
                        ))
                    elif codec == "jpeg_xl":
                        with tempfile.NamedTemporaryFile(suffix=".png") as png:
                            image.save(png.name, "PNG")
                            # cjxl rejects -q below 5 (libjxl 0.7).
                            jxl_quality = str(max(5, quality))
                            compress_times.append(_timed(
                                lambda: subprocess.run(
                                    ["cjxl", png.name, str(out),
                                     "-q", jxl_quality],
                                    capture_output=True,
                                    check=True,
                                )
                            ))
                    elif codec == "jpeg_fzt":
                        compress_times.append(_timed(
                            lambda: jpeg_fzt.compress_with_quality(
                                image, quality
                            )[0].save(
                                out, "JPEG", quality=quality, optimize=True
                            )
                        ))
                    elif codec == "jpeg_ai":
                        with tempfile.NamedTemporaryFile(suffix=".png") as png:
                            image.save(png.name, "PNG")
                            compress_times.append(_timed(
                                lambda: jpeg_ai.encode_at_bpp(
                                    png.name, out, quality
                                )
                            ))

                    if out.exists() and out.stat().st_size > 0:
                        decompress_times.append(_timed(
                            lambda: decode_to_rgb(out, resolution)
                        ))

    return compress_times, decompress_times


def main() -> None:
    """Measure all codecs at one resolution and write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        choices=config.RESOLUTIONS,
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=20,
        help="Number of sampled images (default 20).",
    )
    parser.add_argument(
        "--codecs",
        nargs="+",
        default=list(config.CODECS),
        choices=config.CODECS,
    )
    args = parser.parse_args()

    image_paths = sorted(
        config.aligned_dir(args.resolution).rglob("*.png")
    )[: args.n_images]
    if not image_paths:
        raise SystemExit("No aligned images found; run the data stage first.")

    out_dir = config.metrics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"speed_{args.resolution}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "resolution",
                "codec",
                "n_runs",
                "compress_mean_ms",
                "compress_halfrange_ms",
                "decompress_mean_ms",
                "decompress_halfrange_ms",
            ]
        )
        for codec in args.codecs:
            if codec == "jpeg_xl" and not CJXL_AVAILABLE:
                print("skipping jpeg_xl: cjxl/djxl not installed")
                continue
            if codec == "jpeg_ai" and not jpeg_ai.available():
                print("skipping jpeg_ai: reference software not found")
                continue
            comp, decomp = measure_codec(codec, image_paths, args.resolution)
            row = [args.resolution, codec, len(comp)]
            for values in (comp, decomp):
                mean = sum(values) / len(values)
                halfrange = (max(values) - min(values)) / 2.0
                row.extend([f"{mean:.2f}", f"{halfrange:.2f}"])
            writer.writerow(row)
            print(f"{codec}: compress {row[3]} ms, decompress {row[5]} ms")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
