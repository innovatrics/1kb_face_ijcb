"""File-size statistics and 1 kB compliance of the compressed bitstreams.

For every codec at the given resolution the script reports mean/median/
min/max bitstream size and the share of images that fit the 1024 B budget
(the blue percentages of Figures 4 and 5 in the paper)::

    outputs/metrics/file_sizes_<resolution>.csv

Usage
-----
    python -m face1kb.metrics.collect_file_sizes --resolution 112
"""

import argparse
import csv

import numpy as np

from face1kb import config


def codec_sizes(resolution: int, codec: str) -> np.ndarray:
    """Bitstream sizes (bytes) of all images of one codec."""
    codec_dir = config.compressed_dir(resolution) / codec
    if not codec_dir.is_dir():
        return np.array([])
    return np.array(
        [p.stat().st_size for p in sorted(codec_dir.rglob("*")) if p.is_file()]
    )


def main() -> None:
    """Collect size statistics for all codecs at one resolution."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        choices=config.RESOLUTIONS,
    )
    args = parser.parse_args()

    out_dir = config.metrics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"file_sizes_{args.resolution}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "resolution",
                "codec",
                "n_images",
                "mean_bytes",
                "median_bytes",
                "min_bytes",
                "max_bytes",
                "compliant_pct",
            ]
        )
        for codec in config.CODECS:
            sizes = codec_sizes(args.resolution, codec)
            if sizes.size == 0:
                print(f"skipping {codec}: no bitstreams found")
                continue
            compliant = 100.0 * np.mean(sizes <= config.MAX_SIZE_BYTES)
            writer.writerow(
                [
                    args.resolution,
                    codec,
                    sizes.size,
                    f"{sizes.mean():.1f}",
                    int(np.median(sizes)),
                    int(sizes.min()),
                    int(sizes.max()),
                    f"{compliant:.1f}",
                ]
            )
            print(
                f"{codec}@{args.resolution}: mean {sizes.mean():.0f} B, "
                f"{compliant:.1f}% <= {config.MAX_SIZE_BYTES} B"
            )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
