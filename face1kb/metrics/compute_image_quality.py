"""Perceptual image quality of the compressed images: SSIM and LPIPS.

Compares every decoded compressed image against its aligned original and
reports structural similarity (SSIM, higher is better) and the LPIPS deep
perceptual distance (AlexNet backbone, lower is better). Writes a per-image
CSV and a per-codec summary::

    outputs/metrics/image_quality_<resolution>.csv
    outputs/metrics/image_quality_<resolution>_summary.csv

All images are evaluated, including those whose bitstream exceeded the 1 kB
budget (they are retained in the paper's evaluation as well).

Usage
-----
    python -m face1kb.metrics.compute_image_quality --resolution 112
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

from face1kb import config

_lpips_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _get_lpips():
    """Lazily create the LPIPS(AlexNet) metric on the available device."""
    global _lpips_model
    if _lpips_model is None:
        import lpips

        _lpips_model = lpips.LPIPS(net="alex").to(_device)
    return _lpips_model


def _to_lpips_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert an HxWx3 uint8 array to an LPIPS input in [-1, 1]."""
    tensor = torch.from_numpy(array).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return tensor.to(_device)


def compare(original: np.ndarray, decoded: np.ndarray) -> tuple[float, float]:
    """Return ``(ssim, lpips)`` between two uint8 RGB arrays."""
    if original.shape != decoded.shape:
        decoded = np.asarray(
            Image.fromarray(decoded).resize(
                (original.shape[1], original.shape[0]), Image.LANCZOS
            ),
            dtype=np.uint8,
        )
    ssim = structural_similarity(
        original, decoded, channel_axis=2, data_range=255
    )
    with torch.no_grad():
        lpips_value = _get_lpips()(
            _to_lpips_tensor(original), _to_lpips_tensor(decoded)
        ).item()
    return float(ssim), float(lpips_value)


def evaluate_codec(resolution: int, codec: str) -> list[dict]:
    """Compare all decoded images of one codec against the originals."""
    original_dir = config.aligned_dir(resolution)
    decoded_dir = config.decompressed_dir(resolution) / codec
    if not decoded_dir.is_dir():
        print(f"skipping {codec}@{resolution}: {decoded_dir} does not exist")
        return []

    rows = []
    files = sorted(decoded_dir.rglob("*.png"))
    for decoded_path in tqdm(files, desc=f"{codec}@{resolution}"):
        rel_path = decoded_path.relative_to(decoded_dir)
        original_path = original_dir / rel_path
        if not original_path.exists():
            tqdm.write(f"missing original for {rel_path}")
            continue
        original = np.asarray(
            Image.open(original_path).convert("RGB"), dtype=np.uint8
        )
        decoded = np.asarray(
            Image.open(decoded_path).convert("RGB"), dtype=np.uint8
        )
        ssim, lpips_value = compare(original, decoded)
        rows.append(
            {
                "resolution": resolution,
                "codec": codec,
                "image": str(rel_path),
                "ssim": ssim,
                "lpips": lpips_value,
            }
        )
    return rows


def write_outputs(rows: list[dict], resolution: int) -> None:
    """Write the per-image CSV and the per-codec mean summary."""
    out_dir = config.metrics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / f"image_quality_{resolution}.csv"
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {detail_path}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["codec"]].append((row["ssim"], row["lpips"]))

    summary_path = out_dir / f"image_quality_{resolution}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["resolution", "codec", "n_images", "mean_ssim", "mean_lpips"]
        )
        for codec in config.CODECS:
            if codec not in grouped:
                continue
            values = np.array(grouped[codec])
            writer.writerow(
                [
                    resolution,
                    codec,
                    len(values),
                    f"{values[:, 0].mean():.6f}",
                    f"{values[:, 1].mean():.6f}",
                ]
            )
    print(f"Wrote summary to {summary_path}")


def main() -> None:
    """Evaluate all codecs at one resolution."""
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
    args = parser.parse_args()

    rows = []
    for codec in args.codecs:
        rows.extend(evaluate_codec(args.resolution, codec))
    if not rows:
        raise SystemExit("No decoded images found; run decompression first.")
    write_outputs(rows, args.resolution)


if __name__ == "__main__":
    main()
