"""JPEG-FzT codec: F-transform pre/post-processing around standard JPEG.

Python implementation of the hybrid compression approach evaluated in the
paper. During compression, the image undergoes a direct F-transform (FzT)
before JPEG encoding; the process is reversed during decompression using the
inverse F-transform (iFzT). The implementation mirrors the C++ reference
(``adjoint_compress_color_lite``) including its exact boundary behaviour.

The stored bitstream is the JPEG-encoded low-resolution F-transform stage;
decompression upsamples it back to full resolution with the inverse basis
function stored in ``fzt_weights.txt``.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "fzt_weights.txt"


def load_weights(weights_path: str | Path | None = None) -> np.ndarray:
    """Load the inverse-basis-function weights used by the iFzT stage."""
    path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
    values = [
        float(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"No weights loaded from {path}")
    return np.asarray(values, dtype=np.float32)


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    return np.asarray(rgb, dtype=np.uint8)


def encode_stage(src: np.ndarray, h: int, hh: int) -> np.ndarray:
    """Direct F-transform: cosine-weighted downsampling by factor *h*.

    Parameters
    ----------
    src : numpy.ndarray
        Input image, uint8 array of shape ``(H, W, 3)``.
    h : int
        Sampling step (2 in the paper's configuration).
    hh : int
        Additional basis-function support (0 in the paper's configuration).

    Returns
    -------
    numpy.ndarray
        Low-resolution stage of shape ``(H // h, W // h, 3)``.
    """
    height, width, _ = src.shape
    nw = max(1, width // h)
    nh = max(1, height // h)

    bf = np.array(
        [
            (np.cos(np.pi * (float(i) / float(h + hh))) + 1.0) / 2.0
            for i in range(h + hh + 1)
        ],
        dtype=np.float32,
    )

    # Build weighted sums for all pixels via shifted slice accumulation.
    # This keeps the exact boundary behaviour of the C++ reference loops
    # (out-of-bounds samples are skipped; normalisation uses valid weights).
    src_f = src.astype(np.float32, copy=False)
    weighted_sum = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    radius = h + hh

    for dy in range(-radius, radius + 1):
        wy = float(bf[abs(dy)])
        if dy >= 0:
            dst_y = slice(0, height - dy)
            src_y = slice(dy, height)
        else:
            dst_y = slice(-dy, height)
            src_y = slice(0, height + dy)

        for dx in range(-radius, radius + 1):
            wx = float(bf[abs(dx)])
            w = wx * wy
            if dx >= 0:
                dst_x = slice(0, width - dx)
                src_x = slice(dx, width)
            else:
                dst_x = slice(-dx, width)
                src_x = slice(0, width + dx)

            weighted_sum[dst_y, dst_x] += src_f[src_y, src_x] * w
            weight_sum[dst_y, dst_x] += w

    filtered = weighted_sum / np.maximum(weight_sum[..., None], 1e-12)

    # Match the reference indexing exactly: when dimensions are not divisible
    # by h, the last output cell is overwritten by the final sampled anchor
    # (largest multiple of h still inside the image).
    x_coords = np.arange(nw, dtype=np.int32) * h
    y_coords = np.arange(nh, dtype=np.int32) * h
    x_coords[-1] = ((width - 1) // h) * h
    y_coords[-1] = ((height - 1) // h) * h

    compressed = filtered[y_coords[:, None], x_coords[None, :]]
    return np.clip(np.rint(compressed), 0, 255).astype(np.uint8)


def decode_stage(  # noqa: C901 - mirrors the reference iFzT decode stages
    lowres_jpeg_decoded: np.ndarray,
    output_size: tuple[int, int],
    ibf: np.ndarray,
    h: int,
    rad: int,
) -> np.ndarray:
    """Inverse F-transform: upsample the JPEG-decoded low-resolution stage.

    Parameters
    ----------
    lowres_jpeg_decoded : numpy.ndarray
        JPEG-decoded low-resolution stage, uint8 ``(nh, nw, 3)``.
    output_size : tuple of int
        Target ``(width, height)`` of the reconstruction.
    ibf : numpy.ndarray
        Inverse basis function weights (see :func:`load_weights`).
    h : int
        Sampling step used during encoding.
    rad : int
        Neighbourhood radius of the inverse transform.

    Returns
    -------
    numpy.ndarray
        Reconstructed image, uint8 ``(height, width, 3)``.
    """
    out_w, out_h = output_size
    nh, nw, _ = lowres_jpeg_decoded.shape
    mult = (1.0 / float(h)) * 20.0

    max_neighbors = 2 * rad + 1
    max_ibf_idx = len(ibf) - 1

    # Precompute x-neighbour lists for each output x in the same order as
    # the original per-pixel loop (xx ascending).
    x_nx = np.zeros((out_w, max_neighbors), dtype=np.int32)
    x_w = np.zeros((out_w, max_neighbors), dtype=np.float32)
    x_count = np.zeros(out_w, dtype=np.int32)
    for x in range(out_w):
        cnt = 0
        x0 = max(x - rad, 0)
        x1 = min(x + rad, out_w - 1)
        for xx in range(x0, x1 + 1):
            if xx % h:
                continue
            wx_idx = min(int(round(abs(x - xx) * mult)), max_ibf_idx)
            x_nx[x, cnt] = min(xx // h, nw - 1)
            x_w[x, cnt] = float(ibf[wx_idx])
            cnt += 1
        x_count[x] = cnt

    max_x_count = int(np.max(x_count)) if out_w else 0
    x_valid_masks = [(x_count > i) for i in range(max_x_count)]
    x_nx_cols = [x_nx[mask, i] for i, mask in enumerate(x_valid_masks)]
    x_w_cols = [x_w[mask, i] for i, mask in enumerate(x_valid_masks)]

    # Precompute y-neighbour lists for each output y in the same order as
    # the original per-pixel loop (yy ascending).
    y_ny = np.zeros((out_h, max_neighbors), dtype=np.int32)
    y_w = np.zeros((out_h, max_neighbors), dtype=np.float32)
    y_count = np.zeros(out_h, dtype=np.int32)
    for y in range(out_h):
        cnt = 0
        y0 = max(y - rad, 0)
        y1 = min(y + rad, out_h - 1)
        for yy in range(y0, y1 + 1):
            if yy % h:
                continue
            wy_idx = min(int(round(abs(y - yy) * mult)), max_ibf_idx)
            y_ny[y, cnt] = min(yy // h, nh - 1)
            y_w[y, cnt] = float(ibf[wy_idx])
            cnt += 1
        y_count[y] = cnt

    lowres_f = lowres_jpeg_decoded.astype(np.float32, copy=False)
    result = np.zeros((out_h, out_w, 3), dtype=np.float32)

    for y in range(out_h):
        sum_rgb = np.zeros((out_w, 3), dtype=np.float32)
        num = np.zeros(out_w, dtype=np.float32)
        for yi in range(y_count[y]):
            ny = y_ny[y, yi]
            wy = y_w[y, yi]
            for xi in range(max_x_count):
                mask = x_valid_masks[xi]
                if not np.any(mask):
                    continue
                w = x_w_cols[xi] * wy
                sum_rgb[mask] += lowres_f[ny, x_nx_cols[xi]] * w[:, None]
                num[mask] += w

        valid = num > 0.0
        result[y, valid] = sum_rgb[valid] / num[valid, None]

    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def compress_with_quality(
    image: Image.Image, quality: int
) -> tuple[Image.Image, int]:
    """Run the JPEG-FzT encoder with a fixed JPEG quality.

    Returns
    -------
    compressed_image : PIL.Image.Image
        Low-resolution F-transform stage (this is what gets stored,
        JPEG-encoded, as the ``.fzt`` bitstream).
    compressed_size_bytes : int
        JPEG file size of the stage when stored to disk.
    """
    quality = int(max(1, min(100, quality)))
    h, hh = 2, 0
    src = _to_rgb_array(image)

    compressed_lowres = encode_stage(src, h=h, hh=hh)
    compressed_lowres_pil = Image.fromarray(compressed_lowres, mode="RGB")

    with NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        compressed_lowres_pil.save(
            tmp.name, format="JPEG", quality=quality, optimize=True
        )
        compressed_size_bytes = Path(tmp.name).stat().st_size

    return compressed_lowres_pil, int(compressed_size_bytes)


def decompress(
    fzt_path: str | Path,
    output_size: tuple[int, int],
    weights_path: str | Path | None = None,
) -> np.ndarray:
    """Decode a stored ``.fzt`` bitstream back to a full-resolution image.

    Parameters
    ----------
    fzt_path : str or Path
        Stored JPEG-FzT bitstream (JPEG-encoded low-resolution stage).
    output_size : tuple of int
        Target ``(width, height)`` of the reconstruction.
    weights_path : str or Path, optional
        Override for the inverse-basis-function weights file.

    Returns
    -------
    numpy.ndarray
        Reconstructed RGB image, uint8 ``(height, width, 3)``.
    """
    lowres = np.asarray(Image.open(fzt_path).convert("RGB"), dtype=np.uint8)
    ibf = load_weights(weights_path)
    return decode_stage(lowres, output_size, ibf, h=2, rad=1)
