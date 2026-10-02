# SPDX-License-Identifier: MIT
"""Full-reference image quality of decoded crops against their aligned originals.

Five metrics, computed on RGB images scaled to ``[0, 1]`` (:func:`make_metrics`):

========  =====================================================  =============
name      implementation                                          better
========  =====================================================  =============
psnr      ``pyiqa.create_metric("psnr")`` -- RGB, no border crop  higher
ssim      ``pyiqa.create_metric("ssim")`` -- on the Y channel     higher
          (``test_y_channel=True``), no downsampling
ms_ssim   ``piq.multi_scale_ssim``, kernel 7, 4 scales with       higher
          weights (0.0448, 0.2856, 0.3001, 0.2363), data range 1
lpips     ``pyiqa.create_metric("lpips")`` -- AlexNet, v0.1       lower
dists     ``pyiqa.create_metric("dists")``                        lower
========  =====================================================  =============

MS-SSIM uses 4 scales and a 7 px Gaussian kernel (sigma 1.5) with the first four of
the standard five scale weights, which piq renormalises to sum 1; it is computed per
RGB channel and averaged. piq needs images of at least
``(kernel - 1) * 2 ** (scales - 1) + 1`` px, i.e. 161 px for the standard 5-scale /
11 px configuration but only 49 px here, so all crop sizes (64-224 px) are scored
with one configuration.

Images are scored in batches (:func:`score_images`, 64 per batch in the study); a
cell's per-image records are aggregated to medians by :func:`aggregate_quality`.
pyiqa downloads the LPIPS and DISTS weights on first use.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from .embeddings import CODECS

log = logging.getLogger(__name__)

#: Metric names, in column order.
METRIC_NAMES: tuple[str, ...] = ("psnr", "ssim", "ms_ssim", "lpips", "dists")
#: Metrics where a lower value is better.
LOWER_IS_BETTER: frozenset[str] = frozenset({"lpips", "dists"})
#: MS-SSIM Gaussian kernel size.
MSSSIM_KERNEL_SIZE = 7
#: MS-SSIM scale weights (4 scales).
MSSSIM_SCALE_WEIGHTS: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363)
#: Batch size of the study.
BATCH_SIZE = 64
#: Columns of a per-image record.
PER_IMAGE_COLUMNS: tuple[str, ...] = (
    "image",
    "subject",
    "codec",
    "res",
    "budget",
    *METRIC_NAMES,
)
#: Columns of ``quality/quality_<dataset>.csv`` (per-cell medians).
QUALITY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "codec",
    "res",
    "budget",
    "n",
    *METRIC_NAMES,
)

Metric = Callable[..., "object"]


def to_tensor(arr: np.ndarray):
    """``HxWx3`` uint8 RGB -> ``3xHxW`` float32 tensor in ``[0, 1]``."""
    import torch  # noqa: PLC0415

    return torch.from_numpy(arr.astype(np.float32) / 255).permute(2, 0, 1)


def make_metrics(device="cuda") -> dict[str, Metric]:
    """Build the five metric callables ``(test, ref) -> tensor[N]`` on ``device``."""
    import piq  # noqa: PLC0415
    import pyiqa  # noqa: PLC0415
    import torch  # noqa: PLC0415

    psnr = pyiqa.create_metric("psnr", device=device)
    ssim = pyiqa.create_metric("ssim", device=device)
    lpips = pyiqa.create_metric("lpips", device=device)
    dists = pyiqa.create_metric("dists", device=device)
    w4 = torch.tensor(MSSSIM_SCALE_WEIGHTS, device=device)

    def ms_ssim(t, r):
        return piq.multi_scale_ssim(
            t,
            r,
            kernel_size=MSSSIM_KERNEL_SIZE,
            scale_weights=w4,
            data_range=1.0,
            reduction="none",
        )

    return {
        "psnr": psnr,
        "ssim": ssim,
        "ms_ssim": ms_ssim,
        "lpips": lpips,
        "dists": dists,
    }


def score_batch(metrics: dict[str, Metric], test, ref) -> dict[str, np.ndarray]:
    """Score a batch: ``test``/``ref`` are ``Nx3xHxW`` tensors on the metric device."""
    import torch  # noqa: PLC0415

    scores = {}
    with torch.no_grad():
        for name, fn in metrics.items():
            scores[name] = fn(test, ref).flatten().cpu().numpy()
    return scores


def score_images(
    items: Iterable[tuple[dict, np.ndarray, np.ndarray]],
    metrics: dict[str, Metric],
    *,
    batch: int = BATCH_SIZE,
    device="cuda",
) -> list[dict]:
    """Score decoded crops against their references, ``batch`` images at a time.

    Parameters
    ----------
    items : iterable of (dict, numpy.ndarray, numpy.ndarray)
        ``(record, decoded, reference)``: the record's keys (e.g. ``image``,
        ``subject``, ``codec``, ``res``, ``budget``) are copied into the output;
        the images are ``HxWx3`` uint8 RGB of equal size.
    metrics : dict
        :func:`make_metrics` result.
    batch : int
        Images per metric call (the study used 64; LPIPS and DISTS results can
        depend on the batch composition in the last float32 bits).
    device : str or torch.device
        Device of the metrics.

    Returns
    -------
    list of dict
        One record per item, in input order, with one float per metric.
    """
    import torch  # noqa: PLC0415

    records: list[dict] = []
    buf_t, buf_r, buf_meta = [], [], []

    def flush():
        if not buf_t:
            return
        t = torch.stack(buf_t).to(device)
        r = torch.stack(buf_r).to(device)
        scores = score_batch(metrics, t, r)
        for i, meta in enumerate(buf_meta):
            records.append({**meta, **{k: float(v[i]) for k, v in scores.items()}})
        buf_t.clear()
        buf_r.clear()
        buf_meta.clear()

    for meta, dec, ref in items:
        buf_t.append(to_tensor(dec))
        buf_r.append(to_tensor(ref))
        buf_meta.append(meta)
        if len(buf_t) >= batch:
            flush()
    flush()
    return records


def aggregate_quality(per_image, dataset: str, codecs: Sequence[str] | None = CODECS):
    """Per-cell medians of per-image records (``quality_<dataset>.csv`` rows).

    Parameters
    ----------
    per_image : pandas.DataFrame
        Records with ``codec, res, budget`` and the metric columns.
    dataset : str
        Value of the ``dataset`` column.
    codecs : sequence of str or None
        Keep only these codecs (default :data:`~face1kb.eval.embeddings.CODECS`);
        ``None`` keeps all.

    Returns
    -------
    pandas.DataFrame
        :data:`QUALITY_COLUMNS`, sorted by codec, res and budget.
    """
    if codecs is not None:
        per_image = per_image[per_image["codec"].isin(list(codecs))]
    grp = per_image.groupby(["codec", "res", "budget"], sort=True)
    summary = grp[list(METRIC_NAMES)].median()
    summary.insert(0, "n", grp.size())
    summary = summary.reset_index()
    summary.insert(0, "dataset", dataset)
    return summary[list(QUALITY_COLUMNS)].sort_values(["codec", "res", "budget"])
