# SPDX-License-Identifier: MIT
"""ArcFace five-point alignment template and the crop geometry built on it.

All crops of the study are aligned to the canonical ArcFace five-point template
(:data:`ARCFACE_DST_112`, defined on a 112 x 112 canvas; image-space slot order
left eye, right eye, nose tip, left mouth corner, right mouth corner, where "left"
means lower x). A crop of ``size`` pixels uses the same template scaled by
``size / 112``: with ``M_112`` the similarity transform that maps the source
landmarks onto the 112 px template, the ``size`` px crop is warped with
``M_size = (size / 112) * M_112``. Every resolution therefore shows the same field of
view and only the pixel count changes.

Two consequences are used by the pipeline:

* **Integer-ratio resolutions are exact subsamplings.** ``cv2.warpAffine`` samples
  destination pixel ``(u, v)`` at ``M^-1 (u, v)`` (no half-pixel offset), so the
  pixel ``(u, v)`` of the ``size / k`` px crop is sampled at the same source point
  as the pixel ``(k u, k v)`` of the ``size`` px crop. Hence
  ``aligned_112 == aligned_224[::2, ::2]`` and the 56 px crop is
  ``aligned_112[::2, ::2]`` (``== aligned_224[::4, ::4]``), bit for bit, for any
  interpolation mode. :func:`derive_resolution` uses this; other ratios (e.g. 80 px
  from 224 px) can only be approximated by resampling a crop.
* **Crop-tightness variants are sub-windows of the standard frame.** The
  ``tight`` / ``mid`` / ``fill`` templates (:data:`CROP_VARIANTS`) are the ArcFace
  template scaled to a target inter-ocular distance (IOD) and shifted so that the
  eye line sits at a given height; each is a similarity image ``G`` of the standard
  template (uniform scale, no rotation). Least-squares similarity fitting is
  equivariant under such a ``G``, so the variant transform is exactly
  ``G o M_112`` and a variant crop can be rendered from a standard aligned crop
  without landmarks (:func:`render_variant`). Rendering from the 224 px crop adds a
  second interpolation, so the result differs slightly from a crop warped directly
  from the source photograph.

The similarity estimate (:func:`similarity_transform`) follows Umeyama's
least-squares method, as ``skimage.transform.SimilarityTransform.estimate`` does.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

log = logging.getLogger(__name__)

#: Side of the canvas on which the template is defined (px).
TEMPLATE_SIZE = 112

#: Canonical ArcFace five-point template on the 112 px canvas, slot order
#: (left eye, right eye, nose tip, left mouth corner, right mouth corner) in image
#: space (left = lower x).
ARCFACE_DST_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)

#: Midpoint of the two eye centres of the 112 px template.
EYE_MID_112 = ARCFACE_DST_112[:2].mean(axis=0)
#: Inter-ocular distance of the 112 px template (px).
IOD_112 = float(np.hypot(*(ARCFACE_DST_112[1] - ARCFACE_DST_112[0])))
#: IOD as a fraction of the crop width for the standard template (~0.315).
STANDARD_IOD_RATIO = IOD_112 / TEMPLATE_SIZE

#: Crop-tightness presets: name -> (IOD / crop width, eye line / crop height).
CROP_VARIANTS: dict[str, tuple[float, float]] = {
    "tight": (0.40, 0.34),
    "mid": (0.46, 0.27),
    "fill": (0.52, 0.20),
}


# --------------------------------------------------------------------- template
def template(size: int = TEMPLATE_SIZE) -> np.ndarray:
    """Return the ArcFace template scaled to a ``size`` px crop (5 x 2 float64)."""
    return ARCFACE_DST_112 * (float(size) / TEMPLATE_SIZE)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform mapping ``src`` points onto ``dst``.

    Umeyama's method (rotation, uniform scale and translation; no reflection),
    the estimator of ``skimage.transform.SimilarityTransform.estimate``.

    Parameters
    ----------
    src, dst : ndarray, shape (N, 2)
        Corresponding points (``N >= 2``), ``(x, y)`` order.

    Returns
    -------
    ndarray, shape (2, 3)
        Affine matrix ``[[a, -b, tx], [b, a, ty]]`` with ``dst ~ M @ [x, y, 1]``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"src/dst must both be (N, 2); got {src.shape}, {dst.shape}")
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    a = dst_c.T @ src_c / n
    d = np.ones(2)
    if np.linalg.det(a) < 0:
        d[1] = -1.0
    u, s, vt = np.linalg.svd(a)
    rank = np.linalg.matrix_rank(a)
    if rank == 0:
        raise ValueError("degenerate point configuration")
    if rank == 1:
        if np.linalg.det(u) * np.linalg.det(vt) > 0:
            r = u @ vt
        else:
            dd = d[1]
            d[1] = -1.0
            r = u @ np.diag(d) @ vt
            d[1] = dd
    else:
        r = u @ np.diag(d) @ vt
    scale = 1.0 / src_c.var(axis=0).sum() * (s @ d)
    m = np.zeros((2, 3))
    m[:, :2] = scale * r
    m[:, 2] = dst_mean - scale * (r @ src_mean)
    return m


def alignment_matrix(landmarks: np.ndarray, size: int = TEMPLATE_SIZE) -> np.ndarray:
    """Similarity matrix that warps an image with ``landmarks`` to a ``size`` crop.

    The transform is fitted at 112 px and scaled by ``size / 112``
    (``M_size = (size / 112) * M_112``), so all resolutions share the framing.

    Parameters
    ----------
    landmarks : ndarray, shape (5, 2)
        Source landmarks in :data:`ARCFACE_DST_112` slot order.
    size : int
        Output crop side (px).
    """
    m112 = similarity_transform(landmarks, ARCFACE_DST_112)
    return m112 * (float(size) / TEMPLATE_SIZE)


def warp_to_template(
    image: np.ndarray,
    landmarks: np.ndarray,
    size: int = TEMPLATE_SIZE,
    interpolation: int | None = None,
) -> np.ndarray:
    """Warp ``image`` onto the ArcFace template at ``size`` px (zero border).

    Parameters
    ----------
    image : ndarray
        Source image (any channel order; it is not changed).
    landmarks : ndarray, shape (5, 2)
        Source landmarks in :data:`ARCFACE_DST_112` slot order.
    size : int
        Output crop side (px).
    interpolation : int, optional
        OpenCV interpolation flag; default ``cv2.INTER_LINEAR``.
    """
    import cv2  # noqa: PLC0415

    flags = cv2.INTER_LINEAR if interpolation is None else interpolation
    m = alignment_matrix(landmarks, size)
    return cv2.warpAffine(image, m, (size, size), flags=flags, borderValue=0.0)


# --------------------------------------------------------------------- variants
def _variant_params(variant: str | tuple[float, float]) -> tuple[float, float]:
    if isinstance(variant, str):
        if variant not in CROP_VARIANTS:
            raise ValueError(
                f"unknown crop variant {variant!r}; choose from {list(CROP_VARIANTS)}"
            )
        return CROP_VARIANTS[variant]
    iod_ratio, eye_y_frac = variant
    return float(iod_ratio), float(eye_y_frac)


def variant_scale(variant: str | tuple[float, float]) -> float:
    """Zoom factor of a crop variant relative to the standard template."""
    iod_ratio, _ = _variant_params(variant)
    return (iod_ratio * TEMPLATE_SIZE) / IOD_112


def variant_template(
    variant: str | tuple[float, float], size: int = TEMPLATE_SIZE
) -> np.ndarray:
    """Destination template of a crop variant at ``size`` px (5 x 2 float64).

    The ArcFace template is scaled about the eye midpoint to the variant IOD and
    translated so that the eye midpoint lands at ``(size / 2, eye_y_frac * size)``.

    Parameters
    ----------
    variant : str or (float, float)
        A :data:`CROP_VARIANTS` name or an ``(iod_ratio, eye_y_frac)`` pair.
    size : int
        Crop side (px); the variants of the study are defined at 112 px.
    """
    _, eye_y_frac = _variant_params(variant)
    scale = variant_scale(variant)
    tmpl = template(size)
    eye_mid = tmpl[:2].mean(axis=0)
    return (tmpl - eye_mid) * scale + np.array([size / 2.0, eye_y_frac * size])


def variant_from_standard(variant: str | tuple[float, float]) -> np.ndarray:
    """Affine ``G`` (2 x 3) mapping standard-112 coordinates to variant-112 ones.

    ``G`` maps :data:`ARCFACE_DST_112` onto :func:`variant_template` exactly:
    ``p_v = s (p - eye_mid) + (56, eye_y_frac * 112)`` with ``s``
    :func:`variant_scale`.
    """
    _, eye_y_frac = _variant_params(variant)
    s = variant_scale(variant)
    centre = np.array([TEMPLATE_SIZE / 2.0, eye_y_frac * TEMPLATE_SIZE])
    g = np.zeros((2, 3))
    g[0, 0] = g[1, 1] = s
    g[:, 2] = centre - s * EYE_MID_112
    return g


def variant_matrix(
    variant: str | tuple[float, float], src_res: int = 224, out_res: int = TEMPLATE_SIZE
) -> np.ndarray:
    """Affine (2 x 3) that renders a variant crop from a standard aligned crop.

    Parameters
    ----------
    variant : str or (float, float)
        A :data:`CROP_VARIANTS` name or an ``(iod_ratio, eye_y_frac)`` pair.
    src_res : int
        Resolution of the standard aligned crop that is resampled.
    out_res : int
        Output resolution; the variant geometry is defined at 112 px and scales
        with ``out_res / 112`` like every other crop.
    """
    g = variant_from_standard(variant)
    to_std = np.diag([TEMPLATE_SIZE / src_res, TEMPLATE_SIZE / src_res, 1.0])
    m = (g @ to_std) * (out_res / TEMPLATE_SIZE)
    return m[:2]


def variant_field_of_view(variant: str | tuple[float, float]) -> tuple[float, ...]:
    """Window of the standard 112 px frame shown by a variant crop.

    Returns ``(x0, y0, x1, y1)`` in standard-112 coordinates. A window inside
    ``[0, 112]`` means the variant can be rendered from a standard crop without
    running out of image content.
    """
    g = variant_from_standard(variant)
    s = g[0, 0]
    x0, y0 = (-g[:, 2]) / s
    x1, y1 = (np.array([TEMPLATE_SIZE, TEMPLATE_SIZE]) - g[:, 2]) / s
    return float(x0), float(y0), float(x1), float(y1)


def render_variant(
    crop: np.ndarray,
    variant: str | tuple[float, float],
    out_res: int = TEMPLATE_SIZE,
    interpolation: int | None = None,
) -> np.ndarray:
    """Render a crop-tightness variant from a standard aligned crop.

    Parameters
    ----------
    crop : ndarray, shape (R, R[, C])
        Standard aligned crop of any resolution ``R`` (224 px gives the closest
        match to a variant warped from the source photograph).
    variant : str or (float, float)
        A :data:`CROP_VARIANTS` name or an ``(iod_ratio, eye_y_frac)`` pair.
    out_res : int
        Output resolution (the study uses 112 px).
    interpolation : int, optional
        OpenCV interpolation flag; default ``cv2.INTER_CUBIC``, which from a
        224 px crop comes closest to a variant warped directly from the source
        photograph (mean absolute difference about 1.5/255 on Color FERET,
        against 1.6/255 for bilinear).
    """
    import cv2  # noqa: PLC0415

    src_res = int(crop.shape[0])
    if crop.shape[1] != src_res:
        raise ValueError(f"expected a square crop, got {crop.shape}")
    flags = cv2.INTER_CUBIC if interpolation is None else interpolation
    m = variant_matrix(variant, src_res=src_res, out_res=out_res)
    return cv2.warpAffine(crop, m, (out_res, out_res), flags=flags, borderValue=0.0)


def variants_metadata(
    variants: Mapping[str, tuple[float, float]] | None = None,
) -> dict:
    """Describe the variant templates (the content of ``variants_templates.json``)."""
    variants = CROP_VARIANTS if variants is None else variants
    return {
        "resolution": TEMPLATE_SIZE,
        "variants": {
            k: {
                "iod_ratio": v[0],
                "eye_line_frac": v[1],
                "dst_template": variant_template(v).round(3).tolist(),
                "std_field_of_view": [round(c, 3) for c in variant_field_of_view(v)],
            }
            for k, v in variants.items()
        },
        "standard_iod_ratio": round(STANDARD_IOD_RATIO, 3),
    }


# ------------------------------------------------------------------ resolutions
def is_exact_subsampling(src_res: int, dst_res: int) -> bool:
    """Tell whether a ``dst_res`` crop is an exact subsampling of a ``src_res`` one."""
    return src_res >= dst_res and src_res % dst_res == 0


def derive_resolution(
    crop: np.ndarray, dst_res: int, interpolation: int | None = None
) -> tuple[np.ndarray, bool]:
    """Derive the ``dst_res`` px aligned crop from an aligned crop of another size.

    If the source side is an integer multiple ``k`` of ``dst_res`` the result is
    ``crop[::k, ::k]``, which equals a crop warped directly from the source
    photograph (see the module docstring). Otherwise the crop is resampled with
    ``cv2.warpAffine`` at scale ``dst_res / src_res`` (the sampling positions of a
    direct warp, bilinear by default), which only approximates a direct warp.

    Returns
    -------
    tuple[ndarray, bool]
        The derived crop and whether it is exact.
    """
    src_res = int(crop.shape[0])
    if crop.shape[1] != src_res:
        raise ValueError(f"expected a square crop, got {crop.shape}")
    if dst_res == src_res:
        return crop.copy(), True
    if is_exact_subsampling(src_res, dst_res):
        k = src_res // dst_res
        return np.ascontiguousarray(crop[::k, ::k]), True
    import cv2  # noqa: PLC0415

    flags = cv2.INTER_LINEAR if interpolation is None else interpolation
    r = dst_res / src_res
    m = np.array([[r, 0.0, 0.0], [0.0, r, 0.0]])
    out = cv2.warpAffine(crop, m, (dst_res, dst_res), flags=flags, borderValue=0.0)
    return out, False
