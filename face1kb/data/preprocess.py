# SPDX-License-Identifier: MIT
"""Pre-compression preprocessing operators applied to aligned crops.

Operators (names as in the paper's preprocessing ablations):

====  ==========================================================================
std   no preprocessing
A1    edge-preserving smoothing, ``cv2.edgePreservingFilter`` (recursive filter,
      ``sigma_s=30``, ``sigma_r=0.25``)
A2    bilateral filter, ``d=7``, ``sigma_color=sigma_space=75``
A3    chroma de-emphasis: Gaussian blur (sigma 1.5) of the Cr/Cb planes in YCrCb
A4    non-local-means denoising, ``cv2.fastNlMeansDenoisingColored(8, 8, 7, 21)``
B1    background flattened to grey 128 with a BiRefNet foreground mask
B2    background flattened to grey 128 with the MediaPipe selfie-segmenter mask
C1    B1 followed by A1
C2    foveation: B1, then a bilateral filter (``d=9``, sigma 100) outside a soft
      elliptical mask over the eye-nose-mouth region of the aligned frame
====  ==========================================================================

The OpenCV operators work on BGR ``uint8`` images exactly as written here;
:func:`apply_operator` accepts RGB (default) or BGR input and returns the same
channel order. Segmentation models:

* BiRefNet (``ZhengPeng7/BiRefNet``, MIT) is loaded with ``transformers``
  (``trust_remote_code=True``) at the pinned revision
  :data:`face1kb.data.fetch.BIREFNET_REVISION` and run in fp32; the crop is resized
  to 512 x 512 for the mask, which is resized back with bilinear interpolation.
* The MediaPipe selfie segmenter (Apache-2.0) runs through the MediaPipe Tasks API
  on the crop resized to 256 x 256; install MediaPipe with
  ``pip install --no-deps mediapipe==0.10.35 sounddevice==0.5.5`` (BiRefNet's remote
  code needs ``kornia`` and ``einops``: the ``preprocess`` extra). The model file is
  downloaded on first use
  (:func:`face1kb.data.fetch.selfie_segmenter_path`).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from face1kb.data.crops import check_missing

log = logging.getLogger(__name__)

#: Grey level of the flattened background (B1, B2, C1, C2).
BG_FILL = 128
#: Operators that need only OpenCV.
CPU_OPERATORS: tuple[str, ...] = ("A1", "A2", "A3", "A4")
#: Operators that need a segmentation mask (BiRefNet for B1/C1/C2, MediaPipe B2).
SEG_OPERATORS: tuple[str, ...] = ("B1", "B2", "C1", "C2")
#: Operators that use the BiRefNet mask.
BIREFNET_OPERATORS: tuple[str, ...] = ("B1", "C1", "C2")
#: All operators, ``std`` being the unprocessed crop.
OPERATORS: tuple[str, ...] = ("std",) + CPU_OPERATORS + SEG_OPERATORS
#: Short descriptions of the operators.
OPERATOR_DESCRIPTIONS: dict[str, str] = {
    "std": "no preprocessing",
    "A1": "edge-preserving smoothing",
    "A2": "bilateral filter",
    "A3": "chroma de-emphasis",
    "A4": "non-local-means denoising",
    "B1": "background flatten (BiRefNet)",
    "B2": "background flatten (MediaPipe)",
    "C1": "B1 + A1",
    "C2": "foveated ROI (B1 + peripheral smoothing)",
}


# ------------------------------------------------------------------ OpenCV ops
def op_a1(bgr: np.ndarray) -> np.ndarray:
    """A1: edge-preserving smoothing (recursive filter)."""
    import cv2  # noqa: PLC0415

    return cv2.edgePreservingFilter(
        bgr, flags=cv2.RECURS_FILTER, sigma_s=30, sigma_r=0.25
    )


def op_a2(bgr: np.ndarray) -> np.ndarray:
    """A2: bilateral filter."""
    import cv2  # noqa: PLC0415

    return cv2.bilateralFilter(bgr, 7, 75, 75)


def op_a3(bgr: np.ndarray) -> np.ndarray:
    """A3: chroma de-emphasis (blur of the Cr/Cb planes)."""
    import cv2  # noqa: PLC0415

    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 1] = cv2.GaussianBlur(ycc[:, :, 1], (0, 0), 1.5)
    ycc[:, :, 2] = cv2.GaussianBlur(ycc[:, :, 2], (0, 0), 1.5)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def op_a4(bgr: np.ndarray) -> np.ndarray:
    """A4: non-local-means denoising."""
    import cv2  # noqa: PLC0415

    return cv2.fastNlMeansDenoisingColored(bgr, None, 8, 8, 7, 21)


#: OpenCV operators on BGR uint8 images.
CPU_FUNCS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "A1": op_a1,
    "A2": op_a2,
    "A3": op_a3,
    "A4": op_a4,
}


def composite(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Composite the foreground (``alpha`` in [0, 1], H x W) over grey 128."""
    a = alpha[..., None]
    return (
        (bgr.astype(np.float32) * a + BG_FILL * (1.0 - a)).clip(0, 255).astype(np.uint8)
    )


def fovea_mask(size: int = 112) -> np.ndarray:
    """Soft elliptical mask (H x W x 1, float32) over the eye-nose-mouth core.

    Defined for the 112 px aligned frame (ellipse centre (56, 72), semi-axes
    (36, 46), Gaussian feather sigma 6); other sizes scale these values.
    """
    import cv2  # noqa: PLC0415

    r = size / 112.0
    m = np.zeros((size, size), np.float32)
    centre = (int(round(56 * r)), int(round(72 * r)))
    axes = (int(round(36 * r)), int(round(46 * r)))
    cv2.ellipse(m, centre, axes, 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 6.0 * r)[..., None]


def foveate(flat: np.ndarray, fovea: np.ndarray | None = None) -> np.ndarray:
    """C2 from a background-flattened crop: smooth the periphery outside the fovea."""
    import cv2  # noqa: PLC0415

    if fovea is None:
        fovea = fovea_mask(flat.shape[0])
    flat_f = flat.astype(np.float32)
    smooth = cv2.bilateralFilter(flat_f.astype(np.uint8), 9, 100, 100).astype(
        np.float32
    )
    return (flat_f * fovea + smooth * (1 - fovea)).clip(0, 255).astype(np.uint8)


def apply_with_mask(bgr: np.ndarray, op: str, alpha: np.ndarray) -> np.ndarray:
    """Apply a segmentation operator (B1/B2/C1/C2) given its foreground mask."""
    if op in ("B1", "B2"):
        return composite(bgr, alpha)
    if op == "C1":
        return op_a1(composite(bgr, alpha))
    if op == "C2":
        return foveate(composite(bgr, alpha))
    raise ValueError(f"{op!r} is not a segmentation operator")


# ------------------------------------------------------------------ segmenters
class MediaPipeSegmenter:
    """MediaPipe selfie segmenter returning a foreground probability mask.

    Parameters
    ----------
    model_path : path, optional
        ``selfie_segmenter.tflite``; downloaded and verified when omitted.
    """

    def __init__(self, model_path: str | os.PathLike | None = None):
        from mediapipe.tasks import python as mpp  # noqa: PLC0415
        from mediapipe.tasks.python import vision  # noqa: PLC0415

        if model_path is None:
            from face1kb.data.fetch import selfie_segmenter_path  # noqa: PLC0415

            model_path = selfie_segmenter_path()
        self._seg = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
                output_confidence_masks=True,
            )
        )

    def close(self) -> None:
        """Release the MediaPipe graph (also done by the context manager)."""
        if self._seg is not None:
            self._seg.close()
            self._seg = None

    def __enter__(self) -> MediaPipeSegmenter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def mask(self, bgr: np.ndarray) -> np.ndarray:
        """Foreground mask of a BGR uint8 crop, resized to the crop (float32)."""
        import cv2  # noqa: PLC0415
        import mediapipe as mp  # noqa: PLC0415

        rgb = np.ascontiguousarray(
            cv2.cvtColor(cv2.resize(bgr, (256, 256)), cv2.COLOR_BGR2RGB)
        )
        res = self._seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        return cv2.resize(
            res.confidence_masks[0].numpy_view(), (bgr.shape[1], bgr.shape[0])
        )


def load_birefnet(
    device: str | None = None,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
):
    """Load BiRefNet (fp32, eval mode) at the pinned revision.

    Parameters
    ----------
    device : str, optional
        Torch device; CUDA when available.
    revision : str, optional
        Hub revision; default :data:`face1kb.data.fetch.BIREFNET_REVISION`.
    cache_dir : path, optional
        Hugging Face cache folder; default ``<FACE1KB_MODELS_ROOT>/hf``.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForImageSegmentation  # noqa: PLC0415

    from face1kb import config  # noqa: PLC0415
    from face1kb.data.fetch import BIREFNET_REPO, BIREFNET_REVISION  # noqa: PLC0415

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache = config.models_dir("hf") if cache_dir is None else Path(cache_dir)
    model = AutoModelForImageSegmentation.from_pretrained(
        BIREFNET_REPO,
        trust_remote_code=True,
        revision=revision or BIREFNET_REVISION,
        cache_dir=str(cache),
    )
    return model.to(dev).float().eval()


class BiRefNetSegmenter:
    """BiRefNet foreground masks for batches of crops.

    Parameters
    ----------
    device, revision, cache_dir
        See :func:`load_birefnet`.
    """

    def __init__(self, device=None, revision=None, cache_dir=None):
        import torch  # noqa: PLC0415
        import torchvision.transforms as T  # noqa: PLC0415

        self.torch = torch
        self.model = load_birefnet(device, revision, cache_dir)
        self.device = next(self.model.parameters()).device
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((512, 512)),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def masks(self, bgr_list: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Foreground masks (float32, crop size) of a batch of BGR uint8 crops."""
        import cv2  # noqa: PLC0415

        torch = self.torch
        x = torch.stack([self.transform(b[:, :, ::-1].copy()) for b in bgr_list])
        with torch.no_grad():
            out = self.model(x.to(self.device))
        pred = out[-1] if isinstance(out, (list, tuple)) else out
        alpha = pred.sigmoid().cpu().numpy()[:, 0]
        return [
            cv2.resize(a, (b.shape[1], b.shape[0])) for a, b in zip(alpha, bgr_list)
        ]


# ------------------------------------------------------------------ one image
def apply_operator(
    img: np.ndarray,
    op: str,
    bgr: bool = False,
    birefnet: BiRefNetSegmenter | None = None,
    mediapipe: MediaPipeSegmenter | None = None,
) -> np.ndarray:
    """Apply operator ``op`` to one ``uint8`` crop.

    Parameters
    ----------
    img : ndarray, shape (H, W, 3)
        Crop in RGB order (``bgr=False``) or BGR order (``bgr=True``).
    op : str
        One of :data:`OPERATORS`.
    bgr : bool
        Channel order of ``img`` and of the result.
    birefnet, mediapipe : optional
        Segmenters for the B/C operators; created on demand when omitted (a
        BiRefNet model load is expensive, so pass one when processing many crops).
    """
    if op not in OPERATORS:
        raise ValueError(f"unknown operator {op!r}; choose from {OPERATORS}")
    if op == "std":
        return img
    work = img if bgr else np.ascontiguousarray(img[:, :, ::-1])
    if op in CPU_FUNCS:
        out = CPU_FUNCS[op](work)
    elif op == "B2":
        seg = mediapipe or MediaPipeSegmenter()
        out = apply_with_mask(work, op, seg.mask(work))
    else:
        seg = birefnet or BiRefNetSegmenter()
        out = apply_with_mask(work, op, seg.masks([work])[0])
    return out if bgr else np.ascontiguousarray(out[:, :, ::-1])


# ------------------------------------------------------------------ datasets
def _cpu_task(task: tuple[str, str, str, bool]) -> str:
    """Apply an OpenCV operator to one crop: ``written``, ``skipped`` or ``missing``."""
    import cv2  # noqa: PLC0415

    src, dst, op, overwrite = task
    if not overwrite and os.path.exists(dst):
        return "skipped"
    bgr = cv2.imread(src, cv2.IMREAD_COLOR)
    if bgr is None:
        log.warning("cannot read %s", src)
        return "missing"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, CPU_FUNCS[op](bgr))
    return "written"


def preprocess_folder(
    src_dir: str | os.PathLike,
    dst_dirs: dict[str, str | os.PathLike],
    rel_paths: Sequence[str],
    batch: int = 8,
    workers: int | None = None,
    overwrite: bool = False,
    device: str | None = None,
    strict: bool = True,
) -> dict[str, int]:
    """Write preprocessed copies of the crops ``rel_paths`` of ``src_dir``.

    Parameters
    ----------
    src_dir : path
        Aligned crop folder (e.g. ``aligned_112``).
    dst_dirs : dict
        ``{operator: output folder}``; outputs keep the relative paths.
    rel_paths : sequence of str
        Crops to process, in index order.
    batch : int
        BiRefNet batch size.
    workers : int, optional
        Processes for the OpenCV operators (default: CPU count minus 4).
    overwrite : bool
        Recompute outputs that already exist (otherwise they are skipped).
    device : str, optional
        Torch device for BiRefNet.
    strict : bool
        Raise :class:`face1kb.data.crops.MissingCropsError` if a source crop cannot
        be read (after processing all others); otherwise only log it.

    Returns
    -------
    dict
        Number of written files per operator.
    """
    import cv2  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    src_dir = Path(src_dir)
    unknown = set(dst_dirs) - set(CPU_OPERATORS + SEG_OPERATORS)
    if unknown:
        raise ValueError(f"unknown operators {sorted(unknown)}")
    written = {op: 0 for op in dst_dirs}
    skipped = {op: 0 for op in dst_dirs}
    missing: set[str] = set()
    cpu = [op for op in dst_dirs if op in CPU_OPERATORS]
    seg = [op for op in dst_dirs if op in SEG_OPERATORS]

    if cpu:
        tasks = [
            (str(src_dir / rp), str(Path(dst_dirs[op]) / rp), op, overwrite)
            for rp in rel_paths
            for op in cpu
        ]
        n_workers = workers or max(1, (os.cpu_count() or 8) - 4)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            done = ex.map(_cpu_task, tasks, chunksize=32)
            for (src, _, op, _), status in tqdm(
                zip(tasks, done), total=len(tasks), desc="opencv operators"
            ):
                written[op] += int(status == "written")
                skipped[op] += int(status == "skipped")
                if status == "missing":
                    missing.add(src)

    if seg:
        # per crop, the segmentation outputs still to write (all with ``overwrite``)
        need: dict[str, list[str]] = {}
        for rp in rel_paths:
            ops = [
                op for op in seg if overwrite or not (Path(dst_dirs[op]) / rp).exists()
            ]
            for op in seg:
                skipped[op] += int(op not in ops)
            if ops:
                need[rp] = ops
        todo = list(need)
        # the models are loaded only when there is something to do (resumed runs)
        use_bi = any(op in BIREFNET_OPERATORS for ops in need.values() for op in ops)
        use_mp = any("B2" in ops for ops in need.values())
        bi = BiRefNetSegmenter(device) if use_bi else None
        mp_seg = MediaPipeSegmenter() if use_mp else None
        fovea = None
        try:
            for s in tqdm(range(0, len(todo), batch), desc="segmentation operators"):
                chunk = todo[s : s + batch]
                imgs = [cv2.imread(str(src_dir / rp), cv2.IMREAD_COLOR) for rp in chunk]
                for rp, im in zip(chunk, imgs):
                    if im is None:
                        log.warning("cannot read %s", src_dir / rp)
                        missing.add(str(src_dir / rp))
                keep = [(rp, im) for rp, im in zip(chunk, imgs) if im is not None]
                if not keep:
                    continue
                # BiRefNet masks only for the crops that still need a BiRefNet output
                alphas: list = [None] * len(keep)
                bi_idx = [
                    i
                    for i, (rp, _) in enumerate(keep)
                    if set(need[rp]) & set(BIREFNET_OPERATORS)
                ]
                if bi is not None and bi_idx:
                    for i, a in zip(bi_idx, bi.masks([keep[i][1] for i in bi_idx])):
                        alphas[i] = a
                for (rp, im), a in zip(keep, alphas):
                    outs = {}
                    for op in need[rp]:
                        if op == "B2":
                            outs[op] = composite(im, mp_seg.mask(im))
                        elif op == "C2":
                            if fovea is None or fovea.shape[0] != im.shape[0]:
                                fovea = fovea_mask(im.shape[0])
                            outs[op] = foveate(composite(im, a), fovea)
                        else:
                            outs[op] = apply_with_mask(im, op, a)
                    for op, out in outs.items():
                        p = Path(dst_dirs[op]) / rp
                        p.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(p), out)
                        written[op] += 1
        finally:
            if mp_seg is not None:
                mp_seg.close()
    for op, k in skipped.items():
        if k:
            log.info("%s: skipped %d existing outputs", op, k)
    if strict:
        check_missing(sorted(missing), len(rel_paths))
    return written
