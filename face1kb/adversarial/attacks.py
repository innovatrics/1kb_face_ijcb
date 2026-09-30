# SPDX-License-Identifier: MIT
"""No-box l-inf attacks on aligned face crops (compression-as-defence study).

All three attacks are *no-box*: they never access or query the face matchers that
are evaluated afterwards. They take ``(N, H, W, 3)`` uint8 RGB crops (the study uses
the 112 px aligned crops), add a perturbation bounded by ``eps`` in the l-inf norm on
the ``[0, 1]`` scale, and return uint8 crops of the same shape.

* :func:`hfc_attack` -- training-free high-frequency-component injection after
  Zhang et al. 2022 (arXiv:2203.04607), as a simplified spatial-domain
  re-implementation: a tiled block-noise pattern is high-passed, its sign is added
  at the full budget, and the crop's own high frequencies are attenuated. NumPy and
  OpenCV only; runs on the CPU.
* :func:`clip_surrogate_attack` -- untargeted I-FGSM with a random start that pushes
  the OpenAI CLIP ViT-B/32 image features of the crop away from those of the clean
  crop (the CLIP-surrogate idea of MF-CLIP, arXiv:2307.06608, without its margin
  fine-tuning).
* :func:`li_ae_attack` / :class:`LiAEAttack` -- the prototypical auto-encoder attack
  of Li et al. 2020 (arXiv:2012.02525): an I-FGSM push of the proxy's bottleneck
  code, followed by an Intermediate-Level Attack refinement (ILA, Huang et al. 2019)
  on the proxy's mid layer, against :class:`face1kb.adversarial.proxy.ProxyAE`.

These are independent implementations; no code of the cited authors is used.

Reproducibility: every attack is seeded (``seed=0`` in the paper). HFC draws its
noise image by image from one NumPy generator, so the first ``k`` crops of a run do
not depend on how many crops follow. The CLIP and Li-AE attacks draw their random
starts per batch (``batch_size=64``); a crop's result is therefore reproducible only
with the same seed, batch size and batch position (run from the same first crop).
By default the starts come from a ``torch.Generator`` on the attack device. On CUDA
that stream depends on the GPU model (its number of multiprocessors), so the
``cuda_blocks`` option recomputes the stream of a given GPU on any device; the paper
values are in :data:`PAPER_CUDA_BLOCKS`. CLIP runs in fp16 on CUDA (the
``clip.load`` default) and in fp32 on the CPU, so its results depend on the device;
see ``docs/adversarial.md``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from .proxy import ProxyAE

logger = logging.getLogger(__name__)

#: Attack strengths of the paper (l-inf budget on the [0, 1] scale).
DEFAULT_EPS: tuple[float, ...] = (0.03, 0.06, 0.1)
#: Resolution of the crops attacked in the paper.
ATTACK_RES = 112
#: Default CLIP surrogate model.
CLIP_MODEL = "ViT-B/32"
#: CLIP image-normalisation constants (``clip.load`` preprocessing).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
#: CUDA launch blocks (``cuda_blocks``) whose random-start stream produced the paper
#: crop sets, by ``(dataset, attack)``: 272 = GeForce RTX 2080 Ti (68 SMs), 288 =
#: Quadro RTX 6000 (72 SMs).
PAPER_CUDA_BLOCKS: dict[tuple[str, str], int] = {
    ("colorferet", "clip"): 272,
    ("colorferet", "liae"): 272,
    ("kk", "clip"): 288,
    ("kk", "liae"): 272,
}


def eps_tag(eps: float) -> str:
    """Three-digit tag of an attack strength: ``0.03 -> "003"``, ``0.1 -> "010"``."""
    return f"{int(round(eps * 100)):03d}"


def adv_suffix(attack: str, eps: float) -> str:
    """Alignment suffix of an adversarial crop set, e.g. ``"_adv_hfc_006"``.

    Adversarial crops are stored as an ordinary aligned-crop variant, e.g.
    ``face1kb.config.aligned_dir(dataset, 112, adv_suffix("hfc", 0.06))``, so the
    compression and embedding pipelines treat them like any other crop set.
    """
    if attack not in ATTACKS:
        raise ValueError(f"unknown attack {attack!r}; choose from {sorted(ATTACKS)}")
    return f"_adv_{attack}_{eps_tag(eps)}"


def step_size(eps: float, alpha: float | None = None) -> float:
    """I-FGSM step of the CLIP and Li-AE attacks: ``alpha`` or ``max(eps/4, 1/255)``."""
    return alpha if alpha is not None else max(eps / 4.0, 1.0 / 255.0)


def _resolve_device(
    device: str | torch.device | None, module: torch.nn.Module | None = None
) -> torch.device:
    """Return ``device``; by default the device of ``module``, else CUDA if present."""
    import torch  # noqa: PLC0415

    if device is not None:
        return torch.device(device)
    if module is not None:
        for p in module.parameters():
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _check_crops(imgs: np.ndarray) -> np.ndarray:
    imgs = np.asarray(imgs)
    if imgs.ndim != 4 or imgs.shape[-1] != 3 or imgs.dtype != np.uint8:
        raise ValueError(
            f"expected (N, H, W, 3) uint8 RGB crops, got {imgs.shape} {imgs.dtype}"
        )
    return imgs


def _to_tensor01(batch: np.ndarray, dev: torch.device) -> torch.Tensor:
    import torch  # noqa: PLC0415

    x = torch.from_numpy(batch.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    return x.to(dev)


def _to_uint8(x: torch.Tensor) -> np.ndarray:
    return (
        (x.detach().permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    )


def _project(adv: torch.Tensor, clean: torch.Tensor, eps: float) -> torch.Tensor:
    """Project ``adv`` into the l-inf ball of radius ``eps`` and onto [0, 1]."""
    import torch  # noqa: PLC0415

    return torch.min(torch.max(adv, clean - eps), clean + eps).clamp_(0, 1)


class _RandomStarts:
    """Seeded stream of PGD random starts (uniform points of the l-inf ball).

    At ``adv == clean`` the feature-distance losses are at their minimum and their
    gradient is exactly zero; the random start breaks that symmetry.

    With ``cuda_blocks=None`` the noise comes from a ``torch.Generator`` on
    ``device`` seeded with ``seed``; this equals ``torch.manual_seed(seed)``
    followed by draws from the default generator of that device. With an integer,
    the noise is recomputed by :func:`face1kb.adversarial._philox.cuda_uniform` as
    torch's CUDA ``uniform_`` writes it on a GPU that launches ``cuda_blocks``
    blocks, whatever the attack device.
    """

    def __init__(self, seed: int, device: torch.device, cuda_blocks: int | None = None):
        self.seed = int(seed)
        self.cuda_blocks = None if cuda_blocks is None else int(cuda_blocks)
        self.offset = 0
        self.gen = None
        if self.cuda_blocks is None:
            import torch  # noqa: PLC0415

            self.gen = torch.Generator(device=device).manual_seed(self.seed)
        elif self.cuda_blocks <= 0:
            raise ValueError(f"cuda_blocks must be positive, got {cuda_blocks}")

    def __call__(self, clean: torch.Tensor, eps: float) -> torch.Tensor:
        """Return ``clean + noise`` with noise uniform in ``[-eps, eps)``."""
        import torch  # noqa: PLC0415

        if self.gen is not None:
            noise = torch.empty_like(clean).uniform_(-eps, eps, generator=self.gen)
            return clean + noise
        from . import _philox  # noqa: PLC0415

        if not (
            clean.is_contiguous()
            or clean.is_contiguous(memory_format=torch.channels_last)
        ):
            raise ValueError("the emulated random start needs a dense batch tensor")
        n = clean.numel()
        values = _philox.cuda_uniform(
            n, -eps, eps, self.seed, self.offset, self.cuda_blocks
        )
        self.offset += _philox.offset_increment(n, self.cuda_blocks)
        # values are in memory order; lay them out with the strides of ``clean``
        noise = torch.from_numpy(values).as_strided(clean.shape, clean.stride())
        return clean + noise.to(clean.device)


# ----------------------------------------------------------------------------- HFC
def _linf_clip(adv: np.ndarray, clean: np.ndarray, eps: float) -> np.ndarray:
    """Project ``adv`` into the l-inf ball of radius ``eps`` and onto [0, 1]."""
    return np.clip(np.clip(adv, clean - eps, clean + eps), 0.0, 1.0)


def hfc_attack(
    imgs: np.ndarray,
    eps: float,
    tile: int = 4,
    suppress: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Training-free high-frequency-component attack (l-inf ``eps``).

    For each crop, a ``tile x tile`` block pattern of Gaussian noise is tiled over
    the image (regionally homogeneous and repeating), high-passed by subtracting its
    Gaussian blur (``sigma = tile``) and reduced to its sign. The crop's own high
    frequencies are attenuated towards its Gaussian low-pass by ``suppress``, the
    signed pattern is added with amplitude ``eps``, and the result is projected into
    the l-inf ball around the clean crop.

    Parameters
    ----------
    imgs
        ``(N, H, W, 3)`` uint8 RGB crops.
    eps
        l-inf budget on the [0, 1] scale (the paper uses 0.03, 0.06 and 0.10).
    tile
        Side of the homogeneous noise blocks and sigma of the high-pass blur (px).
    suppress
        Fraction of the crop's own high-frequency content removed before the
        injection (0 = inject only).
    seed
        Seed of the NumPy generator; the noise is drawn crop by crop in order.

    Returns
    -------
    np.ndarray
        ``(N, H, W, 3)`` uint8 adversarial crops.
    """
    import cv2  # noqa: PLC0415

    imgs = _check_crops(imgs)
    rng = np.random.default_rng(seed)
    out = np.empty_like(imgs)
    for k in range(len(imgs)):
        clean = imgs[k].astype(np.float32) / 255.0
        h, w = clean.shape[:2]
        # Regionally homogeneous, repeating base pattern: a small block grid tiled up.
        bh, bw = h // tile + 1, w // tile + 1
        base = rng.standard_normal((bh, bw, 3)).astype(np.float32)
        base = np.kron(base, np.ones((tile, tile, 1), np.float32))[:h, :w]
        # High-pass: subtract a blur at the block scale.
        low = cv2.GaussianBlur(base, (0, 0), sigmaX=float(tile))
        hf = base - low
        hf /= np.abs(hf).max() + 1e-8
        adv = clean.copy()
        if suppress > 0:  # attenuate the crop's own high frequencies
            clean_low = cv2.GaussianBlur(clean, (0, 0), sigmaX=float(tile))
            adv = clean - suppress * (clean - clean_low)
        adv = adv + eps * np.sign(hf)  # inject at the full l-inf budget
        out[k] = np.round(_linf_clip(adv, clean, eps) * 255.0).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------- CLIP
def load_clip(
    model_name: str = CLIP_MODEL,
    device: str | torch.device | None = None,
    download_root: str | Path | None = None,
):
    """Load a frozen OpenAI CLIP model (``pip install face1kb[adversarial]``).

    Parameters
    ----------
    model_name
        CLIP model name (default ``"ViT-B/32"``).
    device
        Torch device (default CUDA if available). ``clip.load`` keeps the weights in
        fp16 on CUDA and converts them to fp32 on the CPU.
    download_root
        Cache folder of the CLIP checkpoints (default
        ``FACE1KB_MODELS_ROOT/clip``); the file is downloaded on first use and
        checked against the SHA-256 recorded by the ``clip`` package.

    Returns
    -------
    torch.nn.Module
        The CLIP model in eval mode with frozen parameters.
    """
    try:
        import clip  # noqa: PLC0415
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the CLIP attack needs OpenAI CLIP: pip install 'face1kb[adversarial]'"
        ) from err
    from face1kb import config  # noqa: PLC0415

    dev = _resolve_device(device)
    root = Path(download_root) if download_root else config.models_dir("clip")
    net, _ = clip.load(model_name, device=dev, download_root=str(root))
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def clip_surrogate_attack(
    imgs: np.ndarray,
    eps: float,
    steps: int = 10,
    alpha: float | None = None,
    device: str | torch.device | None = None,
    model_name: str = CLIP_MODEL,
    seed: int = 0,
    batch_size: int = 64,
    model=None,
    download_root: str | Path | None = None,
    cuda_blocks: int | None = None,
) -> np.ndarray:
    """Untargeted I-FGSM in CLIP image-feature space (l-inf ``eps``).

    The crops are resized bilinearly to the CLIP input resolution (224 px for
    ViT-B/32) inside the differentiable graph and normalised with the CLIP
    constants. Starting from a uniform random point of the ``eps``-ball, each step
    ascends ``1 - cos(f(adv), f(clean))`` by ``step_size(eps, alpha)`` times the
    gradient sign and projects back into the ball.

    Parameters
    ----------
    imgs
        ``(N, H, W, 3)`` uint8 RGB crops.
    eps
        l-inf budget on the [0, 1] scale.
    steps
        Number of I-FGSM steps (10 in the paper).
    alpha
        Step size; default ``max(eps / 4, 1 / 255)``.
    device
        Torch device (default: the device of ``model`` if given, else CUDA if
        available).
    model_name
        CLIP model (ignored when ``model`` is given).
    seed
        Seed of the random starts (the global torch RNG is not touched).
    batch_size
        Crops per batch (64 in the paper; it changes the random starts).
    model
        An already loaded CLIP model (see :func:`load_clip`), e.g. to reuse it
        across several ``eps``.
    download_root
        CLIP checkpoint cache (see :func:`load_clip`).
    cuda_blocks
        ``None`` (default): draw the random starts with a ``torch.Generator`` on
        the attack device. An integer: recompute them as torch's CUDA
        ``uniform_`` draws them on a GPU that launches this many blocks (SMs x
        max threads per SM / 256), on any device; see :data:`PAPER_CUDA_BLOCKS`.

    Returns
    -------
    np.ndarray
        ``(N, H, W, 3)`` uint8 adversarial crops.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as nnf  # noqa: PLC0415

    imgs = _check_crops(imgs)
    dev = _resolve_device(device, model)
    net = model if model is not None else load_clip(model_name, dev, download_root)
    mean = torch.tensor(CLIP_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=dev).view(1, 3, 1, 1)
    res = net.visual.input_resolution
    step = step_size(eps, alpha)

    def feats(x01: torch.Tensor) -> torch.Tensor:
        x = nnf.interpolate(x01, size=res, mode="bilinear", align_corners=False)
        return net.encode_image((x - mean) / std).float()

    starts = _RandomStarts(seed, dev, cuda_blocks)
    out = np.empty_like(imgs)
    for s in range(0, len(imgs), batch_size):
        clean = _to_tensor01(imgs[s : s + batch_size], dev)
        with torch.no_grad():
            f0 = feats(clean)
        adv = starts(clean, eps).clamp_(0, 1)
        for _ in range(steps):
            adv.requires_grad_(True)
            loss = (1 - nnf.cosine_similarity(feats(adv), f0)).mean()
            (grad,) = torch.autograd.grad(loss, adv)
            adv = _project(adv.detach() + step * grad.sign(), clean, eps)
        out[s : s + batch_size] = _to_uint8(adv)
    return out


# --------------------------------------------------------------------------- Li-AE
class LiAEAttack:
    """Prototypical auto-encoder no-box attack (Li et al. 2020) against a proxy.

    The crafting has two stages per batch:

    1. I-FGSM from a random start that maximises the squared distance of the
       proxy's bottleneck ``code`` to that of the clean crop; the resulting
       displacement of the ``mid`` features, normalised per crop, is the guide
       direction.
    2. ILA refinement: from a new random start, I-FGSM that maximises the projection
       of the ``mid`` displacement onto the guide direction.

    Parameters
    ----------
    proxy
        A :class:`~face1kb.adversarial.proxy.ProxyAE` (switched to eval mode and
        frozen in place), a weight-file path, or ``None`` for the released
        ``weights/liae_proxy.safetensors``.
    device
        Torch device (default: the device of a given ``ProxyAE``, else CUDA if
        available).
    """

    def __init__(
        self,
        proxy: ProxyAE | str | Path | None = None,
        device: str | torch.device | None = None,
    ):
        from .proxy import ProxyAE, load_proxy  # noqa: PLC0415

        if isinstance(proxy, ProxyAE):
            self.device = _resolve_device(device, proxy)
            net = proxy.to(self.device).eval()
            for p in net.parameters():
                p.requires_grad_(False)
        else:
            self.device = _resolve_device(device)
            net = load_proxy(proxy, device=self.device)
        self.net = net

    def craft(
        self,
        imgs: np.ndarray,
        eps: float,
        steps: int = 10,
        ila_steps: int = 10,
        alpha: float | None = None,
        seed: int = 0,
        batch_size: int = 64,
        cuda_blocks: int | None = None,
    ) -> np.ndarray:
        """Craft adversarial crops (l-inf ``eps``).

        Parameters
        ----------
        imgs
            ``(N, 112, 112, 3)`` uint8 RGB crops.
        eps
            l-inf budget on the [0, 1] scale.
        steps
            I-FGSM steps of stage 1 (10 in the paper).
        ila_steps
            ILA steps of stage 2 (10 in the paper).
        alpha
            Step size of both stages; default ``max(eps / 4, 1 / 255)``.
        seed
            Seed of the random starts (the global torch RNG is not touched).
        batch_size
            Crops per batch (64 in the paper; it changes the random starts).
        cuda_blocks
            Source of the random starts, as in :func:`clip_surrogate_attack`.

        Returns
        -------
        np.ndarray
            ``(N, 112, 112, 3)`` uint8 adversarial crops.
        """
        import torch  # noqa: PLC0415

        imgs = _check_crops(imgs)
        net, dev = self.net, self.device
        step = step_size(eps, alpha)
        starts = _RandomStarts(seed, dev, cuda_blocks)
        out = np.empty_like(imgs)
        for s in range(0, len(imgs), batch_size):
            clean = _to_tensor01(imgs[s : s + batch_size], dev)
            with torch.no_grad():
                f0 = net.encode(clean)
                code0, mid0 = f0["code"], f0["mid"]
            # Stage 1: I-FGSM push of the bottleneck code.
            adv = _project(starts(clean, eps), clean, eps)
            for _ in range(steps):
                adv.requires_grad_(True)
                code = net.encode(adv)["code"]
                loss = ((code - code0) ** 2).flatten(1).sum(1).mean()
                (grad,) = torch.autograd.grad(loss, adv)
                adv = _project(adv.detach() + step * grad.sign(), clean, eps)
            # Guide direction of ILA: the normalised mid-layer displacement.
            with torch.no_grad():
                ref = (net.encode(adv)["mid"] - mid0).flatten(1)
                ref = ref / (ref.norm(dim=1, keepdim=True) + 1e-12)
            # Stage 2: ILA, maximise the mid displacement along the guide.
            adv = _project(starts(clean, eps), clean, eps)
            for _ in range(ila_steps):
                adv.requires_grad_(True)
                mid = net.encode(adv)["mid"]
                proj = ((mid - mid0).flatten(1) * ref).sum(1)
                (grad,) = torch.autograd.grad(proj.mean(), adv)
                adv = _project(adv.detach() + step * grad.sign(), clean, eps)
            out[s : s + batch_size] = _to_uint8(adv)
        return out


def li_ae_attack(
    imgs: np.ndarray,
    eps: float,
    proxy: ProxyAE | str | Path | None = None,
    device: str | torch.device | None = None,
    steps: int = 10,
    ila_steps: int = 10,
    alpha: float | None = None,
    seed: int = 0,
    batch_size: int = 64,
    cuda_blocks: int | None = None,
) -> np.ndarray:
    """Li-AE attack with a proxy (see :class:`LiAEAttack`); returns uint8 crops."""
    return LiAEAttack(proxy, device=device).craft(
        imgs,
        eps,
        steps=steps,
        ila_steps=ila_steps,
        alpha=alpha,
        seed=seed,
        batch_size=batch_size,
        cuda_blocks=cuda_blocks,
    )


def paper_cuda_blocks(dataset: str, attack: str) -> int | None:
    """``cuda_blocks`` that reproduces the random starts of a paper crop set.

    Returns the :data:`PAPER_CUDA_BLOCKS` entry of ``(dataset, attack)``, or
    ``None`` for HFC (which draws no random start on the GPU).
    """
    if attack not in ATTACKS:
        raise ValueError(f"unknown attack {attack!r}; choose from {sorted(ATTACKS)}")
    if attack == "hfc":
        return None
    try:
        return PAPER_CUDA_BLOCKS[(dataset, attack)]
    except KeyError:
        raise ValueError(f"no paper crop set for {(dataset, attack)}") from None


#: Attack registry: name -> ``callable(imgs_uint8, eps, **kwargs) -> adv_uint8``.
ATTACKS: dict[str, Callable[..., np.ndarray]] = {
    "hfc": hfc_attack,
    "clip": clip_surrogate_attack,
    "liae": li_ae_attack,
}


def craft(attack: str, imgs: np.ndarray, eps: float, **kwargs) -> np.ndarray:
    """Run a registered attack by name (``"hfc"``, ``"clip"`` or ``"liae"``).

    ``kwargs`` are passed to the attack. Options that an attack does not use are
    dropped, so one call works for all three: ``model`` (the loaded CLIP model) is
    used only by ``"clip"`` and dropped for the other two, and the options of the
    gradient attacks that have no meaning for the CPU-only HFC attack (``device``,
    ``batch_size`` and ``cuda_blocks=None``) are dropped for ``"hfc"``.
    """
    if attack not in ATTACKS:
        raise ValueError(f"unknown attack {attack!r}; choose from {sorted(ATTACKS)}")
    if attack != "clip":
        kwargs.pop("model", None)
    if attack == "hfc":
        if kwargs.pop("cuda_blocks", None) is not None:
            raise ValueError("the HFC attack draws no CUDA random start")
        kwargs.pop("device", None)
        kwargs.pop("batch_size", None)
    logger.info("crafting %s (eps=%s) on %d crops", attack, eps, len(imgs))
    return ATTACKS[attack](imgs, eps, **kwargs)
