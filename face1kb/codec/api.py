# SPDX-License-Identifier: MIT
"""Public API of the face1kb codecs.

Example
-------
>>> import face1kb
>>> codec = face1kb.load("accurate", device="cuda")      # doctest: +SKIP
>>> data = codec.encode(img, budget=1024)                 # doctest: +SKIP
>>> img_hat = codec.decode(data)                          # doctest: +SKIP

``img`` is an ``H x W x 3`` ``uint8`` RGB array of a square face crop aligned to the
ArcFace 5-point template (the benchmark uses 64, 96, 112, 168 and 224 px); ``data`` is
the self-describing container (:mod:`face1kb.codec.container`).

Bitstreams are device-specific: the entropy coder's CDF tables are rebuilt at run
time and differ between CPU and CUDA builds, so a stream is only guaranteed to decode
correctly on the same kind of device and software stack that produced it. The paper
bitstreams were produced and verified on CUDA (Turing, torch 2.10, compressai 1.2.8,
timm 1.0.24).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import torch

from . import budget as _budget
from . import container
from .resize import pad_to_multiple
from .variants import build_model

logger = logging.getLogger(__name__)

#: Version of the released weight-file format (``face1kb.format_version`` metadata).
WEIGHTS_FORMAT_VERSION = "1"

_CPU_WARNING = (
    "face1kb codec running on CPU: bitstreams are device-specific. CPU-encoded "
    "streams are self-consistent on CPU, but they differ from CUDA-encoded streams, "
    "and CUDA-encoded streams (including the paper bitstreams) do not decode "
    "correctly on CPU. Use a CUDA device for interchange."
)


def set_reproducible_backend() -> None:
    """Disable TF32 in cuDNN convolutions and matmuls and cuDNN autotuning.

    The released bitstreams were produced with full-fp32 convolutions (TF32 does not
    exist on the Turing GPUs used). On Ampere and newer GPUs PyTorch enables TF32 in
    cuDNN by default, which changes the hyperprior outputs that the entropy coder
    depends on; disabling it keeps encoder and decoder numerics in fp32. This sets
    global ``torch.backends`` flags.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _resolve_device(device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available; pass device='cpu' to run on the "
            "CPU (bitstreams are then not interchangeable with CUDA ones)"
        )
    return dev


def load_state_dict_file(path: str | Path) -> tuple[dict, dict]:
    """Load a safetensors weight file; return ``(state_dict, metadata)``."""
    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"weight file not found: {path}")
    if path.stat().st_size < 1024:
        head = path.read_bytes()[:100]
        if head.startswith(b"version https://git-lfs"):
            raise RuntimeError(
                f"{path} is a git-lfs pointer, not the weights; run `git lfs pull`"
            )
    with safe_open(str(path), framework="pt") as f:
        meta = dict(f.metadata() or {})
    return load_file(str(path), device="cpu"), meta


def _check_default_architecture(path, meta: dict, variant: str) -> None:
    """Refuse weight files of non-default architectures (training ablations).

    :func:`load` builds the released configurations only; ACCURATE files exported
    from a ``--no-side-stream`` or ``--anchor`` training run record that in their
    metadata.
    """
    if variant != "accurate":
        return
    no_side = meta.get("face1kb.no_side", "false").lower() == "true"
    anchor = meta.get("face1kb.anchor", "edgeface_s")
    if no_side or anchor != "edgeface_s":
        raise ValueError(
            f"{path} was exported from a non-default architecture (no_side={no_side},"
            f" anchor={anchor!r}); face1kb.load builds only the released FAST / "
            "ACCURATE configurations. Build the network with "
            "face1kb.codec.variants.build_model(variant, no_side, anchor) and load "
            "the state dict from face1kb.codec.api.load_state_dict_file instead."
        )


class Codec:
    """A loaded face1kb codec (use :func:`load` to construct one).

    Attributes
    ----------
    variant : str
        ``"fast"`` or ``"accurate"``.
    net : torch.nn.Module
        The underlying codec network (eval mode).
    device : torch.device
        Device the network runs on.
    metadata : dict
        Metadata stored in the weight file.

    Notes
    -----
    Instances are stateful (per-gain CDF cache, decoded identity code) and are not
    thread-safe; use one instance per thread or process.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        variant: str,
        device: torch.device,
        metadata: dict | None = None,
    ):
        self.net = net
        self.variant = variant
        self.device = device
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Codec(variant={self.variant!r}, device={str(self.device)!r})"

    # ------------------------------------------------------------------ encode
    def _to_tensor(self, img) -> torch.Tensor:
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected an H x W x 3 RGB image, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            raise ValueError(f"expected a uint8 image, got dtype {arr.dtype}")
        h, w = arr.shape[:2]
        if h != w:
            raise ValueError(f"expected a square crop, got {h} x {w}")
        if not 16 <= h <= 65535:
            raise ValueError(f"unsupported crop size {h} px")
        if not 64 <= h <= 256:
            warnings.warn(
                f"{h} px is outside the trained range 64-256 px", stacklevel=4
            )
        x = np.ascontiguousarray(arr, dtype=np.float32) / 255.0
        return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def encode(
        self,
        img,
        budget: int = 1024,
        *,
        paper_compat: bool = False,
        return_info: bool = False,
        overflow: str | None = None,
    ):
        """Encode an aligned square RGB crop to at most ``budget`` bytes.

        Parameters
        ----------
        img
            ``H x W x 3`` ``uint8`` RGB array (``H == W``), e.g. a 112 or 224 px crop
            aligned to the ArcFace 5-point template.
        budget
            Byte budget of the container.
        paper_compat
            Reproduce the paper bitstreams byte for byte, including the paper's
            budget accounting that ignores the 4-byte raw-geometry trailer at
            resolutions without a bucket code (e.g. 96 and 168 px). With the default
            ``False`` the trailer is reserved as well.
        return_info
            Also return an info dict (``fitted``, ``over_budget``, ``identity_only``,
            ``within_budget``, ``rate_index``, ``gain``, ``bytes``). ``fitted`` means
            that a spatial stream fit the search target; with ``paper_compat=True``
            it does not imply ``len(data) <= budget`` (the 4-byte raw-geometry
            trailer at 96/168 px is not reserved), so budget-fit shares must be
            computed from ``within_budget`` or the byte length.
        overflow
            What to emit when even the lowest gain does not fit: ``"identity"``
            (identity-only container, decodes to a black frame; the default unless
            ``paper_compat``), ``"floor"`` (lowest-gain stream flagged
            ``over_budget``, exceeds the budget; the paper behaviour and the default
            with ``paper_compat=True``) or ``"error"`` (raise
            :class:`~face1kb.codec.budget.BudgetError`). The identity-only fallback
            is common for FAST at 168/224 px with 512 B (see ``docs/codec.md``); it
            issues an :class:`~face1kb.codec.budget.IdentityOnlyWarning`.

        Returns
        -------
        bytes or tuple[bytes, dict]
            The container, and the info dict if ``return_info``. With
            ``paper_compat=False`` the container never exceeds ``budget`` unless
            ``overflow="floor"`` is requested.

        Raises
        ------
        ValueError
            For a malformed image, ``budget < 1``, or, with ``paper_compat=False``,
            a budget below the smallest (identity-only) container: ``3 +``
            side-channel bytes, plus 4 at raw-geometry resolutions (3/7 B for FAST,
            11/15 B for ACCURATE).
        ~face1kb.codec.budget.BudgetError
            With ``overflow="error"`` when no spatial stream fits.
        """
        budget = int(budget)
        x = self._to_tensor(img)
        res = int(x.shape[-1])
        xp, _ = pad_to_multiple(x)
        side = getattr(self.net, "side", None)
        sc = side.encode(x) if side is not None else b""
        data, info = _budget.encode_to_budget(
            self.net,
            xp,
            budget,
            res,
            sidechannel=sc,
            paper_compat=paper_compat,
            overflow=overflow,
        )
        if info["identity_only"]:
            warnings.warn(
                f"face1kb-{self.variant}: no spatial stream fits {budget} B at "
                f"{res} px; emitted the identity-only container, which decodes to a "
                "black frame (use a larger budget or overflow='floor')",
                _budget.IdentityOnlyWarning,
                stacklevel=3,  # the caller of encode (past the no_grad wrapper)
            )
        return (data, info) if return_info else data

    # ------------------------------------------------------------------ decode
    @torch.no_grad()
    def decode_tensor(
        self, data: bytes
    ) -> tuple[torch.Tensor | None, container.Header]:
        """Decode to a ``(1, 3, res, res)`` float tensor in ``[0, 1]`` (or ``None``).

        ``None`` is returned for identity-only containers. The header is returned as
        the second element.
        """
        h = container.unpack(data)
        expected = container.VARIANT_ACCURATE if self.variant == "accurate" else 0
        if h.variant_id not in (expected, container.VARIANT_IDENTITY_ONLY):
            raise ValueError(
                f"container was produced by variant "
                f"{container.VARIANT_NAMES.get(h.variant_id, h.variant_id)!r}, "
                f"this codec is {self.variant!r}"
            )
        return _budget.decode_from_container(self.net, data, device=self.device)

    @torch.no_grad()
    def decode(self, data: bytes) -> np.ndarray:
        """Decode a container to an ``H x W x 3`` ``uint8`` RGB image.

        Identity-only containers decode to a black frame of the stored size.
        """
        x_hat, h = self.decode_tensor(data)
        if x_hat is None:
            return np.zeros((h.res, h.res, 3), dtype=np.uint8)
        x = x_hat[..., : h.res, : h.res][0].permute(1, 2, 0).cpu().numpy()
        return (x * 255).round().astype(np.uint8)

    # ------------------------------------------------------------------ misc
    @staticmethod
    def info(data: bytes) -> dict:
        """Parse a container header without decoding (see :func:`inspect`)."""
        return inspect(data)

    @property
    def num_parameters(self) -> int:
        """Number of parameters of the network (including the frozen anchor)."""
        return sum(p.numel() for p in self.net.parameters())


def inspect(data: bytes) -> dict:
    """Return the header fields and payload sizes of a container (no decoding)."""
    return container.describe(data)


def load(
    variant: str = "fast",
    device: str | torch.device = "cuda",
    weights_dir: str | Path | None = None,
    *,
    weights_path: str | Path | None = None,
    reproducible: bool = True,
) -> Codec:
    """Load a released face1kb codec.

    Parameters
    ----------
    variant
        ``"fast"`` (face1kb-FAST, "Ours-FAST" in the paper) or ``"accurate"``
        (face1kb-ACCURATE, "Ours-ACCURATE").
    device
        Torch device. CUDA is required to decode the paper bitstreams and to produce
        streams that decode on other CUDA machines; on CPU a warning is issued.
    weights_dir
        Folder with the ``face1kb_<variant>.safetensors`` files (default
        ``FACE1KB_WEIGHTS_DIR``, i.e. ``<repo>/weights``).
    weights_path
        Explicit weight file (overrides ``weights_dir``).
    reproducible
        Call :func:`set_reproducible_backend` (fp32 convolutions, no autotuning).

    Returns
    -------
    Codec
        The loaded codec, in eval mode.
    """
    from face1kb import config  # noqa: PLC0415

    variant = variant.lower()
    if variant not in config.CODEC_VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r}; choose from {config.CODEC_VARIANTS}"
        )
    dev = _resolve_device(device)
    if weights_path is None:
        root = Path(weights_dir) if weights_dir is not None else config.WEIGHTS_DIR
        weights_path = root / config.WEIGHT_FILES[variant]
        if not Path(weights_path).is_file() and weights_dir is None:
            hint = config.weights_dir_hint()
            if hint:
                raise FileNotFoundError(
                    f"weight file not found: {weights_path}. {hint}"
                )
    state, meta = load_state_dict_file(weights_path)
    if meta.get("face1kb.variant", variant) != variant:
        raise ValueError(
            f"{weights_path} holds variant {meta['face1kb.variant']!r}, not {variant!r}"
        )
    _check_default_architecture(weights_path, meta, variant)
    if reproducible:
        set_reproducible_backend()
    net = build_model(variant)
    net.load_state_dict(state, strict=True)
    net = net.eval().to(dev)
    if dev.type == "cpu":
        warnings.warn(_CPU_WARNING, stacklevel=2)
    logger.info("loaded face1kb-%s from %s on %s", variant, weights_path, dev)
    return Codec(net, variant, dev, meta)


_CODEC_CACHE: dict[tuple, Codec] = {}


def decode(
    data: bytes,
    device: str | torch.device = "cuda",
    weights_dir: str | Path | None = None,
) -> np.ndarray:
    """Decode any face1kb container, loading the variant named in its header.

    Loaded codecs are cached per ``(variant, device, weights_dir)``. Identity-only
    containers decode to a black frame without loading a model.
    """
    h = container.unpack(data)
    if h.variant_id == container.VARIANT_IDENTITY_ONLY:
        return np.zeros((h.res, h.res, 3), dtype=np.uint8)
    if h.variant_id not in (container.VARIANT_FAST, container.VARIANT_ACCURATE):
        raise ValueError(f"unsupported container variant id {h.variant_id}")
    variant = container.VARIANT_NAMES[h.variant_id]
    key = (variant, str(torch.device(device)), str(weights_dir))
    codec = _CODEC_CACHE.get(key)
    if codec is None:
        codec = _CODEC_CACHE[key] = load(variant, device, weights_dir)
    return codec.decode(data)
