# SPDX-License-Identifier: MIT
"""Proxy auto-encoder of the Li-AE no-box attack, with its weight-file I/O.

The Li-AE attack (Li, Guo and Chen, "Practical No-box Adversarial Attacks against
DNNs", NeurIPS 2020, arXiv:2012.02525) crafts perturbations against a small
auto-encoder that the attacker trains on their own face images; the victim matcher is
never queried. :class:`ProxyAE` is that surrogate: a plain convolutional
encoder-decoder for 112 x 112 RGB crops in ``[0, 1]``.

* encoder: four stages of ``Conv2d(k=4, s=2, p=1) -> BatchNorm2d -> LeakyReLU(0.2)``
  with ``w, 2w, 4w, 8w`` channels (112 -> 56 -> 28 -> 14 -> 7 px);
* decoder: three stages of ``ConvTranspose2d(k=4, s=2, p=1) -> BatchNorm2d -> ReLU``
  and a final ``ConvTranspose2d`` to 3 channels followed by a sigmoid.

With the default width ``w = 32`` the model has 1,381,443 parameters. The attack uses
two encoder features: ``mid`` (the ``2w x 28 x 28`` output of the second stage, the
intermediate layer of the ILA refinement) and ``code`` (the ``8w x 7 x 7``
bottleneck, the target of the I-FGSM feature push).

Provenance: this architecture and the training recipe in
:mod:`face1kb.adversarial.train` are an independent implementation written for this
project. No code of the authors' reference implementation
(github.com/qizhangli/nobox-attacks, which carries no licence and uses a different,
ResNet-block auto-encoder with several decoder heads) is used, copied or vendored.

The released proxy weights (``weights/liae_proxy.safetensors``, CC BY-NC-SA 4.0) are
a net-only safetensors file: fp32 parameters and batch-norm statistics plus the int64
batch-norm step counters, loaded with ``strict=True``. See ``weights/README.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

#: Input resolution of the proxy (the aligned 112 px crops).
RES = 112
#: Default channel width of the released proxy.
DEFAULT_WIDTH = 32
#: File name of the released proxy weights inside ``FACE1KB_WEIGHTS_DIR``.
PROXY_FILE = "liae_proxy.safetensors"
#: Version of the proxy weight-file format (``face1kb.format_version`` metadata).
PROXY_FORMAT_VERSION = "1"
#: Architecture name recorded in the weight-file metadata.
ARCHITECTURE = "ProxyAE"


def _down(cin: int, cout: int) -> nn.Sequential:
    """Stride-2 convolution stage that halves the spatial size."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, 2, 1),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _up(cin: int, cout: int, act: bool = True) -> nn.Sequential:
    """Stride-2 transposed-convolution stage that doubles the spatial size."""
    layers: list[nn.Module] = [nn.ConvTranspose2d(cin, cout, 4, 2, 1)]
    if act:
        layers += [nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class ProxyAE(nn.Module):
    """Convolutional auto-encoder surrogate of the Li-AE attack (112 px RGB, [0, 1]).

    Parameters
    ----------
    width
        Channel width ``w`` of the first encoder stage (the stages use ``w, 2w, 4w,
        8w`` channels).

    Notes
    -----
    ``forward`` returns ``(recon, feats)``: the sigmoid reconstruction and the dict
    of :meth:`encode`.
    """

    def __init__(self, width: int = DEFAULT_WIDTH):
        super().__init__()
        w = int(width)
        self.width = w
        self.enc1 = _down(3, w)  # 112 -> 56
        self.enc2 = _down(w, 2 * w)  # 56 -> 28 (ILA layer "mid")
        self.enc3 = _down(2 * w, 4 * w)  # 28 -> 14
        self.enc4 = _down(4 * w, 8 * w)  # 14 -> 7 (bottleneck "code")
        self.dec1 = _up(8 * w, 4 * w)  # 7 -> 14
        self.dec2 = _up(4 * w, 2 * w)  # 14 -> 28
        self.dec3 = _up(2 * w, w)  # 28 -> 56
        self.dec4 = _up(w, 3, act=False)  # 56 -> 112
        self.out = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the encoder only.

        Parameters
        ----------
        x
            ``(N, 3, 112, 112)`` float tensor in ``[0, 1]``.

        Returns
        -------
        dict
            ``{"mid": (N, 2w, 28, 28), "code": (N, 8w, 7, 7)}``.
        """
        mid = self.enc2(self.enc1(x))
        code = self.enc4(self.enc3(mid))
        return {"mid": mid, "code": code}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Encode and decode ``x``; return ``(recon, feats)``."""
        feats = self.encode(x)
        d = self.dec3(self.dec2(self.dec1(feats["code"])))
        return self.out(self.dec4(d)), feats


def count_parameters(net: nn.Module) -> int:
    """Return the number of learnable parameters of ``net``."""
    return sum(p.numel() for p in net.parameters())


def default_proxy_path() -> Path:
    """Path of the released proxy weights (``FACE1KB_WEIGHTS_DIR/liae_proxy...``)."""
    from face1kb import config  # noqa: PLC0415

    return config.WEIGHTS_DIR / PROXY_FILE


def _read_safetensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    if path.stat().st_size < 1024:
        head = path.read_bytes()[:100]
        if head.startswith(b"version https://git-lfs"):
            raise RuntimeError(
                f"{path} is a git-lfs pointer, not the weights; run `git lfs pull`"
            )
    with safe_open(str(path), framework="pt") as f:
        meta = dict(f.metadata() or {})
    return load_file(str(path), device="cpu"), meta


def _read_torch_checkpoint(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Read a ``.pt`` checkpoint: a bare state dict or ``{"model": sd, "width": w}``.

    Loaded with ``weights_only=True`` (no pickled code is executed).
    """
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        meta = {
            f"face1kb.{k}": str(v) for k, v in obj.items() if not isinstance(v, dict)
        }
        return obj["model"], meta
    return obj, {}


def read_proxy_file(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Read a proxy weight file; return ``(state_dict, metadata)``.

    ``.safetensors`` files are read with safetensors; any other suffix is read as a
    torch checkpoint with ``weights_only=True``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"proxy weight file not found: {path}")
    if path.suffix == ".safetensors":
        return _read_safetensors(path)
    return _read_torch_checkpoint(path)


def load_proxy(
    path: str | Path | None = None, device: str | torch.device = "cpu"
) -> ProxyAE:
    """Build :class:`ProxyAE` and load weights with ``strict=True``.

    Parameters
    ----------
    path
        Weight file; default :func:`default_proxy_path` (the released
        ``weights/liae_proxy.safetensors``). A ``.pt`` checkpoint of
        :func:`torch.save` (bare state dict or ``{"model": state_dict, "width": w}``)
        is also accepted.
    device
        Torch device of the returned model.

    Returns
    -------
    ProxyAE
        The model in eval mode with ``requires_grad=False`` on all parameters.
    """
    if path is None:
        path = default_proxy_path()
        if not path.is_file():
            from face1kb import config  # noqa: PLC0415

            hint = config.weights_dir_hint() or "run `git lfs pull` in the checkout"
            raise FileNotFoundError(f"released proxy weights not found: {path}; {hint}")
    path = Path(path)
    state, meta = read_proxy_file(path)
    arch = meta.get("face1kb.architecture", ARCHITECTURE)
    if arch != ARCHITECTURE:
        raise ValueError(f"{path} holds architecture {arch!r}, not {ARCHITECTURE!r}")
    version = meta.get("face1kb.format_version", PROXY_FORMAT_VERSION)
    if version != PROXY_FORMAT_VERSION:
        raise ValueError(
            f"{path} has proxy format version {version!r}; this face1kb reads "
            f"version {PROXY_FORMAT_VERSION!r}"
        )
    width = int(meta.get("face1kb.width", DEFAULT_WIDTH))
    net = ProxyAE(width=width)
    net.load_state_dict(state, strict=True)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    logger.info("loaded Li-AE proxy (width %d) from %s", width, path)
    return net


def save_proxy(
    net: ProxyAE, path: str | Path, metadata: dict[str, object] | None = None
) -> Path:
    """Write ``net`` as a net-only safetensors file with metadata.

    Parameters
    ----------
    net
        The proxy to save (its state dict is copied to the CPU).
    path
        Output ``.safetensors`` file (parent folders are created).
    metadata
        Extra header entries; keys without a ``face1kb.`` prefix get one, values
        are converted to strings. ``format``, ``face1kb.format_version``,
        ``face1kb.architecture``, ``face1kb.width`` and ``face1kb.parameters`` are
        always written.

    Returns
    -------
    Path
        ``path``.
    """
    from safetensors.torch import save_file  # noqa: PLC0415

    path = Path(path)
    meta: dict[str, str] = {}
    for key, value in (metadata or {}).items():
        k = key if key.startswith("face1kb.") or key == "format" else f"face1kb.{key}"
        meta[k] = str(value)
    meta.update(
        {
            "format": "pt",
            "face1kb.format_version": PROXY_FORMAT_VERSION,
            "face1kb.architecture": ARCHITECTURE,
            "face1kb.width": str(int(net.width)),
            "face1kb.parameters": str(count_parameters(net)),
        }
    )
    tensors = {
        k: v.detach().to("cpu").contiguous().clone()
        for k, v in net.state_dict().items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=meta)
    return path
