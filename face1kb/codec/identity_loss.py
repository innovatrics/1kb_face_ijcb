# SPDX-License-Identifier: MIT
"""Frozen torch face-recognition models and the in-loop identity loss.

The codecs use EdgeFace (George et al., IEEE T-BIOM 2024) in two places:

* **training** -- :class:`InLoopIdentityLoss` computes ``1 - cos(FR(x_hat), FR(x))``
  with a frozen EdgeFace model, with gradients flowing to the decoded pixels through
  the differentiable resize to 112 px (EdgeFace-XS for FAST, EdgeFace-S for
  ACCURATE);
* **inference (ACCURATE only)** -- the identity side-stream embeds the input with a
  frozen EdgeFace-S anchor (:mod:`face1kb.codec.side_stream`).

The network architectures come from the vendored EdgeFace code
(:mod:`face1kb.third_party.edgeface`, BSD-3-Clause). Pretrained weights (CC BY-NC-SA
4.0, Idiap Research Institute) are downloaded on first use from the official Idiap
repositories into ``FACE1KB_MODELS_ROOT/edgeface/`` and verified by SHA-256. The
ACCURATE codec weights already contain the EdgeFace-S anchor, so inference never
downloads anything.

Other torch FR models can be made available to the identity loss and to the
side-stream with :func:`register_torch_fr`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resize import resize112

logger = logging.getLogger(__name__)

#: Model families whose embedding call :func:`_fr_embed` knows.
_TORCH_FAMILIES = {"edgeface", "topofr", "cvlface"}


@dataclass(frozen=True)
class PretrainedWeights:
    """Location and checksum of a pretrained FR checkpoint.

    Attributes
    ----------
    filename
        File name under ``FACE1KB_MODELS_ROOT/<subdir>/``.
    sha256
        Expected SHA-256 of the file.
    urls
        Download URLs, tried in order.
    subdir
        Sub-folder of ``FACE1KB_MODELS_ROOT``.
    """

    filename: str
    sha256: str
    urls: tuple[str, ...]
    subdir: str = "edgeface"


_HF = "https://huggingface.co/Idiap"
_IDIAP_GITLAB = (
    "https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface/-/raw/master/checkpoints"
)

#: Official EdgeFace checkpoints (Idiap, CC BY-NC-SA 4.0), pinned by revision + hash.
EDGEFACE_WEIGHTS: dict[str, PretrainedWeights] = {
    "edgeface_xs_gamma_06": PretrainedWeights(
        filename="edgeface_xs_gamma_06.pt",
        sha256="5ae7504cd9aee0a5d52c2115fd2eb66b0985dd1730f40134b5854e0cb658ce16",
        urls=(
            f"{_HF}/EdgeFace-XS-GAMMA/resolve/"
            "735c1b59bdc798260e56f12533fddfe8a8c4c568/edgeface_xs_gamma_06.pt",
            f"{_IDIAP_GITLAB}/edgeface_xs_gamma_06.pt",
        ),
    ),
    "edgeface_s_gamma_05": PretrainedWeights(
        filename="edgeface_s_gamma_05.pt",
        sha256="dc59abda2e8580399fd115a1eeb07e1f21156196db604b884407bcf0f17efb07",
        urls=(
            f"{_HF}/EdgeFace-S-GAMMA/resolve/"
            "a1c171036159e94bd4c766088370a37d0899d31f/edgeface_s_gamma_05.pt",
            f"{_IDIAP_GITLAB}/edgeface_s_gamma_05.pt",
        ),
    ),
    "edgeface_xxs": PretrainedWeights(
        filename="edgeface_xxs.pt",
        sha256="5b6bac48ea660aca185be8a8ac934db2093b51259461721da53c468230bc24c9",
        urls=(
            f"{_HF}/EdgeFace-XXS/resolve/"
            "e710f4647ac4dde011c13f82224539d24800ab92/edgeface_xxs.pt",
            f"{_IDIAP_GITLAB}/edgeface_xxs.pt",
        ),
    ),
    "edgeface_base": PretrainedWeights(
        filename="edgeface_base.pt",
        sha256="95861c09b22810136f43ec98845e7f09bfc3c43f5a804984a7bd2eac20abc30c",
        urls=(
            f"{_HF}/EdgeFace-Base/resolve/"
            "2944a63d6a73efb3754c553e119b57fe7e54a179/edgeface_base.pt",
            f"{_IDIAP_GITLAB}/edgeface_base.pt",
        ),
    ),
}

#: Short names -> EdgeFace architecture names. The codecs use ``edgeface_xs`` (FAST
#: identity loss) and ``edgeface_s`` (ACCURATE anchor and identity loss).
EDGEFACE_ARCHS: dict[str, str] = {
    "edgeface_xxs": "edgeface_xxs",
    "edgeface_xs": "edgeface_xs_gamma_06",
    "edgeface_s": "edgeface_s_gamma_05",
    "edgeface_base": "edgeface_base",
}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch_pretrained(spec: PretrainedWeights, models_root: Path | None = None) -> Path:
    """Return the local path of a pretrained checkpoint, downloading it if needed.

    The file is stored as ``<models_root>/<spec.subdir>/<spec.filename>`` (default
    root ``FACE1KB_MODELS_ROOT``). An existing file is re-hashed; a download is
    written to a temporary file and moved into place only after its SHA-256 matched.

    Raises
    ------
    RuntimeError
        If an existing file has the wrong hash or no URL yields the expected file.
    """
    from face1kb import config  # noqa: PLC0415

    root = Path(models_root) if models_root is not None else config.MODELS_ROOT
    dst = root / spec.subdir / spec.filename
    if dst.exists():
        got = sha256_file(dst)
        if got != spec.sha256:
            raise RuntimeError(
                f"{dst} has sha256 {got}, expected {spec.sha256}; delete it to "
                "download it again"
            )
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in spec.urls:
        logger.info("downloading %s", url)
        fd, tmp_name = tempfile.mkstemp(dir=dst.parent, suffix=".part")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            got = sha256_file(tmp)
            if got != spec.sha256:
                errors.append(f"{url}: sha256 {got} != {spec.sha256}")
                continue
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)  # mkstemp creates the file with 0600
            tmp.replace(dst)
            return dst
        except OSError as e:  # URLError is an OSError
            errors.append(f"{url}: {e}")
        finally:
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"could not fetch {spec.filename}: " + "; ".join(errors))


def build_edgeface(arch: str, pretrained: bool = True) -> nn.Module:
    """Build an EdgeFace network, optionally with the official pretrained weights.

    Parameters
    ----------
    arch
        EdgeFace architecture name (a key of :data:`EDGEFACE_WEIGHTS`, e.g.
        ``edgeface_xs_gamma_06`` or ``edgeface_s_gamma_05``).
    pretrained
        Load the official checkpoint (downloaded and hash-verified on first use).
        With ``False`` the network is randomly initialised; this is how the
        ACCURATE codec builds its anchor before loading the codec weights.
    """
    from face1kb.third_party.edgeface import get_model  # noqa: PLC0415

    model = get_model(arch)
    if pretrained:
        path = fetch_pretrained(EDGEFACE_WEIGHTS[arch])
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model


#: name -> (family, builder(pretrained) -> nn.Module)
_TORCH_FR: dict[str, tuple[str, Callable[[bool], nn.Module]]] = {
    name: ("edgeface", lambda pretrained, _a=arch: build_edgeface(_a, pretrained))
    for name, arch in EDGEFACE_ARCHS.items()
}


def register_torch_fr(
    name: str, family: str, builder: Callable[[bool], nn.Module]
) -> None:
    """Make a torch FR model available to the identity loss and the side-stream.

    Parameters
    ----------
    name
        Name used in ``InLoopIdentityLoss(model_names=...)`` or as the ACCURATE
        ``anchor``.
    family
        One of ``edgeface``, ``topofr``, ``cvlface``; selects the call signature in
        :func:`_fr_embed` (inputs are 112 x 112 RGB in ``[-1, 1]``).
    builder
        ``builder(pretrained: bool) -> nn.Module`` returning the network.
    """
    if family not in _TORCH_FAMILIES:
        raise ValueError(f"family {family!r} must be one of {sorted(_TORCH_FAMILIES)}")
    _TORCH_FR[name] = (family, builder)


def available_torch_fr() -> list[str]:
    """Names accepted by :func:`load_torch_fr`."""
    return sorted(_TORCH_FR)


def load_torch_fr(name: str, pretrained: bool = True) -> tuple[str, nn.Module]:
    """Build a registered torch FR model; return ``(family, module)`` in eval mode."""
    if name not in _TORCH_FR:
        raise ValueError(
            f"unknown torch FR model {name!r}; available: {available_torch_fr()} "
            "(see register_torch_fr)"
        )
    family, builder = _TORCH_FR[name]
    return family, builder(pretrained).eval()


def _fr_embed(family: str, net: nn.Module, x112_01: torch.Tensor) -> torch.Tensor:
    """Embed a 112 x 112 RGB ``[0, 1]`` batch; return L2-normalised embeddings."""
    xin = (x112_01 - 0.5) / 0.5
    out = net(xin, phase="infer") if family == "topofr" else net(xin)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return F.normalize(out, dim=1)


class InLoopIdentityLoss(nn.Module):
    """``1 - cos`` identity loss over one or more frozen torch FR models.

    Parameters
    ----------
    model_names
        Registered names (see :func:`available_torch_fr`); ``("edgeface_xs",)`` for
        FAST and ``("edgeface_s",)`` for ACCURATE.
    device
        Device to host the frozen FR models on.
    reduction
        ``"mean"`` for a scalar, ``"none"`` for per-sample values ``(N,)``.
    """

    def __init__(
        self,
        model_names: tuple[str, ...] = ("edgeface_xs",),
        device: str | torch.device = "cuda",
        reduction: str = "mean",
    ):
        super().__init__()
        self.reduction = reduction
        self.families: list[str] = []
        nets: list[nn.Module] = []
        for name in model_names:
            family, net = load_torch_fr(name, pretrained=True)
            net = net.to(device).eval()
            net.requires_grad_(False)
            self.families.append(family)
            nets.append(net)
        self.nets = nn.ModuleList(nets)

    def train(self, mode: bool = True):
        """Keep the frozen FR models in eval mode whatever ``mode`` is."""
        super().train(mode)
        self.nets.eval()
        return self

    def forward(self, x_hat: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
        """Identity loss between decoded ``x_hat`` and clean ``x_clean`` (RGB [0,1])."""
        xh = resize112(x_hat)
        xc = resize112(x_clean)
        total = x_hat.new_zeros(x_hat.shape[0])
        for family, net in zip(self.families, self.nets):
            eh = _fr_embed(family, net, xh)
            with torch.no_grad():
                ec = _fr_embed(family, net, xc)
            total = total + (1.0 - (eh * ec).sum(dim=1))
        total = total / len(self.nets)
        return total.mean() if self.reduction == "mean" else total
