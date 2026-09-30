# SPDX-License-Identifier: MIT
"""Identity side-stream of the ACCURATE variant.

A frozen FR **anchor** (EdgeFace-S) embeds the input crop (512-D, L2-normalised); a
learned projection maps the embedding to a ``dim``-D code (128), a factorized
``EntropyBottleneck`` entropy-codes it into the container's side-channel, and the
decoded code drives per-stage FiLM ``(gamma, beta)`` in the decoder synthesis and the
gated refinement head (:mod:`face1kb.codec.refine_head`).

The anchor runs at encode time: an ACCURATE encoder performs one EdgeFace-S forward
pass per image. The anchor embedding is detached, so only the projection, the
entropy bottleneck and the FiLM heads are trained.

Notes
-----
With the released ACCURATE weights the stored identity code is **image-independent**.
The gain of the projection's ``LayerNorm`` collapsed during training (mean
``|weight|`` about 4e-4, maximum 1.5e-3), so the projected code equals the
``LayerNorm`` bias up to about 1e-3 for every input; entropy coding rounds it to
integers, which removes the remaining image dependence. Every stored ACCURATE
container of the paper carries the same 8-byte side-channel (``f82a5c8500000000`` on
CUDA), and fresh encodes of different subjects give the same bytes. The side entropy
model assigns this code a probability of practically 1 (about 1e-7 bits), and 8 bytes
is the minimum output of CompressAI's 64-bit rANS coder. (The quantiles of ``eb``
also never left their initial values, because the auxiliary loss covers only the
hyperprior bottleneck; this does not cause the constant code.)

Consequently, with the released weights the decoded code acts as a constant, learned
conditioning of the FiLM stages and of the refinement head (the refinement head still
depends on the image through the decoded ``x_hat``), and the EdgeFace-S forward pass
of the encoder does not change the output for any tested image. The stored bytes are
not a linkable biometric descriptor; the unquantised projection, which is never
stored, still carries a small identity-dependent residual.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck

from .identity_loss import _TORCH_FAMILIES, _fr_embed, load_torch_fr
from .resize import resize112

#: ``anchor_factory(name, pretrained) -> (family, module)``
AnchorFactory = Callable[[str, bool], tuple[str, nn.Module]]


def default_anchor_factory(name: str, pretrained: bool) -> tuple[str, nn.Module]:
    """Build a registered torch FR model as the side-stream anchor."""
    return load_torch_fr(name, pretrained=pretrained)


class IdentitySideStream(nn.Module):
    """Frozen anchor -> learned projection -> entropy-coded code -> per-stage FiLM.

    Parameters
    ----------
    dim
        Dimensionality of the projected (entropy-coded) identity code.
    n_stages
        Number of synthesis stages to modulate (matches
        :class:`~face1kb.codec.variants.CondSynthesis`).
    stage_ch
        Channel count of each modulated stage.
    anchor
        Name of the frozen torch FR anchor (see
        :func:`face1kb.codec.identity_loss.available_torch_fr`).
    pretrained_anchor
        Load the anchor's pretrained weights. Inference does not need them because
        the ACCURATE weight file contains the anchor; training from scratch does.
    anchor_factory
        Optional callable ``(name, pretrained) -> (family, module)`` replacing the
        default registry lookup.
    """

    def __init__(
        self,
        dim: int = 128,
        n_stages: int = 3,
        stage_ch: int = 128,
        anchor: str = "edgeface_s",
        pretrained_anchor: bool = False,
        anchor_factory: AnchorFactory | None = None,
    ):
        super().__init__()
        factory = anchor_factory or default_anchor_factory
        family, anchor_net = factory(anchor, pretrained_anchor)
        if family not in _TORCH_FAMILIES:
            raise ValueError(f"anchor {anchor!r} must be a torch family")
        self.anchor_family = family
        self.anchor = anchor_net.eval()
        self.anchor.requires_grad_(False)
        self.proj = nn.Sequential(nn.Linear(512, dim, bias=False), nn.LayerNorm(dim))
        self.eb = EntropyBottleneck(dim)
        self.film_heads = nn.ModuleList(
            [nn.Linear(dim, 2 * stage_ch) for _ in range(n_stages)]
        )
        self.dim = dim
        self._last_likelihoods = None
        self._last_p_hat = None

    #: Keep the frozen anchor in eval mode when the module is put into training
    #: mode. ``False`` lets ``train()`` switch the anchor to training mode as well,
    #: which is what the trainer of the paper runs did; the two settings differ only
    #: for anchors with batch-norm or dropout layers (EdgeFace has neither).
    freeze_anchor_mode: bool = True

    def train(self, mode: bool = True):
        """Set training mode on the trainable parts.

        With :attr:`freeze_anchor_mode` (the default) the anchor stays in eval mode,
        which keeps the batch-norm statistics of such anchors frozen. EdgeFace has no
        batch-norm or active dropout, so for the released configuration both
        settings give the same outputs.
        """
        super().train(mode)
        if self.freeze_anchor_mode:
            self.anchor.eval()
        return self

    def _code(self, x01: torch.Tensor) -> torch.Tensor:
        """Project the (detached) anchor embedding to the ``dim``-D identity code."""
        with torch.no_grad():
            e = _fr_embed(self.anchor_family, self.anchor, resize112(x01))
        return self.proj(e)

    def _films_from_code(self, p_hat: torch.Tensor) -> list:
        films = []
        for head in self.film_heads:
            gamma, beta = head(p_hat).chunk(2, dim=-1)
            # The FiLM scalars broadcast over H x W, so an unbounded affine could
            # impose a global colour/contrast cast. Bound the scale to +-20 % and the
            # shift to +-0.2 so the code modulates rather than recolours.
            films.append((1.0 + 0.2 * torch.tanh(gamma), 0.2 * torch.tanh(beta)))
        return films

    def film_params(self, x01: torch.Tensor) -> list:
        """Training path: code -> EntropyBottleneck (noise) -> per-stage FiLM.

        The code likelihoods are stashed on ``self._last_likelihoods`` so the variant
        can add the side-stream rate to the loss, and the noisy code on
        ``self._last_p_hat`` for the refinement head.
        """
        p = self._code(x01)
        p_in = p.unsqueeze(-1).unsqueeze(-1)  # (N, dim, 1, 1)
        p_hat, lik = self.eb(p_in)
        p_hat = p_hat.squeeze(-1).squeeze(-1)
        self._last_likelihoods = lik
        self._last_p_hat = p_hat
        return self._films_from_code(p_hat)

    def _ensure_eb_updated(self) -> None:
        """Build the entropy-bottleneck CDF tables once per instance.

        ``update(force=True)`` is a comparatively slow ``_pmf_to_cdf`` rebuild and the
        quantiles are frozen at inference, so it runs only on first use.
        """
        if not getattr(self, "_eb_updated", False):
            self.eb.update(force=True)
            self._eb_updated = True

    @torch.no_grad()
    def encode(self, x01: torch.Tensor) -> bytes:
        """Encode the identity code of a single image to side-channel bytes."""
        self._ensure_eb_updated()
        p = self._code(x01)
        strings = self.eb.compress(p.unsqueeze(-1).unsqueeze(-1))
        return strings[0]

    @torch.no_grad()
    def decode(self, sc: bytes) -> list:
        """Decode side-channel bytes to per-stage FiLM ``(gamma, beta)``.

        The decoded code is also stashed on ``self._last_p_hat``, which the ACCURATE
        ``decompress`` feeds to the refinement head, as the training forward does.
        """
        self._ensure_eb_updated()
        p_hat = self.eb.decompress([sc], (1, 1))  # (1, dim, 1, 1)
        p_hat = p_hat.squeeze(-1).squeeze(-1)
        self._last_p_hat = p_hat
        return self._films_from_code(p_hat)
