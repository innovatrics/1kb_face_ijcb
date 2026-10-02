# SPDX-License-Identifier: MIT
"""The two codec variants: :class:`FaceCodecFast` and :class:`FaceCodecAccurate`.

Both subclass :class:`~face1kb.codec.backbone.MeanScaleHyperpriorVBR3` (gain-unit
variable rate, third hyper-downsample) and replace the stock synthesis ``g_s`` with
:class:`CondSynthesis`, which applies FiLM on ``(gain, resolution)`` after every
inverse-GDN block and, for ACCURATE, FiLM from the decoded identity side-stream.

* **FAST** (``variant_id = 0``): ``N=64, M=96, Nz=48``, no attention, no
  side-stream; 1,352,021 parameters.
* **ACCURATE** (``variant_id = 1``): ``N=192, M=320, Nz=64``, two attention blocks in
  the analysis and two in the synthesis transform, the identity side-stream (frozen
  EdgeFace-S anchor, 3,652,520 parameters) and a gated refinement head; 18,712,400
  parameters in total.

The constructor defaults are the shipped configurations; the released weights load
into ``FaceCodecFast()`` / ``FaceCodecAccurate()`` with ``strict=True``.

Conditioning is passed by stashing it on ``g_s`` just before delegating to the parent
``forward``/``compress``/``decompress`` (which call ``self.g_s(y_hat)`` internally),
so the parent's gain logic is used unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from compressai.layers import GDN, AttentionBlock
from compressai.models.utils import conv

from .backbone import MeanScaleHyperpriorVBR3
from .film import RateResFiLM, film
from .layers import ResizeConv


class CondSynthesis(nn.Module):
    """Synthesis transform with per-stage FiLM on ``(gain, resolution[, identity])``.

    Mirrors the stock mean-scale ``g_s`` (x16 upsample, 3 output channels), but every
    x2 upsample is a checkerboard-free :class:`~face1kb.codec.layers.ResizeConv`
    instead of a transposed convolution. An inverse GDN follows each of the first three
    upsamples and a :class:`~face1kb.codec.film.RateResFiLM` is applied after each
    inverse GDN. :meth:`set_cond` stashes the conditioning used by :meth:`forward`.

    Parameters
    ----------
    N, M
        Channel width and latent channel count.
    attention
        Insert an ``AttentionBlock`` after the first and second stages.
    """

    def __init__(self, N: int, M: int, attention: bool = False):
        super().__init__()
        self.N, self.M = N, M
        self.d0 = ResizeConv(M, N)
        self.i0 = GDN(N, inverse=True)
        self.att0 = AttentionBlock(N) if attention else None
        self.att1 = AttentionBlock(N) if attention else None
        self.d1 = ResizeConv(N, N)
        self.i1 = GDN(N, inverse=True)
        self.d2 = ResizeConv(N, N)
        self.i2 = GDN(N, inverse=True)
        self.head = ResizeConv(N, 3)
        self.f0 = RateResFiLM(N)
        self.f1 = RateResFiLM(N)
        self.f2 = RateResFiLM(N)
        self._gain: float = 1.0
        self._res: int = 128
        self._side = None  # optional list of 3 (gamma, beta) from the side-stream

    def set_cond(self, gain: float, res: int, side=None) -> None:
        """Stash the conditioning for the next forward call."""
        self._gain, self._res, self._side = float(gain), int(res), side

    def _stage(self, h, frlm, idx):
        gamma, beta = frlm(self._gain, self._res, device=h.device)
        h = film(h, gamma, beta)
        if self._side is not None:
            sg, sb = self._side[idx]
            h = film(h, sg, sb)
        return h

    def forward(self, y_hat):
        """Decode the latent to an image, applying FiLM after each inverse GDN."""
        h = self._stage(self.i0(self.d0(y_hat)), self.f0, 0)
        if self.att0 is not None:
            h = self.att0(h)
        h = self._stage(self.i1(self.d1(h)), self.f1, 1)
        if self.att1 is not None:
            h = self.att1(h)
        h = self._stage(self.i2(self.d2(h)), self.f2, 2)
        return self.head(h)


class _CondCodecMixin:
    """Passes FiLM conditioning to ``g_s`` around the parent entropy methods."""

    VARIANT_ID = 0

    def _cond_gain(self, inputscale, s: int) -> float:
        if inputscale:
            return float(inputscale)
        idx = max(0, min(int(s), len(self.Gain) - 1))
        return float(self.Gain[idx].item())

    def _prep(self, inputscale, s, res, padded_side, side=None):
        self.g_s.set_cond(self._cond_gain(inputscale, s), res or padded_side, side)

    def forward(self, x, stage: int = 2, s: int = 1, inputscale=0, res=None):
        """FiLM-conditioned forward (``res`` = true side; default: padded side)."""
        self._prep(inputscale, s, res, x.shape[-1], self._side_films(x))
        return super().forward(x, stage=stage, s=s, inputscale=inputscale)

    def compress(self, x, stage: int = 2, s: int = 1, inputscale=0, res=None):
        """Encode to rANS strings with FiLM conditioning."""
        self._prep(inputscale, s, res, x.shape[-1], self._side_films(x))
        return super().compress(x, stage=stage, s=s, inputscale=inputscale)

    def decompress(
        self, strings, shape, stage=2, s=1, inputscale=0, res=None, side=None
    ):
        """Decode strings; ``res`` should be the true side length (from the header)."""
        self._prep(inputscale, s, res, shape[-1] * self.downsampling_factor, side)
        return super().decompress(
            strings, shape, stage=stage, s=s, inputscale=inputscale
        )

    def _side_films(self, x):  # FAST has no side-stream
        return None


class FaceCodecFast(_CondCodecMixin, MeanScaleHyperpriorVBR3):
    """face1kb-FAST: slim channels, no attention, no identity side-stream.

    Parameters
    ----------
    N, M, Nz
        Channel width, latent channels and hyperprior channels.
    """

    VARIANT_ID = 0

    def __init__(self, N: int = 64, M: int = 96, Nz: int = 48):
        super().__init__(N=N, M=M, Nz=Nz, vr_entbttlnck=True)
        self.g_s = CondSynthesis(N, M, attention=False)


class FaceCodecAccurate(_CondCodecMixin, MeanScaleHyperpriorVBR3):
    """face1kb-ACCURATE: attention, identity side-stream and gated refinement head.

    Parameters
    ----------
    N, M, Nz
        Channel width, latent channels and hyperprior channels.
    side_dim
        Dimensionality of the entropy-coded identity code.
    anchor
        Name of the frozen FR anchor of the side-stream (EdgeFace-S).
    refine
        Build the gated refinement head.
    use_side
        Build the identity side-stream. ``False`` gives the side-stream ablation:
        same transforms, but no stored identity code and no refinement head, so the
        decoder is conditioned on ``(gain, resolution)`` only.
    pretrained_anchor
        Load the anchor's pretrained weights (needed only to train from scratch; the
        released codec weights contain the anchor).
    anchor_factory
        Optional ``(name, pretrained) -> (family, module)`` anchor builder.
    """

    VARIANT_ID = 1

    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        Nz: int = 64,
        side_dim: int = 128,
        anchor: str = "edgeface_s",
        refine: bool = True,
        use_side: bool = True,
        pretrained_anchor: bool = False,
        anchor_factory=None,
    ):
        super().__init__(N=N, M=M, Nz=Nz, vr_entbttlnck=True)
        from .refine_head import RefinementHead  # noqa: PLC0415
        from .side_stream import IdentitySideStream  # noqa: PLC0415

        # Wider analysis with two attention blocks.
        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            AttentionBlock(N),
            conv(N, N),
            GDN(N),
            AttentionBlock(N),
            conv(N, M),
        )
        self.g_s = CondSynthesis(N, M, attention=True)
        self.anchor_name = anchor if use_side else None
        if use_side:
            self.side = IdentitySideStream(
                dim=side_dim,
                n_stages=3,
                stage_ch=N,
                anchor=anchor,
                pretrained_anchor=pretrained_anchor,
                anchor_factory=anchor_factory,
            )
            self.refine = RefinementHead(p_dim=side_dim) if refine else None
        else:
            self.side = None
            self.refine = None

        # Widen the variable-rate range downward (min lmbda 0.0018 -> 0.0002): at
        # 224/256 px (both padded to 256) the 16x16 latent plus the side-channel give
        # a high rate floor; the low-lmbda levels train the network at the low gains
        # to reach tight budgets.
        # The maximum (0.18) is kept so 1024 B can still be filled. Gain is
        # re-initialised log-spaced so the gain table spans the lower range; it is not
        # trained (CompressAI detaches Gain in the training forward).
        lo, hi, levels = 0.0002, 0.18, len(self.lmbda)
        self.lmbda = [lo * (hi / lo) ** (i / (levels - 1)) for i in range(levels)]
        glo, ghi = 0.04, 1.0
        gain = [glo * (ghi / glo) ** (i / (levels - 1)) for i in range(levels)]
        self.Gain = nn.Parameter(torch.tensor(gain, dtype=torch.float32))

    def _side_films(self, x):
        """Encode the identity side-stream and return the per-stage FiLM parameters."""
        return self.side.film_params(x) if self.side is not None else None

    def forward(self, x, stage: int = 2, s: int = 1, inputscale=0, res=None):
        """Forward, adding the side-stream rate to the likelihoods, then refine."""
        out = super().forward(x, stage=stage, s=s, inputscale=inputscale, res=res)
        if self.side is None:  # side-stream ablation: no side rate, no refine
            return out
        if self.side._last_likelihoods is not None:
            out["likelihoods"]["side"] = self.side._last_likelihoods
        if self.refine is not None and self.side._last_p_hat is not None:
            out["x_hat"] = self.refine(out["x_hat"], self.side._last_p_hat)
        return out

    def decompress(self, *args, **kwargs):
        """Decode strings, then apply the gated refinement head as ``forward`` does.

        :func:`face1kb.codec.budget.decode_from_container` runs ``self.side.decode``
        first, which stashes the decoded identity code on ``self.side._last_p_hat``;
        the synthesis then runs with the side FiLM and the refinement head is applied
        to its output, reproducing the trained forward pass.
        """
        out = super().decompress(*args, **kwargs)
        if (
            self.side is not None
            and self.refine is not None
            and self.side._last_p_hat is not None
        ):
            out["x_hat"] = self.refine(out["x_hat"], self.side._last_p_hat)
        return out


VARIANTS: dict[str, type] = {"fast": FaceCodecFast, "accurate": FaceCodecAccurate}


def build_model(
    variant: str,
    no_side: bool = False,
    anchor: str = "edgeface_s",
    pretrained_anchor: bool = False,
):
    """Instantiate a codec variant with the shipped constructor defaults.

    Parameters
    ----------
    variant
        ``"fast"`` or ``"accurate"``.
    no_side
        ACCURATE only: build the side-stream ablation (no side-stream, no refine).
    anchor
        ACCURATE only: name of the side-stream anchor.
    pretrained_anchor
        ACCURATE only: load the anchor's pretrained weights (for training).
    """
    if variant == "fast":
        return FaceCodecFast()
    if variant == "accurate":
        return FaceCodecAccurate(
            anchor=anchor,
            use_side=not no_side,
            refine=not no_side,
            pretrained_anchor=pretrained_anchor,
        )
    raise ValueError(f"unknown variant {variant!r}; choose 'fast' or 'accurate'")
