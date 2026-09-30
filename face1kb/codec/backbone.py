# SPDX-License-Identifier: MIT
"""``MeanScaleHyperpriorVBR3``: the shared backbone of both face1kb codecs.

It subclasses CompressAI's variable-rate ``MeanScaleHyperpriorVbr`` (Kamisli, Racape
and Choi, "Variable-Rate Learned Image Compression with Multi-Objective Optimization
and Quantization-Reconstruction Offsets", DCC 2024: gain units ``self.Gain``, the
``QuantABCD`` reconstruction-offset MLP and the ``gayn2zqstep`` variable-rate
hyperprior step) and changes the hyperprior so that its byte floor is small enough
for sub-1 kB budgets:

* a third stride-2 stage in ``h_a`` (mirrored by a third x2 ``ResizeConv`` in
  ``h_s``), so the hyperprior latent ``z`` is ``x / 128`` instead of the stock
  ``x / 64``, which quarters the z-grid area, and
* a small hyperprior channel count ``Nz``.

Entropy tables are built with ``update(force=True, scale=g)``, the variable-rate path
in which the z quantisation step is derived from the gain. For both released models
that step sits at its lower bound of 0.5 for all 64 gains of the rate table, so the
z tables are effectively gain-independent; the gain controls the rate through the
quantisation of ``y`` (and the ``QuantABCD`` offsets). The total downsampling factor
is ``2 ** (4 + 3) = 128``, so inputs must be edge-padded to a multiple of 128 (see
:mod:`face1kb.codec.resize`).

Notes
-----
The rANS bitstreams depend on CompressAI's ``update``/``_pmf_to_cdf``/rANS code, so
``compressai`` is pinned to 1.2.8. With that version, ``EntropyBottleneckVbr.forward``
passes the quantisation step positionally to ``_likelihood_variable``, where it lands
in the ``stop_gradient`` argument; the z density model of the shipped weights is
therefore still at its initialisation. This affects training only; it is kept as is
so that training reproduces the shipped recipe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, EntropyBottleneckVbr
from compressai.models.utils import conv
from compressai.models.vbr import MeanScaleHyperpriorVbr

from .layers import ResizeConv


class MeanScaleHyperpriorVBR3(MeanScaleHyperpriorVbr):
    """Mean-scale hyperprior with gain-unit variable rate and a 3rd hyper-downsample.

    Parameters
    ----------
    N
        Main transform (``g_a``/``g_s``) channel width.
    M
        Latent ``y`` channel count.
    Nz
        Hyperprior (``z``) channel count, kept small so the z floor is small.
    vr_entbttlnck
        Use the variable-rate ``EntropyBottleneckVbr`` for ``z`` so the gain also
        coarsens the hyperprior (otherwise ``z`` is a fixed floor).
    **kwargs
        Forwarded to ``MeanScaleHyperpriorVbr``.
    """

    def __init__(
        self,
        N: int = 96,
        M: int = 128,
        Nz: int = 64,
        vr_entbttlnck: bool = True,
        **kwargs,
    ):
        super().__init__(N=N, M=M, vr_entbttlnck=vr_entbttlnck, **kwargs)
        self.Nz = int(Nz)
        # h_a: stride-1 head + three stride-2 convs -> z = y / 8 = x / 128.
        self.h_a = nn.Sequential(
            conv(M, self.Nz, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(self.Nz, self.Nz),
            nn.LeakyReLU(inplace=True),
            conv(self.Nz, self.Nz),
            nn.LeakyReLU(inplace=True),
            conv(self.Nz, self.Nz),  # the extra stride-2 stage
        )
        # h_s: three x2 ResizeConv upsamples (no transposed conv) + a stride-1 head
        # producing M * 2 channels (mean, scale). Mirrors the three stride-2 of h_a.
        self.h_s = nn.Sequential(
            ResizeConv(self.Nz, M),
            nn.LeakyReLU(inplace=True),
            ResizeConv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            ResizeConv(M * 3 // 2, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        # z has Nz channels, so the entropy bottleneck must match.
        if vr_entbttlnck:
            self.entropy_bottleneck = EntropyBottleneckVbr(self.Nz)
        else:
            self.entropy_bottleneck = EntropyBottleneck(self.Nz)

    @property
    def downsampling_factor(self) -> int:
        """Total geometric downsampling: ``g_a`` 16 x ``h_a`` 8 = 128."""
        return 2 ** (4 + 3)

    def _coerce_scale(self, inputscale):
        """Coerce a Python-float gain to a device tensor (0 means use ``Gain[s]``)."""
        if isinstance(inputscale, (int, float)):
            if inputscale == 0:
                return 0  # sentinel handled by the parent's _get_scale
            return torch.tensor(float(inputscale), device=self.Gain.device)
        return inputscale

    def forward(self, x, stage: int = 2, s: int = 1, inputscale=0):
        """Gain-modulated forward; ``inputscale`` may be a float or a tensor."""
        return super().forward(
            x, stage=stage, s=s, inputscale=self._coerce_scale(inputscale)
        )

    def compress(self, x, stage: int = 2, s: int = 1, inputscale=0):
        """Encode to rANS strings at the continuous gain ``inputscale``."""
        return super().compress(
            x, stage=stage, s=s, inputscale=self._coerce_scale(inputscale)
        )

    def decompress(self, strings, shape, stage: int = 2, s: int = 1, inputscale=0):
        """Decode strings produced at the same gain ``inputscale``."""
        return super().decompress(
            strings, shape, stage=stage, s=s, inputscale=self._coerce_scale(inputscale)
        )

    def aux_loss(self):
        """Entropy-bottleneck quantile loss for the auxiliary optimizer.

        ``EntropyBottleneckVbr`` is not an ``EntropyBottleneck`` subclass, so the base
        ``CompressionModel.aux_loss`` (which filters by ``isinstance``) would return 0;
        the bottleneck loss is returned directly (its parameter is
        ``entropy_bottleneck.quantiles``).
        """
        return self.entropy_bottleneck.loss()

    def update_for_gain(self, gain: float, force: bool = True) -> bool:
        """Build the entropy-coder CDF tables for a continuous gain ``gain``.

        Uses the variable-rate update path (``scale=gain``), which derives the z
        quantisation step from the gain (clamped at its lower bound for the released
        weights). Returns whether any buffer changed.
        """
        sc = torch.as_tensor(float(gain), device=self.Gain.device).view(1)
        return self.update(force=force, scale=sc)
