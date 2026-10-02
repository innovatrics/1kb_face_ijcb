# SPDX-License-Identifier: MIT
"""Rate-distortion-identity training objective.

``L = w_R * R + w_D * MSE + w_p * LPIPS + w_id * (1 - cos) + w_ms * (1 - MS-SSIM)
+ w_floor * relu(floor - MS-SSIM)``

``R`` is the CompressAI training rate (``-log2`` likelihood per pixel, noise path).
The MS-SSIM floor hinge penalises blurred reconstructions that could still fool a
face matcher. LPIPS (AlexNet) is optional; it disables itself if its backbone weights
cannot be loaded (e.g. offline).

The weights are passed per step, so the trainer can schedule them
(:func:`phase_weights`).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

logger = logging.getLogger(__name__)


def rate_bpp(likelihoods: dict, num_pixels: int) -> torch.Tensor:
    """Bits per pixel from CompressAI likelihoods (summed over all entries).

    Likelihoods are cast to fp32 and floored before ``log2`` so the rate is stable
    under fp16 autocast.
    """
    bits = sum(
        (-torch.log2(lk.float().clamp_min(1e-12))).sum() for lk in likelihoods.values()
    )
    return bits / num_pixels


class CodecLoss(nn.Module):
    """Composite codec objective with optional LPIPS and an MS-SSIM floor hinge.

    Parameters
    ----------
    identity_loss
        An :class:`~face1kb.codec.identity_loss.InLoopIdentityLoss` (or ``None``).
    use_lpips
        Try to construct an LPIPS (AlexNet) metric; disabled with a warning if its
        weights cannot be loaded.
    msssim_floor
        MS-SSIM below which the hinge penalises the reconstruction.
    """

    def __init__(
        self,
        identity_loss: nn.Module | None = None,
        use_lpips: bool = True,
        msssim_floor: float = 0.6,
    ):
        super().__init__()
        self.identity = identity_loss
        self.msssim_floor = msssim_floor
        self.lpips = None
        if use_lpips:
            try:
                import lpips  # noqa: PLC0415

                self.lpips = lpips.LPIPS(net="alex", verbose=False)
                self.lpips.requires_grad_(False)
                self.lpips.eval()
            except Exception as e:  # noqa: BLE001 - offline weights etc.
                logger.warning("LPIPS disabled (%s: %s)", type(e).__name__, e)
                self.lpips = None

    def forward(
        self,
        x: torch.Tensor,
        out: dict,
        weights: dict,
    ) -> tuple[torch.Tensor, dict]:
        """Return ``(total_loss, terms)`` for one batch.

        ``out`` is the codec forward dict (``x_hat`` and ``likelihoods``); ``weights``
        holds ``R, D, p, id, ms, floor`` (missing or zero terms are skipped; ``D``
        defaults to 1).
        """
        x_hat = out["x_hat"]
        n, _, h, w = x.shape
        terms: dict[str, float] = {}

        rate = rate_bpp(out["likelihoods"], n * h * w)
        mse = F.mse_loss(x_hat, x)
        total = weights.get("R", 0.0) * rate + weights.get("D", 1.0) * mse
        terms["rate_bpp"] = float(rate.detach())
        terms["mse"] = float(mse.detach())

        if self.lpips is not None and weights.get("p", 0.0) > 0:
            lp = self.lpips(x_hat * 2 - 1, x * 2 - 1).mean()
            total = total + weights["p"] * lp
            terms["lpips"] = float(lp.detach())

        need_ms = weights.get("ms", 0.0) > 0 or weights.get("floor", 0.0) > 0
        if need_ms:
            msssim_val = ms_ssim(x_hat, x, data_range=1.0, win_size=7)
            if weights.get("ms", 0.0) > 0:
                total = total + weights["ms"] * (1.0 - msssim_val)
            if weights.get("floor", 0.0) > 0:
                floor = torch.as_tensor(self.msssim_floor, device=x.device)
                total = total + weights["floor"] * F.relu(floor - msssim_val)
            terms["ms_ssim"] = float(msssim_val.detach())

        if self.identity is not None and weights.get("id", 0.0) > 0:
            idl = self.identity(x_hat, x)
            total = total + weights["id"] * idl
            terms["id_cos_dist"] = float(idl.detach())

        terms["total"] = float(total.detach())
        return total, terms


def phase_weights(step: int, total_steps: int, base: dict, fast: bool = False) -> dict:
    """Loss-weight schedule: identity term off, then ramped, then full.

    Rate, MSE, LPIPS and MS-SSIM are active from step 0 (they penalise blur, which
    prevents the degenerate "blur that fools the matcher" solution). The identity
    weight is warmed up in three phases over the fraction ``step / total_steps`` of
    training: 0 below 2 %, a linear ramp from 2 % to 20 %, and the full weight after
    20 %. ``fast`` is accepted for API compatibility and has no effect.
    """
    frac = step / max(total_steps, 1)
    w = dict(base)
    if frac < 0.02:
        id_ramp = 0.0
    elif frac < 0.20:
        id_ramp = (frac - 0.02) / 0.18
    else:
        id_ramp = 1.0
    if "id" in w:
        w["id"] = w["id"] * id_ramp
    return w
