# SPDX-License-Identifier: MIT
"""Hard byte-budget encoding and container decoding.

A learned codec controls its rate through a continuous gain, not a byte target, so
the budget is enforced at encode time: a **binary search** over a frozen 64-entry,
log-spaced gain table finds the largest gain whose *real* rANS output fits, in about
``log2(64) = 6`` encodes. Size grows monotonically with the gain (gain = inverse
quantisation step: higher gain, finer quantisation, more bytes).

The gain is snapped to the table so that the 6-bit ``rate_index`` in the container
header reconstructs it exactly at decode time; a float mismatch would silently corrupt
the decode, because the scales and the ``QuantABCD`` offsets depend on the gain.

Budget accounting
-----------------
The search target is ``budget - 7 - len(sidechannel)`` (the fixed container overhead
plus the identity side-channel). Resolutions without a bucket code (96 and 168 px of
the benchmark grid) also carry a 4-byte raw-geometry trailer:

* ``paper_compat=True`` ignores the trailer. This is the accounting used for every
  bitstream of the paper, and it reproduces them byte for byte. Such containers can
  exceed the budget at 96/168 px by at most 4 bytes; because the rANS strings are
  whole 32-bit words, the overshoot is exactly 3 bytes for budgets that are
  multiples of 4, as in every over-budget paper container of this kind.
* ``paper_compat=False`` (default) reserves the trailer as well. At bucketed
  resolutions (64/112/128/192/224/256 px) the search is unchanged, so every stream
  that fits is byte-identical to the ``paper_compat=True`` one; the outputs differ
  only where even the lowest gain overflows, because the default ``overflow`` policy
  differs (below).

Overflow
--------
When even the lowest gain does not fit, the encoder either emits that smallest
spatial stream flagged ``over_budget`` (``overflow="floor"``, the paper behaviour,
which exceeds the budget), emits an identity-only container that decodes to a black
frame (``overflow="identity"``, the default with ``paper_compat=False``) or raises
:class:`BudgetError` (``overflow="error"``).

Large budgets
-------------
Each rANS string has a 2-byte length prefix, so a stream fits only when both of its
strings are at most 65535 bytes. For budgets above about 64 kB (large crops only) the
search therefore picks the largest gain whose strings the container can hold, which
may leave part of the budget unused.

Minimum budget
--------------
The smallest container is the identity-only one: ``3 + len(sidechannel)`` bytes,
plus the 4-byte trailer at raw-geometry resolutions (3 or 7 B for FAST, 11 or 15 B
for ACCURATE with its 8-byte side-channel). With ``paper_compat=False`` a smaller
budget raises ``ValueError``, so the returned container never exceeds ``budget``
unless ``overflow="floor"`` is requested. With ``paper_compat=True`` a budget too
small for the container header and side-channel yields the identity-only container
(paper behaviour) unless ``overflow="error"``. Budgets below 1 byte are rejected.
"""

from __future__ import annotations

import math

import torch

from . import container
from .resize import pad_to_multiple

#: Number of entries of the gain table (the 6-bit ``rate_index``).
N_RATE_LEVELS = 64
#: Overflow policies accepted by :func:`encode_to_budget`.
OVERFLOW_POLICIES = ("floor", "identity", "error")


class BudgetError(RuntimeError):
    """Raised by ``overflow="error"`` when no gain of the table fits the budget."""


class IdentityOnlyWarning(UserWarning):
    """Issued when an encode falls back to the identity-only container.

    The identity-only container has no spatial latent and decodes to a black frame.
    Filter it with ``warnings.simplefilter("ignore", IdentityOnlyWarning)``.
    """


def gain_table(net) -> list:
    """Per-model 64-entry log-spaced gain table spanning the ``Gain`` training range.

    ``net.Gain`` holds the 8 fixed gains the network was trained at (CompressAI
    detaches it, so it is never updated). Gains outside that range are an untrained
    regime, so the table is derived from the loaded ``net.Gain`` (float32 min/max,
    interpolated in float64).
    """
    g = net.Gain.detach().abs().float()
    lo, hi = float(g.min()), float(g.max())
    return [lo * (hi / lo) ** (i / (N_RATE_LEVELS - 1)) for i in range(N_RATE_LEVELS)]


def _total_bytes(enc) -> int:
    return sum(len(s) for ss in enc["strings"] for s in ss)


# Entropy-coder CDF buffers rebuilt by ``update_for_gain``; cached per gain below.
_CDF_BUFS = ("_quantized_cdf", "_cdf_length", "_offset")


def _entropy_mods(net):
    """Return the entropy modules whose rANS tables ``update_for_gain`` rebuilds."""
    return [
        m
        for m in (
            getattr(net, "entropy_bottleneck", None),
            getattr(net, "gaussian_conditional", None),
        )
        if m is not None
    ]


def _set_gain(net, gain) -> None:
    """Run ``net.update_for_gain(gain)``, memoised per distinct gain.

    Rebuilding the CDF tables dominates the encode cost, and the 64 table gains recur
    across images and search steps. The (read-only) buffers are cached the first time
    a gain is built and restored by reference afterwards; they are identical to what a
    rebuild would produce, so the emitted bytes do not change. The cache lives on the
    module instance, which makes it not thread-safe.
    """
    cache = net.__dict__.setdefault("_gain_cdf_cache", {})
    key = round(float(gain), 9)
    snap = cache.get(key)
    if snap is not None:
        for m, bufs in zip(_entropy_mods(net), snap):
            for name, val in bufs.items():
                m._buffers[name] = val
        return
    net.update_for_gain(gain)
    cache[key] = [
        {n: getattr(m, n) for n in _CDF_BUFS if getattr(m, n, None) is not None}
        for m in _entropy_mods(net)
    ]


@torch.no_grad()
def measure_z_floor(net, res: int, device="cuda") -> int:
    """Measure the hyperprior (z) byte count of a random crop at the minimum gain."""
    gain = gain_table(net)[0]
    x, _ = pad_to_multiple(torch.rand(1, 3, res, res, device=device))
    net.update_for_gain(gain)
    enc = net.compress(x, inputscale=gain, res=res)
    return sum(len(s) for s in enc["strings"][1])


def _fits(measurement, target: int) -> bool:
    """Return True when a measured stream fits ``target`` and the container.

    The container stores each rANS string behind a 2-byte length prefix, so a stream
    with a string longer than :data:`container.MAX_STRING_BYTES` cannot be packed
    whatever the budget. For ``target <= 65535`` the second condition is implied by
    the first.
    """
    enc, size = measurement[2], measurement[3]
    return size <= target and all(
        len(s) <= container.MAX_STRING_BYTES for ss in enc["strings"] for s in ss
    )


@torch.no_grad()
def _search_gain(net, x, res, target, table):
    """Largest table gain whose real rANS size is ``<= target`` (binary search).

    A stream also counts as not fitting when one of its strings is too long for
    the container's 2-byte length prefix (see :func:`_fits`).

    Returns ``(best, last)`` as ``(idx, gain, enc, size)`` tuples. ``best`` is
    ``None`` when even the lowest gain overflows the target; ``last`` is then the
    stream at the lowest gain (the smallest possible).
    """

    def measure(idx):
        g = table[idx]
        _set_gain(net, g)
        enc = net.compress(x, inputscale=g, res=res)
        return (idx, g, enc, _total_bytes(enc))

    lo, hi = 0, N_RATE_LEVELS - 1
    best = None
    last = None
    while lo <= hi:
        mid = (lo + hi) // 2
        last = measure(mid)
        if _fits(last, target):
            best = last  # fits: try a higher gain (more bytes, better quality)
            lo = mid + 1
        else:
            hi = mid - 1  # overflows: go lower
    return best, last


def budget_overhead(res: int, sc_len: int, paper_compat: bool = False) -> int:
    """Bytes reserved outside the rANS strings for a given resolution.

    ``7 + sc_len``, plus the 4-byte raw-geometry trailer at resolutions without a
    bucket code unless ``paper_compat`` is set.
    """
    overhead = container.HEADER_OVERHEAD + sc_len
    if not paper_compat and container.needs_raw_geometry(res):
        overhead += container.RAW_GEOM_TRAILER
    return overhead


def min_container_bytes(res: int, sc_len: int) -> int:
    """Size of the smallest (identity-only) container at ``res`` px.

    ``3 + sc_len``, plus the 4-byte raw-geometry trailer at resolutions without a
    bucket code. With ``paper_compat=False`` smaller budgets are rejected.
    """
    return container.packed_size(res, sc_len, 0, 0, identity_only=True)


@torch.no_grad()
def encode_to_budget(
    net,
    x: torch.Tensor,
    budget: int,
    res: int,
    sidechannel: bytes = b"",
    paper_compat: bool = False,
    overflow: str | None = None,
):
    """Encode a single padded crop ``x`` ``(1, 3, Hp, Wp)`` to a budgeted container.

    Parameters
    ----------
    net
        A codec variant (in eval mode, on the device of ``x``).
    x
        The crop in ``[0, 1]``, edge-padded to a multiple of 128.
    budget
        Byte budget.
    res
        True (unpadded) side length of the crop.
    sidechannel
        Identity side-channel bytes (ACCURATE; computed on the unpadded crop).
    paper_compat
        Use the paper's budget accounting (see the module docstring).
    overflow
        ``"floor"``, ``"identity"`` or ``"error"``; ``None`` selects ``"floor"`` with
        ``paper_compat=True`` and ``"identity"`` otherwise.

    Returns
    -------
    tuple[bytes, dict]
        The container and an info dict with ``fitted`` (a spatial stream fit the
        search target), ``over_budget`` (the lowest-gain stream was emitted although
        it does not fit), ``identity_only``, ``within_budget``
        (``len(container) <= budget``), ``rate_index``, ``gain`` and ``bytes``. With
        ``paper_compat=True``, ``fitted`` does not imply ``within_budget``: at
        raw-geometry resolutions a fitted container can exceed the budget by the
        4-byte trailer, so compute budget-fit shares from ``within_budget`` (or the
        byte length), not from ``fitted``.

    Raises
    ------
    ValueError
        If ``budget < 1``, or with ``paper_compat=False`` if ``budget`` is below
        :func:`min_container_bytes`.
    BudgetError
        With ``overflow="error"`` when no spatial stream fits.
    """
    if overflow is None:
        overflow = "floor" if paper_compat else "identity"
    if overflow not in OVERFLOW_POLICIES:
        raise ValueError(
            f"overflow must be one of {OVERFLOW_POLICIES}, got {overflow!r}"
        )
    budget = int(budget)
    if budget < 1:
        raise ValueError(f"budget must be a positive number of bytes, got {budget}")
    variant_id = getattr(net, "VARIANT_ID", 0)
    target = budget - budget_overhead(res, len(sidechannel), paper_compat)
    if not paper_compat:
        smallest = min_container_bytes(res, len(sidechannel))
        if budget < smallest:
            raise ValueError(
                f"budget {budget} B is below the smallest (identity-only) container "
                f"of {smallest} B at {res} px with a {len(sidechannel)} B "
                "side-channel"
            )

    def emit_identity_only():
        packed = container.pack(
            container.VARIANT_IDENTITY_ONLY, 0, res, sidechannel, b"", b""
        )
        return packed, {
            "fitted": False,
            "over_budget": False,
            "identity_only": True,
            "within_budget": len(packed) <= budget,
            "rate_index": 0,
            "gain": 0.0,
            "bytes": len(packed),
        }

    if target <= 0:
        # No room for a spatial latent after the container header + side-channel.
        if overflow == "error":
            raise BudgetError(
                f"budget {budget} B leaves no room for a spatial latent at {res} px"
            )
        return emit_identity_only()

    # The identity side-channel is never dropped: the ACCURATE decoder's FiLM and
    # refinement head require it.
    table = gain_table(net)
    best, last = _search_gain(net, x, res, target, table)

    if best is None:
        if overflow == "error":
            if last[3] <= target:
                raise BudgetError(
                    f"the lowest-gain stream at {res} px has a rANS string longer "
                    f"than the container limit of {container.MAX_STRING_BYTES} B"
                )
            raise BudgetError(
                f"lowest-gain stream needs {last[3]} B of rANS payload, "
                f"target is {target} B (budget {budget} B at {res} px)"
            )
        if overflow == "identity":
            return emit_identity_only()
        # overflow == "floor": emit the smallest spatial stream, flagged over budget.
        idx, g, enc, size = last
        packed = container.pack(
            variant_id,
            idx,
            res,
            sidechannel,
            enc["strings"][0][0],
            enc["strings"][1][0],
        )
        return packed, {
            "fitted": False,
            "over_budget": True,
            "identity_only": False,
            "within_budget": len(packed) <= budget,
            "rate_index": idx,
            "gain": g,
            "bytes": len(packed),
        }

    idx, g, enc, size = best
    packed = container.pack(
        variant_id, idx, res, sidechannel, enc["strings"][0][0], enc["strings"][1][0]
    )
    return packed, {
        "fitted": True,
        "over_budget": False,
        "identity_only": False,
        "within_budget": len(packed) <= budget,
        "rate_index": idx,
        "gain": g,
        "bytes": len(packed),
    }


@torch.no_grad()
def decode_from_container(net, buf: bytes, device="cuda"):
    """Decode container bytes to a cropped RGB ``[0, 1]`` image ``(1, 3, res, res)``.

    Returns ``(x_hat, header)``. Identity-only containers (no spatial latent) return
    ``None`` for the image. Non-finite values (possible on the CPU float path) are
    replaced before clamping, so a NaN never propagates into downstream metrics.
    """
    h = container.unpack(buf)
    if h.variant_id == container.VARIANT_IDENTITY_ONLY:
        return None, h
    g = gain_table(net)[h.rate_index]
    d = net.downsampling_factor
    padded = math.ceil(h.res / d) * d
    shape = (padded // d, padded // d)
    _set_gain(net, g)
    side = None
    if getattr(net, "side", None) is not None:
        if h.sidechannel:
            side = net.side.decode(h.sidechannel)  # also stashes _last_p_hat
        else:
            net.side._last_p_hat = None  # no code: skip refine (avoid a stale p_hat)
    dec = net.decompress(
        [[h.y_string], [h.z_string]],
        shape,
        inputscale=g,
        res=h.res,
        side=side,
    )
    x_hat = dec["x_hat"][..., : h.res, : h.res]
    if not torch.isfinite(x_hat).all():
        x_hat = torch.nan_to_num(x_hat, nan=0.0, posinf=1.0, neginf=0.0)
    x_hat = x_hat.clamp_(0, 1)
    return x_hat, h
