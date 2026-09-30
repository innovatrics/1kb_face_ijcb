# SPDX-License-Identifier: MIT
"""Budget searches over a codec's rate knob.

Every baseline codec exposes one rate knob (a quality factor, a compression ratio, a
target bit rate or a model index). A *budget search* picks the setting whose output
is the largest one that still fits a byte budget ``B``. The benchmark uses
:func:`binary_search_fit`; :func:`scan_fit` implements the exhaustive or early-stopping
linear scans that some side studies use, and :func:`hinted_binary_search_fit` the
warm-started search of the ISO/IEC 29794-5 annex study. All work on a caller-supplied
``encode`` callable, so they are independent of the codec.

Fallback rule (all searches): when no setting fits, the smallest-output setting
``settings[0]`` is encoded and returned with ``fitted=False``. The benchmark keeps
that over-budget encode rather than dropping the crop ("failure to compress").
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

__all__ = [
    "FitResult",
    "binary_search_fit",
    "hinted_binary_search_fit",
    "scan_fit",
    "sized",
]


class FitResult(NamedTuple):
    """Outcome of a budget search.

    Attributes
    ----------
    setting
        The selected setting.
    size : int
        Size in bytes that the search compared against the budget (for most codecs
        ``len(payload)``; see the codec modules for exceptions).
    payload : bytes
        The encoded data of ``setting``.
    fitted : bool
        ``size <= budget``; ``False`` means that no setting fitted and ``setting`` is
        the smallest-output fallback ``settings[0]``.
    """

    setting: Any
    size: int
    payload: bytes
    fitted: bool


Encoder = Callable[[Any], "tuple[int, bytes]"]


def sized(encode: Callable[[Any], bytes]) -> Encoder:
    """Wrap ``encode(setting) -> bytes`` as ``encode(setting) -> (len, bytes)``."""

    def _enc(setting):
        data = encode(setting)
        return len(data), data

    return _enc


def binary_search_fit(
    encode: Encoder, settings: Sequence[Any], budget: int
) -> FitResult:
    """Binary-search the largest setting whose output fits ``budget``.

    Parameters
    ----------
    encode
        ``encode(setting) -> (size, payload)``.
    settings
        Settings ordered so that the output size is non-decreasing with the index
        (e.g. ascending JPEG quality, descending JPEG 2000 compression ratio).
    budget
        Byte budget.

    Returns
    -------
    FitResult
        The highest-index setting whose ``size <= budget`` among the settings the
        search probed, or ``settings[0]`` with ``fitted=False`` if none fitted.

    Notes
    -----
    The search probes ``O(log n)`` settings: ``lo, hi = 0, n - 1``; the midpoint
    ``(lo + hi) // 2`` moves ``lo`` up when it fits and ``hi`` down otherwise. With
    a non-monotone size curve it returns the fitting setting it last probed on
    that path, which need not be the global best; this is the benchmark behaviour.
    """
    if not settings:
        raise ValueError("settings must not be empty")
    lo, hi, best = 0, len(settings) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        size, payload = encode(settings[mid])
        if size <= budget:
            best = (settings[mid], size, payload)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        size, payload = encode(settings[0])
        return FitResult(settings[0], size, payload, False)
    return FitResult(best[0], best[1], best[2], True)


def _bisect(encode: Encoder, settings: Sequence[Any], budget: int, lo: int, hi: int):
    """Binary search on ``settings[lo..hi]``; return ``(index, size, payload)``."""
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        size, payload = encode(settings[mid])
        if size <= budget:
            best, lo = (mid, size, payload), mid + 1
        else:
            hi = mid - 1
    return best


def hinted_binary_search_fit(
    encode: Encoder,
    settings: Sequence[Any],
    budget: int,
    hint: int | None = None,
    window: int = 4,
) -> tuple[int, FitResult]:
    """Binary search warm-started from the answer for a previous, similar image.

    With ``hint`` (an index into ``settings``) the search first runs on the window
    ``[hint - window, hint + window]`` (clipped to the grid). Its answer is kept if
    it lies strictly inside the window, is the last setting of the grid, or is the
    first setting of a window that starts at the grid's beginning; otherwise, and
    without a hint, the full :func:`binary_search_fit` runs.

    Returns
    -------
    tuple[int, FitResult]
        The index of the selected setting (the next image's ``hint``) and the
        result. When nothing fits, index ``0`` with ``fitted=False``.
    """
    if not settings:
        raise ValueError("settings must not be empty")
    top = len(settings) - 1
    if hint is not None:
        lo, hi = max(0, hint - window), min(top, hint + window)
        b = _bisect(encode, settings, budget, lo, hi)
        if b is not None and (lo < b[0] < hi or b[0] == top or b[0] == 0 == lo):
            return b[0], FitResult(settings[b[0]], b[1], b[2], True)
    b = _bisect(encode, settings, budget, 0, top)
    if b is None:
        size, payload = encode(settings[0])
        return 0, FitResult(settings[0], size, payload, False)
    return b[0], FitResult(settings[b[0]], b[1], b[2], True)


def scan_fit(
    encode: Encoder,
    settings: Sequence[Any],
    budget: int,
    *,
    keep: str = "largest",
    stop_at_overflow: bool = False,
    skip_errors: bool = False,
) -> FitResult:
    """Linear scan over ``settings`` in order.

    Parameters
    ----------
    encode
        ``encode(setting) -> (size, payload)``.
    settings
        Settings in scan order (ascending output size for the stopping variant).
    budget
        Byte budget.
    keep
        ``"largest"``: keep the fitting output with the largest size (a later
        setting replaces the current best only if it is strictly larger);
        ``"last"``: keep the last fitting setting seen.
    stop_at_overflow
        Stop at the first setting whose output exceeds ``budget``.
    skip_errors
        Skip settings whose ``encode`` raises (the fallback encode of
        ``settings[0]`` is never guarded).

    Returns
    -------
    FitResult
        See :func:`binary_search_fit`; the fallback rule is the same.
    """
    if not settings:
        raise ValueError("settings must not be empty")
    if keep not in ("largest", "last"):
        raise ValueError(f"keep must be 'largest' or 'last', got {keep!r}")
    best = None
    for s in settings:
        try:
            size, payload = encode(s)
        except Exception:
            if skip_errors:
                continue
            raise
        if size <= budget:
            if best is None or keep == "last" or size > best[1]:
                best = (s, size, payload)
        elif stop_at_overflow:
            break
    if best is None:
        size, payload = encode(settings[0])
        return FitResult(settings[0], size, payload, False)
    return FitResult(best[0], best[1], best[2], True)
