# SPDX-License-Identifier: MIT
"""Budget searches against straightforward reference implementations."""

from __future__ import annotations

import random

import pytest

from face1kb.baselines.search import (
    FitResult,
    binary_search_fit,
    hinted_binary_search_fit,
    scan_fit,
    sized,
)


def _ref_binary(encode, settings, budget):
    # The benchmark's search, written out as the reference.
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
        return settings[0], size, payload, False
    return best[0], best[1], best[2], True


def _ref_scan_largest(encode, settings, budget):
    # Exhaustive scan, larger-output-wins, errors skipped (difficulty study).
    best = None
    for s in settings:
        try:
            n, data = encode(s)
        except ValueError:
            continue
        if n <= budget and (best is None or n > best[1]):
            best = (s, n, data)
    if best is None:
        n, data = encode(settings[0])
        return settings[0], n, data, False
    return best[0], best[1], best[2], True


def _ref_scan_stop(encode, settings, budget):
    # Early-stopping scan, larger-output-wins (codec-comparison JPEG-FzT).
    best = None
    for s in settings:
        n, data = encode(s)
        if n <= budget and (best is None or n > best[1]):
            best = (s, n, data)
        elif n > budget:
            break
    if best is None:
        n, data = encode(settings[0])
        return settings[0], n, data, False
    return best[0], best[1], best[2], True


def _ref_fit_setting(encode, settings, budget):
    # Early-stopping scan, last-fitting-wins (speed benchmark).
    best = None
    for s in settings:
        if encode(s)[0] <= budget:
            best = s
        else:
            break
    return best if best is not None else settings[0]


def _table_encoder(sizes, calls=None):
    def enc(s):
        if calls is not None:
            calls.append(s)
        return sizes[s], f"payload-{s}".encode()

    return enc


def test_monotone_picks_largest_fitting():
    settings = list(range(2, 96, 2))
    sizes = {q: 100 + 10 * q for q in settings}
    r = binary_search_fit(_table_encoder(sizes), settings, 800)
    assert r == FitResult(70, 800, b"payload-70", True)
    assert isinstance(r, tuple) and len(r) == 4  # unpacks like the original tuple


def test_nothing_fits_returns_smallest_setting():
    settings = [400, 396, 392]
    sizes = {400: 900, 396: 950, 392: 990}
    calls = []
    r = binary_search_fit(_table_encoder(sizes, calls), settings, 512)
    assert r.setting == 400 and r.size == 900 and not r.fitted
    assert calls[-1] == 400  # the fallback is re-encoded


def test_everything_fits_returns_last():
    settings = list(range(1, 9))
    r = binary_search_fit(_table_encoder({q: q for q in settings}), settings, 10**6)
    assert r.setting == 8 and r.fitted


def test_probes_logarithmically():
    settings = list(range(1000))
    calls = []
    binary_search_fit(_table_encoder({s: s for s in settings}, calls), settings, 321)
    assert len(calls) <= 11


@pytest.mark.parametrize("seed", range(200))
def test_random_curves_match_reference(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 60)
    settings = list(range(n))
    # Mostly increasing curves with occasional dips (non-monotone encoders).
    size, sizes = rng.randint(50, 400), {}
    for s in settings:
        size += rng.randint(-15, 40)
        sizes[s] = max(size, 1)
    budget = rng.randint(40, 1500)
    enc = _table_encoder(sizes)
    assert tuple(binary_search_fit(enc, settings, budget)) == _ref_binary(
        enc, settings, budget
    )
    assert tuple(scan_fit(enc, settings, budget, skip_errors=True)) == (
        _ref_scan_largest(enc, settings, budget)
    )
    assert tuple(scan_fit(enc, settings, budget, stop_at_overflow=True)) == (
        _ref_scan_stop(enc, settings, budget)
    )
    r = scan_fit(enc, settings, budget, keep="last", stop_at_overflow=True)
    assert r.setting == _ref_fit_setting(enc, settings, budget)


def test_scan_skips_errors_only_when_asked():
    sizes = {1: 100, 2: 200, 3: 300}

    def enc(s):
        if s == 2:
            raise ValueError("encoder failure")
        return sizes[s], b"x" * sizes[s]

    r = scan_fit(enc, [1, 2, 3], 1000, skip_errors=True)
    assert r.setting == 3 and r.fitted
    with pytest.raises(ValueError):
        scan_fit(enc, [1, 2, 3], 1000)


def test_sized_and_validation():
    enc = sized(lambda s: b"a" * s)
    assert enc(5) == (5, b"aaaaa")
    with pytest.raises(ValueError):
        binary_search_fit(enc, [], 10)
    with pytest.raises(ValueError):
        scan_fit(enc, [1], 10, keep="first")


def _ref_hinted(enc, settings, budget, hint=None):
    # The warm-started search of the annex study, written out as the reference.
    def _bs(lo, hi):
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            n, d = enc(settings[mid])
            if n <= budget:
                best, lo = (mid, n, d), mid + 1
            else:
                hi = mid - 1
        return best

    top = len(settings) - 1
    if hint is not None:
        lo, hi = max(0, hint - 4), min(top, hint + 4)
        b = _bs(lo, hi)
        if b is not None and (lo < b[0] < hi or (b[0] == top) or (b[0] == 0 == lo)):
            return (*b, True)
    b = _bs(0, top)
    if b is None:
        n, d = enc(settings[0])
        return 0, n, d, False
    return (*b, True)


@pytest.mark.parametrize("seed", range(200))
def test_hinted_search_matches_reference(seed):
    rng = random.Random(1000 + seed)
    n = rng.randint(1, 60)
    settings = [400 - 4 * i for i in range(n)]
    size, sizes = rng.randint(50, 400), {}
    for s in settings:
        size += rng.randint(-15, 40)
        sizes[s] = max(size, 1)
    budget = rng.randint(40, 1500)
    hint = rng.choice([None, *range(n)])
    calls, ref_calls = [], []
    idx, r = hinted_binary_search_fit(
        _table_encoder(sizes, calls), settings, budget, hint
    )
    ref = _ref_hinted(_table_encoder(sizes, ref_calls), settings, budget, hint)
    assert (idx, r.size, r.payload, r.fitted) == ref
    assert r.setting == settings[idx] and calls == ref_calls


def test_hinted_search_saves_encodes():
    settings = list(range(2, 96, 2))
    sizes = {q: 100 + 10 * q for q in settings}
    full = []
    i0, r0 = hinted_binary_search_fit(_table_encoder(sizes, full), settings, 800)
    warm = []
    i1, r1 = hinted_binary_search_fit(_table_encoder(sizes, warm), settings, 800, i0)
    assert (i0, r0) == (i1, r1) and r1.setting == 70
    assert len(warm) < len(full)
