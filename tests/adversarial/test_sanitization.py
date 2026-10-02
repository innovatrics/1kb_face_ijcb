# SPDX-License-Identifier: MIT
"""Residual / sanitization metric of the compression-as-defence study (NumPy)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from face1kb import adversarial as adv


def unit2(deg):
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])


def test_valid_rows():
    e = np.array([[1.0, 0.0], [0.0, 0.0], [np.nan, 1.0], [0.0, -2.0]])
    assert adv.valid_rows(e).tolist() == [True, False, False, True]


def test_mean_cosine_is_scale_free():
    a = np.array([[1.0, 0.0], [0.0, 3.0]])
    b = np.array([[2.0, 0.0], [0.0, -1.0]])
    assert adv.mean_cosine(a, b, np.array([True, True])) == pytest.approx(0.0)
    assert adv.mean_cosine(a, b, np.array([True, False])) == pytest.approx(1.0)
    assert math.isnan(adv.mean_cosine(a, b, np.array([False, False])))


def test_record_known_values():
    # every crop: clean at 0 deg, adv at 60, codec(clean) at 20, codec(adv) at 40
    clean = np.stack([unit2(0)] * 3)
    advs = np.stack([unit2(60)] * 3)
    cc = np.stack([unit2(20)] * 3)
    ca = np.stack([unit2(40)] * 3)
    r = adv.sanitization_record(clean, advs, cc, ca)
    a, b, c = (math.cos(math.radians(d)) for d in (60, 20, 40))
    assert r["n"] == 3
    assert r["cos_adv"] == pytest.approx(a)
    assert r["cos_clean_comp"] == pytest.approx(b)
    assert r["cos_adv_comp"] == pytest.approx(c)
    assert r["residual"] == pytest.approx(b - c)
    assert r["sanitization"] == pytest.approx((c - a) / (b - a))
    assert tuple(r) == adv.METRIC_COLUMNS


def test_common_mask_over_all_four_arrays():
    clean = np.stack([unit2(0), unit2(0), unit2(0), unit2(0)])
    advs = np.stack([unit2(90), unit2(60), unit2(60), unit2(60)])
    cc = np.stack([unit2(0), unit2(10), np.zeros(2), unit2(10)])  # crop 2 missing
    ca = np.stack([unit2(0), unit2(30), unit2(30), [np.nan, np.nan]])  # crop 3 NaN
    a, b, c, n = adv.cell_cosines(clean, advs, cc, ca)
    # only crops 0 and 1 are valid in every array
    assert n == 2
    assert a == pytest.approx((0.0 + 0.5) / 2)
    assert b == pytest.approx((1.0 + math.cos(math.radians(10))) / 2)
    assert c == pytest.approx((1.0 + math.cos(math.radians(30))) / 2)


def test_missing_cell():
    clean = np.stack([unit2(0)] * 2)
    for missing in range(3):
        arrays = [np.stack([unit2(10)] * 2) for _ in range(3)]
        arrays[missing] = None
        r = adv.sanitization_record(clean, *arrays)
        assert r["n"] == 0
        assert all(math.isnan(r[k]) for k in adv.METRIC_COLUMNS if k != "n")
    zero = np.zeros((2, 2))
    assert adv.cell_cosines(clean, zero, zero, zero)[3] == 0


def test_no_attack_gap_gives_nan_sanitization():
    clean = np.stack([unit2(0)] * 2)
    same = np.stack([unit2(30)] * 2)
    r = adv.sanitization_record(clean, same, same, np.stack([unit2(40)] * 2))
    assert math.isnan(r["sanitization"])
    assert r["residual"] == pytest.approx(
        math.cos(math.radians(30)) - math.cos(math.radians(40))
    )
