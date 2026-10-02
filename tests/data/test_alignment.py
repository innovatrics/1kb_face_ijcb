# SPDX-License-Identifier: MIT
"""ArcFace template math, crop-tightness variants and resolution derivation."""

from __future__ import annotations

import numpy as np
import pytest

from face1kb.data import alignment as al

cv2 = pytest.importorskip("cv2")

# dst_template values of the paper's crop-tightness presets (rounded to 3 decimals)
PUBLISHED_VARIANTS = {
    "tight": [
        [33.6, 38.204],
        [78.4, 37.956],
        [56.142, 63.682],
        [37.738, 89.909],
        [74.837, 89.704],
    ],
    "mid": [
        [30.24, 30.382],
        [81.76, 30.098],
        [56.164, 59.683],
        [34.999, 89.844],
        [77.663, 89.608],
    ],
    "fill": [
        [26.88, 22.561],
        [85.12, 22.239],
        [56.185, 55.683],
        [32.26, 89.778],
        [80.489, 89.511],
    ],
}


def _similarity(scale, angle_deg, tx, ty):
    a = np.deg2rad(angle_deg)
    c, s = scale * np.cos(a), scale * np.sin(a)
    return np.array([[c, -s, tx], [s, c, ty]])


def _apply(m, pts):
    return pts @ m[:, :2].T + m[:, 2]


def _source(h=300, w=260, seed=0):
    """Smooth random colour image (band-limited, so interpolation error is small)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (h // 10, w // 10, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def test_template_scaling():
    assert al.template(112) is not al.ARCFACE_DST_112
    assert np.array_equal(al.template(112), al.ARCFACE_DST_112)
    assert np.allclose(al.template(224), 2 * al.ARCFACE_DST_112)
    assert al.STANDARD_IOD_RATIO == pytest.approx(0.3146, abs=1e-4)


def test_similarity_transform_recovers_known_transform():
    m = _similarity(1.7, 12.0, -30.0, 11.0)
    src = np.array([[10.0, 20.0], [60.0, 22.0], [35.0, 50.0], [15.0, 80.0], [58, 79]])
    got = al.similarity_transform(src, _apply(m, src))
    assert np.allclose(got, m, atol=1e-10)


def test_similarity_matches_skimage():
    tf = pytest.importorskip("skimage.transform")
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 500, (5, 2))
    if hasattr(tf.SimilarityTransform, "from_estimate"):
        t = tf.SimilarityTransform.from_estimate(src, al.ARCFACE_DST_112)
    else:
        t = tf.SimilarityTransform()
        t.estimate(src, al.ARCFACE_DST_112)
    assert np.allclose(al.similarity_transform(src, al.ARCFACE_DST_112), t.params[:2])


def test_alignment_matrix_scales_with_size():
    m = _similarity(0.3, -5, 20, 40)
    lm = _apply(np.linalg.inv(np.vstack([m, [0, 0, 1]]))[:2], al.ARCFACE_DST_112)
    assert np.allclose(al.alignment_matrix(lm, 112), m)
    assert np.allclose(al.alignment_matrix(lm, 56), m * 0.5)


def test_variant_templates_match_published():
    for name, ref in PUBLISHED_VARIANTS.items():
        assert np.array_equal(al.variant_template(name).round(3), np.array(ref))
    meta = al.variants_metadata()
    assert meta["standard_iod_ratio"] == 0.315
    assert meta["variants"]["mid"]["dst_template"] == PUBLISHED_VARIANTS["mid"]
    assert set(meta) == {"resolution", "variants", "standard_iod_ratio"}
    assert set(meta["variants"]["fill"]) == {
        "iod_ratio",
        "eye_line_frac",
        "dst_template",
        "std_field_of_view",
    }


def test_variant_from_standard_maps_template():
    for name in al.CROP_VARIANTS:
        g = al.variant_from_standard(name)
        assert np.allclose(_apply(g, al.ARCFACE_DST_112), al.variant_template(name))
        x0, y0, x1, y1 = al.variant_field_of_view(name)
        assert 0 <= x0 < x1 <= 112 and 0 <= y0 < y1 <= 112


def test_variant_equals_composed_fit():
    """Fitting source landmarks to a variant template == G o M_std (equivariance)."""
    rng = np.random.default_rng(2)
    lm = rng.uniform(100, 300, (5, 2))
    m_std = al.similarity_transform(lm, al.ARCFACE_DST_112)
    for name in al.CROP_VARIANTS:
        m_var = al.similarity_transform(lm, al.variant_template(name))
        g = np.vstack([al.variant_from_standard(name), [0, 0, 1]])
        assert np.allclose(m_var, (g @ np.vstack([m_std, [0, 0, 1]]))[:2], atol=1e-9)


def test_render_variant_close_to_direct_warp():
    src = _source()
    lm = _apply(_similarity(2.2, 4.0, 18.0, 30.0), al.ARCFACE_DST_112)
    std224 = al.warp_to_template(src, lm, 224)
    m_std = al.alignment_matrix(lm, 112)
    for name in al.CROP_VARIANTS:
        g = np.vstack([al.variant_from_standard(name), [0, 0, 1]])
        m_var = (g @ np.vstack([m_std, [0, 0, 1]]))[:2]
        direct = cv2.warpAffine(src, m_var, (112, 112), flags=cv2.INTER_LINEAR)
        derived = al.render_variant(std224, name)
        assert derived.shape == (112, 112, 3) and derived.dtype == np.uint8
        mae = np.abs(derived.astype(float) - direct.astype(float)).mean()
        assert mae < 1.5, (name, mae)


@pytest.mark.parametrize("interp", ["INTER_LINEAR", "INTER_CUBIC"])
def test_integer_ratio_resolutions_are_exact(interp):
    flag = getattr(cv2, interp)
    src = _source(seed=3)
    lm = _apply(_similarity(2.0, -7.0, 25.0, 35.0), al.ARCFACE_DST_112)
    crops = {r: al.warp_to_template(src, lm, r, flag) for r in (56, 112, 224)}
    assert np.array_equal(crops[224][::2, ::2], crops[112])
    for base in (112, 224):
        out, exact = al.derive_resolution(crops[base], 56)
        assert exact and np.array_equal(out, crops[56])


def test_non_integer_ratio_is_flagged_approximate():
    src = _source(seed=4)
    lm = _apply(_similarity(2.0, 3.0, 20.0, 30.0), al.ARCFACE_DST_112)
    c224 = al.warp_to_template(src, lm, 224)
    direct80 = al.warp_to_template(src, lm, 80)
    out, exact = al.derive_resolution(c224, 80)
    assert not exact and out.shape == (80, 80, 3)
    assert np.abs(out.astype(float) - direct80.astype(float)).mean() < 2.0
    same, exact_same = al.derive_resolution(c224, 224)
    assert exact_same and np.array_equal(same, c224) and same is not c224
    assert al.is_exact_subsampling(224, 56) and not al.is_exact_subsampling(224, 96)
