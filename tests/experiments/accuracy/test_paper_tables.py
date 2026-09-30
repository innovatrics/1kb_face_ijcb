# SPDX-License-Identifier: MIT
"""Re-render the accuracy tables from the shipped results/ and compare with the paper.

Every rendered table is compared byte for byte with the arXiv v1 table source,
recorded here by its SHA-256 (first 40 hex digits). ``best_config.tex`` differs in
one documented cell: JPEG-AI at 1024 B is 0.71 in ``metrics.csv`` and 0.73 in the
paper; the test swaps that one value back before hashing.
"""

from __future__ import annotations

import hashlib

import pytest

from ._scripts import REPO, load

RESULTS = REPO / "results"

#: First 40 hex digits of the SHA-256 of the arXiv v1 table sources.
PAPER_SHA256 = dict(
    line.split()
    for line in """
best_config                            a80bdb47a27263eef696658d1f2a779a0c0e639e
frr_far_colorferet_1024                91a20ab4e670bb19982cd8d1310a409451b72f71
frr_far_colorferet_1024_224            94336b288062c6f61bad657719df9e711640721d
frr_far_colorferet_512                 ce723c56dda1d6f4eda0c52dabd2eb6aa622c686
frr_far_colorferet_512_224             da7c6f61813e96484ebcb8355e94b2fc1038d40a
frr_far_kk_1024                        b8ab2cea2092a8d8367d9b044937286f34cc2adb
frr_far_kk_1024_224                    b4f12a99b2025bc6bd5ef5b6843503f48e1e2669
frr_far_kk_512                         25f8dfefa6b71b3beeb86e4e3f50ba06c6196781
frr_far_kk_512_224                     e34ca11ac291541f8e81e4095920cb472b9a7cac
frr_far_summary                        26c6979250ade3b04d57a09a13c78ca77deff189
h4_budget                              af29db091258a736b9131cdf738ef20e31120faf
heldout_cvlface                        7b03f5108ea3372ab73a3806bf3fba75ef009e50
idcos_tail                             be2e62b03f89d4163a6fb0036b81001cf2a7d947
model_effect_colorferet_1024           60df346dcddf191f79f8aabd47be9aef06dc1315
model_effect_colorferet_1024_224       141a06bda77b9476ec8f8bc59c8a403bd1b82533
model_effect_kk_1024                   fe4d7a31d0c32736edf887b06aa4140f328e05d5
model_effect_kk_1024_224               58ecb401c10139dfa209f5cbd0bc5aba1f22f2b1
posthoc_stats                          f1838efd8780efd7bf4d8e8d300c68f8a50bb022
rate_eer_colorferet_arcface_antelopev2 af839ef3ab770b8f1aa06add129b78ffb982f50b
rate_eer_colorferet_edgeface_xs        61ad57aa55a579444017550b4bea312f41c75af4
rate_eer_colorferet_lvface_l           8019d73b717b7d42a9770a12bf8e901fc6b24544
rate_eer_colorferet_topofr_r100        0d17748d13eee122b9fb60e220f7c2fee2a4af8b
rate_eer_kk_arcface_antelopev2         b0e6fa4cfa29fb77de8c2c81f187b3bca9088fca
rate_eer_kk_edgeface_xs                2b9b27eb150ca517cc5afaab8890331b223340bf
rate_eer_kk_lvface_l                   e2d5823349b8ed865b76c2dc6c4815faa338b62f
rate_eer_kk_topofr_r100                1ed6662bd5cb779caa11685c4dc0eea3bdc24486
sig_cf_edgeface                        3e7f0439338dc311126c0361f3116aefd783c08a
sig_kk                                 8da3a23dcf434774c29bb4a74791751f54833dbc
""".strip().splitlines()
)

GENERATED = {
    "rate_eer": [s for s in PAPER_SHA256 if s.startswith("rate_eer_")],
    "frr_far": [s for s in PAPER_SHA256 if s.startswith(("frr_far_", "model_effect_"))],
    "h4_curve": ["h4_budget"],
    "heldout_table": ["heldout_cvlface"],
    "posthoc_stats": ["posthoc_stats"],
    "significance_tables": ["sig_cf_edgeface", "sig_kk"],
    "best_config": ["best_config"],
    "idcos_tail": ["idcos_tail"],
}
PAPER_HEADER = {"best_config": False, "idcos_tail": False}
EXTRA = {
    "h4_curve": ["--no-figures"],
    "frr_far": ["--no-figures"],
    "idcos_tail": ["--render-only"],
}
#: The one cell of best_config.tex where the paper and metrics.csv disagree.
BEST_CONFIG_CELL = ("JPEG-AI       & 0.71 @224 &", "JPEG-AI       & 0.73 @224 &")

needs_results = pytest.mark.skipif(
    not (RESULTS / "accuracy" / "metrics.csv").is_file(), reason="results/ not present"
)


def _render(script: str, out) -> None:
    args = ["--from-results", "--out-dir", str(out)] + EXTRA.get(script, [])
    if PAPER_HEADER.get(script, True):
        args.append("--paper-header")
    if script == "posthoc_stats":
        args += ["--scalars-dir", str(out)]
    assert load(script).main(args) == 0


def test_every_paper_table_is_covered():
    assert sorted(s for v in GENERATED.values() for s in v) == sorted(PAPER_SHA256)


@needs_results
@pytest.mark.parametrize("script", sorted(GENERATED))
def test_render_from_results(script, tmp_path):
    _render(script, tmp_path)
    if script == "posthoc_stats":
        want = (RESULTS / "accuracy" / "posthoc_scalars.txt").read_text()
        assert (tmp_path / "posthoc_scalars.txt").read_text() == want
    for stem in GENERATED[script]:
        data = (tmp_path / f"{stem}.tex").read_bytes()
        if stem == "best_config":
            ours, paper = (c.encode() for c in BEST_CONFIG_CELL)
            assert data.count(ours) == 1
            data = data.replace(ours, paper)
        assert hashlib.sha256(data).hexdigest()[:40] == PAPER_SHA256[stem], stem


@needs_results
def test_posthoc_without_edgeface(tmp_path):
    args = ["--from-results", "--exclude-prefix", "edgeface"]
    args += ["--out-dir", str(tmp_path), "--scalars-dir", str(tmp_path)]
    assert load("posthoc_stats").main(args) == 0
    text = (tmp_path / "posthoc_scalars_excl_edgeface.txt").read_text()
    assert text.startswith("n=10 k=12 friedman_chi2=98.3 ")
    assert "kendall_w=0.893" in text
