# SPDX-License-Identifier: MIT
"""Pure logic of the quality, codec-comparison and fairness generators (CPU)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from face1kb import config

from ._scripts import load


# ------------------------------------------------------------------ quality matrix
def _quality_frame():
    rows = []
    for codec, n, base in (
        ("webp", 1000, 30.0),
        ("jpeg", 1000, 25.0),
        ("ours_accurate", 1000, 29.0),
        ("neural_mbt2018_mean", 300, 35.0),  # low support: never shaded
    ):
        for res in (112, 224):
            rows.append(
                {
                    "dataset": "colorferet",
                    "codec": codec,
                    "res": res,
                    "budget": 1024,
                    "n": n,
                    "psnr": base,
                    "ssim": base / 40,
                    "ms_ssim": base / 40,
                    "lpips": 1 / base,
                    "dists": 1 / base,
                }
            )
    return pd.DataFrame(rows)


def test_low_support_rule():
    qt = load("quality", "quality_tables")
    rows = [("a", {"n": 1000}), ("b", {"n": 1000}), ("c", {"n": 499})]
    assert qt.low_support(rows) == {"c"}
    assert qt.low_support([("a", {"n": 10}), ("b", {"n": 10})]) == set()


def test_quality_matrix_shading_and_dagger():
    qt = load("quality", "quality_tables")
    tex = qt.quality_matrix(_quality_frame())
    mbt = [ln for ln in tex.splitlines() if ln.startswith("mbt2018")]
    assert len(mbt) == 2 and all("$^{\\ddagger}$" in ln for ln in mbt)
    assert all("cellcolor" not in ln for ln in mbt)
    webp = next(ln for ln in tex.splitlines() if ln.startswith("WebP & 112"))
    # best PSNR/SSIM/MS-SSIM and best (lowest) LPIPS/DISTS of the ranked rows
    assert webp.count("\\cellcolor{green!25}\\textbf{") == 5
    jpeg = next(ln for ln in tex.splitlines() if ln.startswith("JPEG & 112"))
    assert jpeg.count("\\cellcolor{red!22}\\underline{") == 5
    # rows are sorted by SSIM within a block
    order = [ln.split(" & ")[0] for ln in tex.splitlines() if " & 112 & " in ln]
    assert order == ["mbt2018$^{\\ddagger}$", "WebP", "Ours-ACCURATE", "JPEG"]


def test_budget_sweep_rows():
    qt = load("quality", "quality_tables")
    df = _quality_frame()
    df = pd.concat([df, df.assign(budget=512, psnr=df.psnr - 3)], ignore_index=True)
    tex = qt.budget_sweep({"colorferet": df, "kk": df.assign(dataset="kk")})
    lines = [ln for ln in tex.splitlines() if ln.startswith(("Color", "AI-"))]
    assert len(lines) == 8
    assert (
        lines[0] == "AI-Solutions-KK & WebP & 1024 & 30.00 & 0.750 & 0.033 & 1,000 \\\\"
    )


# ------------------------------------------------------------------ FIQ
def test_fiq_shading_rules():
    fr = load("quality", "fiq_report")
    vals = {"original": 0.99, "a": 0.9571, "b": 0.9574, "c": 0.9000, "d": None}
    assert fr.shade_column(vals, "raw") == {"b": "best", "c": "worst"}
    # a and b print as 0.957: both are shaded under the printed rule
    assert fr.shade_column(vals, "printed") == {"a": "best", "b": "best", "c": "worst"}
    assert fr.shade_column({"a": 0.95001, "b": 0.95002}, "printed") == {}
    with pytest.raises(ValueError):
        fr.shade_column(vals, "other")


def test_fiq_augmentation():
    pytest.importorskip("cv2")
    fiq = load("quality", "fiq")
    a, b = fiq.aug_params(3, 4, 1), fiq.aug_params(3, 4, 1)
    assert a == b and len(a) == 3 and len(a[0]) == 4
    img = (np.arange(112 * 112 * 3) % 251).astype(np.uint8).reshape(112, 112, 3)
    same = {
        "flip": False,
        "scale": 1.0,
        "dx": 0.0,
        "dy": 0.0,
        "gamma": 1.0,
        "bright": 1.0,
    }
    assert np.array_equal(fiq.apply_aug(img, same), img)
    flipped = fiq.apply_aug(img, {**same, "flip": True})
    assert np.array_equal(flipped, img[:, ::-1])
    emb = np.tile(np.arange(1, 9, dtype=np.float32), (5, 1))
    assert fiq.image_fiq(emb) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------ codec comparison
def test_codec_comparison_tables(tmp_path):
    cc = load("codec_comparison", "tables")
    for res in (112, 224):
        d = tmp_path / "codec_comparison" / f"res{res}"
        d.mkdir(parents=True)
        rows = [
            {
                "dataset": ds,
                "codec": c,
                "budget": b,
                "n": 4,
                "id_cos": v,
                "psnr": 1.0,
                "ssim": 1.0,
                "lpips": 1.0,
                "bytes": b - 10,
            }
            for ds in ("colorferet", "kk")
            for b in (1024, 512)
            for c, v in (("webp", 0.9), ("ours_accurate", 0.95), ("jpeg", 0.8))
        ]
        (d / "comparison.json").write_text(json.dumps(rows))
    (tmp_path / "quality").mkdir()
    pd.DataFrame(
        [
            {
                "dataset": "colorferet",
                "codec": "webp",
                "res": 112,
                "budget": 1024,
                "n": 9,
                "psnr": 33.0,
                "ssim": 0.9,
                "ms_ssim": 0.9,
                "lpips": 0.1,
                "dists": 0.1,
            },
        ]
    ).to_csv(tmp_path / "quality" / "quality_summary.csv", index=False)
    assert cc.main(["--input-root", str(tmp_path), "--out-dir", str(tmp_path)]) == 0
    tex = (tmp_path / "codec_comparison_112.tex").read_text()
    webp = next(ln for ln in tex.splitlines() if ln.startswith("Color~FERET & WebP"))
    assert "33.0" in webp
    jpeg = next(ln for ln in tex.splitlines() if ln.startswith("Color~FERET & JPEG "))
    assert jpeg.endswith("& -- & -- & -- \\\\")  # no quality row: fidelity is --
    ours = next(
        ln for ln in tex.splitlines() if ln.startswith("Color~FERET & Ours-ACCURATE")
    )
    assert "\\cellcolor{green!25}\\textbf{0.950}" in ours
    first = [ln for ln in tex.splitlines() if ln.startswith("KK")][0]
    assert "Ours-ACCURATE" in first  # sorted by id-cos
    res = (tmp_path / "codec_results.tex").read_text()
    assert res.count("\\\\") == 6 + 2  # 3 header lines + 6 codec rows (2 datasets)


# ------------------------------------------------------------------ fairness tables
def _write_fairness_csvs(root):
    fdir = root / "fairness"
    fdir.mkdir(parents=True)
    rows, disp, fmr = [], [], []
    models = ("arcface_antelopev2", "lvface_l", "topofr_r100", "edgeface_xs")
    for m in models:
        for codec, budget, k in (
            ("aligned", 0, 1.0),
            ("webp", 1024, 2.0),
            ("jpeg", 1024, 3.0),
            ("jpeg_ai", 1024, 0.0),
        ):
            for i, g in enumerate(("MST5", "MST6", "MST7", "MST9", "MST10")):
                rows.append(
                    {
                        "dataset": "kk",
                        "model": m,
                        "res": 112,
                        "budget": budget,
                        "codec": codec,
                        "attribute": "skin_tone",
                        "subgroup": g,
                        "n_pos": 100,
                        "n_neg": 5000,
                        "eer": 0.001 * k * (i + 1),
                    }
                )
            for attr, n in (("skin_tone", 4), ("pose", 4)):
                disp.append(
                    {
                        "dataset": "colorferet",
                        "model": m,
                        "res": 112,
                        "budget": budget,
                        "codec": codec,
                        "attribute": attr,
                        "eer_min": 0.001,
                        "eer_max": 0.001 + 0.01 * k,
                        "eer_std": 0.0,
                        "n_subgroups": n,
                    }
                )
            fmr.append(
                {
                    "dataset": "colorferet",
                    "model": m,
                    "res": 112,
                    "budget": budget,
                    "codec": codec,
                    "attribute": "skin_tone",
                    "subgroup": "__overall__",
                    "n_neg": 1000,
                    "tau": 0.3,
                    "target_fmr": 0.01,
                    "fmr": 0.01,
                    "fmr_min": 0.0,
                    "fmr_max": 0.0,
                    "fmr_disparity_pp": 0.01 * k,
                    "n_subgroups": 4,
                }
            )
    kk = pd.DataFrame(rows)
    kk.to_csv(fdir / "fairness_kk.csv", index=False)
    kk.assign(dataset="colorferet").to_csv(
        fdir / "fairness_colorferet.csv", index=False
    )
    pd.DataFrame(disp).to_csv(fdir / "fairness_disparity_colorferet.csv", index=False)
    pd.DataFrame(fmr).to_csv(fdir / "fmr_fairness_colorferet.csv", index=False)
    ci = []
    for m in models:
        ci.append(
            {
                "anchor": m,
                "codec": "aligned",
                "disparity_pp": 0.5,
                "disp_lo": 0.2,
                "disp_hi": 0.9,
                "n_boot": 500,
            }
        )
        for codec, r in (
            ("webp", 1.5),
            ("jpeg2000", 9.0),
            ("ours_accurate", 1.2),
            ("ours_fast", 2.0),
        ):
            ci.append(
                {
                    "anchor": m,
                    "codec": codec,
                    "disparity_pp": 0.5 * r,
                    "disp_lo": 0.1,
                    "disp_hi": 2.0,
                    "n_boot": 500,
                    "ratio": r,
                    "ratio_lo": r / 2,
                    "ratio_hi": r * 2,
                }
            )
    pd.DataFrame(ci).to_csv(fdir / "disparity_ci_kk.csv", index=False)


def test_fairness_tables_synthetic(tmp_path):
    ft = load("fairness", "tables")
    _write_fairness_csvs(tmp_path)
    out = tmp_path / "tables"
    assert ft.main(["--input-root", str(tmp_path), "--out-dir", str(out)]) == 0
    kk = (out / "fairness_disparity_kk.tex").read_text()
    # max - min over MST5..MST10: 0.001 * k * (5 - 1) -> 0.4 k pp
    assert "base (aligned)" in kk and "0.40 & 0.40 & 0.40 & 0.40" in kk
    assert "JPEG-AI" in kk and "0.00 & 0.00" in kk
    # FMR table: a codec that prints 0.00 for every anchor is omitted
    fmr = (out / "fmr_fairness_cf.tex").read_text()
    assert "JPEG-AI" not in fmr and "WebP" in fmr
    assert "\\cellcolor{green!25}\\textbf{0.02}" in fmr  # WebP (base never competes)
    cf = (out / "fairness_disparity_cf.tex").read_text()
    assert "(4 subgroups)" in cf and "\\cellcolor{red!22}\\underline{3.00}" in cf
    ci = (out / "disparity_ci_kk.tex").read_text()
    assert "\\emph{aligned} $\\Delta$ (pp) & 0.50 & 0.50 & 0.50 & 0.50" in ci
    assert "\\cellcolor{green!25}\\textbf{1.2 [0.6,2.4]}" in ci
    s512 = (out / "fairness_disparity_512.tex").read_text()
    assert "cellcolor" not in s512  # published without shading
    sub = (out / "fairness_subgroup_kk.tex").read_text()
    assert sub.splitlines()[2].startswith("Subgroup & base & WebP")
    assert "MST8  & --- & --- & --- & --- & --- & --- & ---" in sub


# ------------------------------------------------------------------ fairness drivers
@pytest.fixture
def synth(tmp_path, monkeypatch):
    """Tiny public-layout datasets with one matcher and three sources."""
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(config, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(config, "LAYOUT", "public")
    rng = np.random.default_rng(0)
    n_subj, per = 12, 6
    subj = np.repeat(np.arange(1, n_subj + 1), per)
    for ds in ("colorferet", "kk"):
        rel = [
            f"{s:05d}/{s:05d}_{k}.png"
            for s, k in zip(subj, np.tile(range(per), n_subj))
        ]
        config.dataset_dir(ds).mkdir(parents=True)
        pd.DataFrame(
            {
                "id": range(subj.size),
                "rel_path": rel,
                "subject": [f"{s:05d}" for s in subj],
            }
        ).to_csv(config.index_csv(ds), index=False)
        i1, i2 = np.triu_indices(subj.size, 1)
        pd.DataFrame(
            {
                "idx1": i1.astype(np.int32),
                "idx2": i2.astype(np.int32),
                "label": (subj[i1] == subj[i2]).astype(np.int8),
            }
        ).to_parquet(config.pairs_parquet(ds), index=False)
        centers = rng.normal(size=(n_subj, 512))
        config.embeddings_dir(ds, "arcface_antelopev2").mkdir(parents=True)
        for tag, noise in (
            ("aligned_112", 0.5),
            ("webp_112_1024", 1.0),
            ("webp_112_512", 1.5),
        ):
            e = centers[subj - 1] + noise * rng.normal(size=(subj.size, 512))
            np.save(
                config.embeddings_path(ds, "arcface_antelopev2", tag),
                e.astype(np.float32),
            )
    stems = [f"{s:05d}_{k}" for s, k in zip(subj, np.tile(range(per), n_subj))]
    pd.DataFrame(
        {
            "image": stems,
            "pose": np.where(np.arange(subj.size) % 2, "profile left", "frontal image"),
            "gender": np.where(subj % 2, "male", "female"),
            "race": np.where(subj <= 6, "White", "Asian"),
            "age_from": 20 + 3 * subj,
            "age_to": 20 + 3 * subj,
        }
    ).to_csv(config.labels_csv("colorferet"), index=False)
    pd.DataFrame(
        {
            "subject": [f"{s:05d}" for s in subj],
            "mst_label": [f"MST{5 + s % 2}" for s in subj],
            "gender": np.where(subj % 2, "M", "F"),
            "age": 30.0,
        }
    ).to_csv(config.attributes_csv("kk"), index=False)
    return tmp_path


def test_fairness_drivers_end_to_end(synth):
    sub = load("fairness", "subgroup")
    fmr = load("fairness", "fmr_fairness")
    base = [
        "--models",
        "arcface_antelopev2",
        "--device",
        "cpu",
        "--sample-nonmated",
        "500",
        "--min-mated",
        "5",
    ]
    assert sub.main(base) == 0
    f = pd.read_csv(config.output_dir("fairness") / "fairness_colorferet.csv")
    assert list(f.columns) == [
        "dataset",
        "model",
        "res",
        "budget",
        "codec",
        "attribute",
        "subgroup",
        "n_pos",
        "n_neg",
        "eer",
    ]
    assert set(f.codec) == {"aligned", "webp"} and set(f.budget) == {0, 512, 1024}
    d = pd.read_csv(config.output_dir("fairness") / "fairness_disparity_kk.csv")
    assert list(d.columns)[-1] == "n_subgroups"
    assert (
        fmr.main(
            [
                *base[:4],
                "--sample-nonmated",
                "500",
                "--min-neg",
                "5",
                "--budgets",
                "1024",
            ]
        )
        == 0
    )
    m = pd.read_csv(config.output_dir("fairness") / "fmr_fairness_kk.csv")
    assert set(m.budget) == {0, 1024} and "fmr_disparity_pp" in m.columns


def test_disparity_ci_requires_aligned_first(synth):
    ci = load("fairness", "disparity_ci")
    with pytest.raises(SystemExit):
        ci.main(["--codecs", "webp,aligned", "--device", "cpu"])


# ------------------------------------------------------------------ montage samples
def test_visual_samples(synth):
    vq = load("quality", "visual_quality")
    assert vq.parse_samples("12:Left,00003/00003_1") == [
        ("12", "Left"),
        ("00003/00003_1", "00003/00003_1"),
    ]
    got = vq.resolve_samples("kk", [(7, "x"), ("00001/00001_0", "y")])
    assert got == [("00002/00002_1", "x"), ("00001/00001_0", "y")]
    with pytest.raises(SystemExit):
        vq.resolve_samples("kk", [(10_000, "missing")])
