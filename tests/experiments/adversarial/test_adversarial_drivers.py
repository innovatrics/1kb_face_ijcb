# SPDX-License-Identifier: MIT
"""Tests of the adversarial-study drivers in ``experiments/adversarial`` (CPU)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from face1kb import config

REPO = Path(__file__).resolve().parents[3]
DRIVERS = REPO / "experiments" / "adversarial"
ANCHORS = ("arcface_antelopev2", "lvface_l", "topofr_r100", "edgeface_xs")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_adv_{name}", DRIVERS / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tables():
    return _load("generate_sanitization_table")


@pytest.fixture(scope="module")
def analyze():
    return _load("analyze")


@pytest.fixture(scope="module")
def craft():
    return _load("craft")


@pytest.fixture
def public_roots(tmp_path, monkeypatch):
    """Point the config roots at empty temporary folders (public layout)."""
    for attr in ("DATA_ROOT", "WORK_ROOT", "OUTPUT_ROOT"):
        monkeypatch.setattr(config, attr, tmp_path / attr.lower())
    monkeypatch.setattr(config, "LAYOUT", "public")
    return tmp_path


# ------------------------------------------------------------------ tables
def _frame(rows):
    cols = [
        "dataset",
        "model",
        "attack",
        "eps",
        "codec",
        "budget",
        "n",
        "cos_adv",
        "cos_clean_comp",
        "cos_adv_comp",
        "residual",
        "sanitization",
    ]
    return pd.DataFrame(rows, columns=cols)


def _cell(model, codec, eps, budget, residual, n=100, cos_adv=0.6, attack="hfc"):
    return [
        "colorferet",
        model,
        attack,
        eps,
        codec,
        budget,
        n,
        cos_adv,
        0.8,
        0.8 - residual,
        residual,
        0.5,
    ]


def _grid(codec, base, models=ANCHORS, n=100, eps_list=(0.03, 0.06, 0.1)):
    rows = []
    for m in models:
        for i, e in enumerate(eps_list):
            for b in (512, 1024):
                rows.append(
                    _cell(m, codec, e, b, base + 0.01 * i + (b == 1024) * 0.1, n)
                )
    return rows


def test_residual_table_ranking_shading_and_coverage(tables):
    df = _frame(
        _grid("jpeg2000", 0.0)
        + _grid("webp", 0.2)
        + _grid("jpeg", 0.1)
        # fewer crops: daggered, never shaded
        + _grid("jpeg_fzt", -0.5, n=30)
        # fewer matchers: daggered
        + _grid("jpeg_ai", 0.9, models=ANCHORS[:2])
        # measured at eps 0.06 only: '---' elsewhere
        + _grid("ours_fast", 0.3, eps_list=(0.06,))
    )
    text = tables.residual_table(df, "colorferet", "hfc", "sanitization_hfc")
    lines = text.splitlines()
    assert lines[0] == tables.HEADER
    body = [ln for ln in lines if " & " in ln and not ln.startswith(("&", "Codec"))]
    names = [ln.split(" & ")[0] for ln in body]
    assert names == [
        "JPEG-FzT$^\\ddagger$",
        "JPEG~2000",
        "JPEG",
        "WebP",
        "Ours-FAST",
        "JPEG-AI$^\\ddagger$",
    ]
    fzt, j2k = body[0], body[1]
    assert "cellcolor" not in fzt and "$-0.500$" in fzt
    assert j2k.count("\\cellcolor{green!25}\\textbf{") == 6
    ours = body[4]
    assert ours.count("---") == 4
    # Ours-FAST is the worst ranked codec at eps 0.06; WebP elsewhere
    assert ours.count("\\cellcolor{red!22}\\underline{") == 2
    assert body[3].count("\\cellcolor{red!22}\\underline{") == 4
    assert "JPEG-AI" in body[5] and "cellcolor" not in body[5]


def test_residual_table_empty(tables):
    df = _frame(_grid("jpeg", 0.1))
    assert tables.residual_table(df, "kk", "hfc", "sanitization_hfc_kk") is None


def test_attack_strength_uses_full_coverage_cells(tables):
    rows = []
    for m in ANCHORS:
        for attack, v in (("hfc", 0.63), ("liae", 0.62), ("clip", 0.76)):
            for e in (0.03, 0.06, 0.1):
                rows.append(
                    _cell(m, "jpeg", e, 512, 0.1, n=100, cos_adv=v, attack=attack)
                )
                # a partial-coverage cell with another cos_adv must be ignored
                rows.append(
                    _cell(m, "jpeg_fzt", e, 512, 0.1, n=10, cos_adv=0.1, attack=attack)
                )
    text = tables.attack_strength(_frame(rows))
    lines = text.splitlines()
    assert lines[2] == (
        "Attack & $\\varepsilon=0.03$ & $\\varepsilon=0.06$ & $\\varepsilon=0.10$ \\\\"
    )
    liae = next(ln for ln in lines if ln.startswith("Li-AE"))
    assert liae.count("\\textbf{0.620}") == 3
    assert next(ln for ln in lines if ln.startswith("CLIP")).count("0.760") == 3


def test_ours_defense_bolds_column_minimum(tables):
    rows = []
    for attack in ("hfc", "clip", "liae"):
        for codec, v in (
            ("jpeg2000", -0.001),
            ("webp", 0.2),
            ("ours_fast", 0.25),
            ("ours_accurate", 0.3),
        ):
            for m in ANCHORS:
                for b in (512, 1024):
                    rows.append(_cell(m, codec, 0.06, b, v, attack=attack))
    text = tables.ours_defense(_frame(rows))
    j2k = next(ln for ln in text.splitlines() if ln.startswith("JPEG~2000"))
    assert j2k.count("\\textbf{$-0.001$}") == 6
    assert "\\multicolumn{2}{c}{Li-AE, $\\varepsilon=0.06$}" in text


def test_tables_cli(tables, tmp_path):
    df = _frame(_grid("jpeg", 0.1) + _grid("webp", 0.2))
    src = tmp_path / "sanitization.csv"
    df.to_csv(src, index=False)
    out = tmp_path / "out"
    rc = tables.main(
        ["--input", str(src), "--out-dir", str(out), "--tables", "sanitization_hfc"]
    )
    assert rc == 0
    assert (out / "sanitization_hfc.tex").read_text().startswith(tables.HEADER)
    rc = tables.main(
        ["--input", str(src), "--out-dir", str(out), "--tables", "sanitization_clip"]
    )
    assert rc == 1


# ------------------------------------------------------------------ analyze
def test_metric_rows(analyze, public_roots):
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(6, 8)).astype(np.float32)
    adv = clean + 0.5 * rng.normal(size=clean.shape).astype(np.float32)
    cc = clean + 0.2 * rng.normal(size=clean.shape).astype(np.float32)
    ca = adv.copy()
    ca[5] = 0  # a crop the codec did not produce
    arrays = {
        "aligned_112": clean,
        "aligned_112_adv_hfc_006": adv,
        "jpeg_112_512": cc,
        "jpeg_112_adv_hfc_006_512": ca,
    }
    d = config.embeddings_dir("colorferet", "m")
    d.mkdir(parents=True)
    for tag, arr in arrays.items():
        np.save(d / f"{tag}.npy", arr)
    rows = analyze.metric_rows(
        "colorferet", "m", ["hfc"], ["006"], ["jpeg", "webp"], [512]
    )
    assert [r["codec"] for r in rows] == ["jpeg", "webp"]
    got, missing = rows
    assert got["eps"] == 0.06 and got["n"] == 5
    unit = lambda e: e / np.linalg.norm(e, axis=1, keepdims=True)  # noqa: E731
    b = float((unit(clean[:5]) * unit(cc[:5])).sum(1).mean())
    c = float((unit(clean[:5]) * unit(ca[:5])).sum(1).mean())
    assert got["residual"] == pytest.approx(b - c, abs=1e-6)
    assert missing["n"] == 0 and np.isnan(missing["residual"])
    assert (
        analyze.metric_rows("colorferet", "other", ["hfc"], ["006"], ["jpeg"], [512])
        == []
    )


def test_analyze_cli_writes_full_grid(analyze, public_roots):
    out = public_roots / "s.csv"
    d = config.embeddings_dir("kk", "m")
    d.mkdir(parents=True)
    np.save(d / "aligned_112.npy", np.ones((3, 4), np.float32))
    rc = analyze.main(
        ["--datasets", "kk", "--models", "m", "--codecs", "jpeg", "--out", str(out)]
    )
    assert rc == 0
    df = pd.read_csv(out)
    assert list(df.columns[:6]) == list(analyze.KEY_COLUMNS)
    assert len(df) == 3 * 3 * 1 * 2  # attacks x eps x codecs x budgets
    assert set(df.eps) == {0.03, 0.06, 0.1}


# ------------------------------------------------------------------ craft
def _write_crops(root: Path, dataset: str, n: int) -> list[str]:
    rng = np.random.default_rng(1)
    rels = []
    for i in range(n):
        rel = f"{i // 2:05d}/{i // 2:05d}_000000_fa_{i}.png"
        p = config.aligned_dir(dataset, 112) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)).save(p)
        rels.append(rel)
    idx = pd.DataFrame(
        {"id": range(n), "rel_path": rels, "subject": [r[:5] for r in rels]}
    )
    config.index_csv(dataset).parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(config.index_csv(dataset), index=False)
    return rels


def test_load_clean_skips_missing(craft, public_roots):
    rels = _write_crops(public_roots, "colorferet", 4)
    (config.aligned_dir("colorferet", 112) / rels[1]).unlink()
    kept, imgs = craft.load_clean("colorferet")
    assert kept == [rels[0], rels[2], rels[3]]
    assert imgs.shape == (3, 112, 112, 3)
    kept, _ = craft.load_clean("colorferet", limit=2)
    assert kept == [rels[0]]


def test_craft_hfc_cli(craft, public_roots):
    pytest.importorskip("cv2")
    from face1kb import adversarial as adv

    rels = _write_crops(public_roots, "colorferet", 3)
    out = public_roots / "adv"
    assert (
        craft.main(
            [
                "--dataset",
                "colorferet",
                "--attack",
                "hfc",
                "--eps",
                "0.06",
                "--out-root",
                str(out),
            ]
        )
        == 0
    )
    dst = out / "aligned_112_adv_hfc_006"
    _, clean = craft.load_clean("colorferet")
    want = adv.hfc_attack(clean, 0.06)
    for i, rel in enumerate(rels):
        got = np.asarray(Image.open(dst / rel).convert("RGB"))
        assert np.array_equal(got, want[i])
    # existing files are kept without --overwrite
    stamp = (dst / rels[0]).stat().st_mtime_ns
    craft.main(
        [
            "--dataset",
            "colorferet",
            "--attack",
            "hfc",
            "--eps",
            "0.06",
            "--out-root",
            str(out),
        ]
    )
    assert (dst / rels[0]).stat().st_mtime_ns == stamp


def test_craft_default_destination(craft, public_roots):
    assert craft.out_dir("kk", "_adv_liae_003", None) == config.aligned_dir(
        "kk", 112, "_adv_liae_003"
    )
    assert craft.out_dir("kk", "_adv_liae_003", "/x").as_posix() == (
        "/x/aligned_112_adv_liae_003"
    )


def test_craft_refuses_legacy_store(craft, monkeypatch):
    monkeypatch.setattr(config, "LAYOUT", "legacy")
    with pytest.raises(SystemExit):
        craft.main(["--dataset", "colorferet", "--attack", "hfc"])
