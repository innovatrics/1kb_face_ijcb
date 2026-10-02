# SPDX-License-Identifier: MIT
"""Export of an aligned-crop release in Hugging Face ``datasets`` format."""

from __future__ import annotations

import numpy as np
import pytest

from face1kb.data.release import export_release

datasets = pytest.importorskip("datasets")
PIL = pytest.importorskip("PIL.Image")


def _release(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for res in (112, 224):
        for name in ("pins_A/A0_0.png", "pins_A/A1_1.png", "pins_B/B0_7.png"):
            img = rng.integers(0, 256, (res, res, 3), dtype=np.uint8)
            rows.append(
                {
                    "image": PIL.fromarray(img),
                    "identity": name.split("/")[0].removeprefix("pins_"),
                    "file_name": name,
                    "resolution": res,
                }
            )
    features = datasets.Features(
        {
            "image": datasets.Image(),
            "identity": datasets.Value("string"),
            "file_name": datasets.Value("string"),
            "resolution": datasets.Value("int32"),
        }
    )
    ds = datasets.Dataset.from_list(rows, features=features)
    path = tmp_path / "release"
    ds.save_to_disk(str(path))
    return path, rows


def test_export_release(tmp_path):
    src, rows = _release(tmp_path)
    out = tmp_path / "kk"
    counts = export_release(src, lambda r: out / f"aligned_{r}")
    assert counts == {112: 3, 224: 3}
    for row in rows:
        p = out / f"aligned_{row['resolution']}" / row["file_name"]
        got = np.asarray(PIL.open(p).convert("RGB"))
        assert np.array_equal(got, np.asarray(row["image"]))
    only = export_release(src, lambda r: tmp_path / "x" / f"aligned_{r}", [224])
    assert only == {224: 3} and not (tmp_path / "x" / "aligned_112").exists()
