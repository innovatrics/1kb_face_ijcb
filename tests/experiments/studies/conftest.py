# SPDX-License-Identifier: MIT
"""Fixtures of the study tests: temporary face1kb roots and a synthetic dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face1kb import config


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """Point every face1kb root at a temporary folder (public layout)."""
    for attr, name in (
        ("DATA_ROOT", "data"),
        ("WORK_ROOT", "work"),
        ("OUTPUT_ROOT", "outputs"),
    ):
        monkeypatch.setattr(config, attr, tmp_path / name)
    monkeypatch.setattr(config, "LAYOUT", "public")
    return tmp_path


def smooth_image(rng, res: int, subject: int) -> np.ndarray:
    """A smooth colour image whose low frequencies depend on ``subject``."""
    yy, xx = np.mgrid[0:res, 0:res] / res
    base = np.stack(
        [
            np.sin(2 * np.pi * (xx * (1 + subject % 3) + yy)),
            np.cos(2 * np.pi * (yy * (1 + subject % 4) - xx)),
            np.sin(2 * np.pi * (xx + yy) * (1 + subject % 2)),
        ],
        axis=-1,
    )
    img = 128 + 60 * base + rng.normal(0, 6, size=base.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_dataset(
    dataset: str,
    n_subjects: int = 6,
    per_subject: int = 4,
    resolutions=(112,),
    seed: int = 0,
) -> pd.DataFrame:
    """Write aligned crops, ``index.csv`` and ``pairs.parquet`` of a toy dataset."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_subjects + 1):
        for k in range(per_subject):
            rel = f"{s:05d}/{s:05d}_{k:02d}.png"
            for res in resolutions:
                path = config.aligned_dir(dataset, res) / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(smooth_image(rng, res, s)).save(path)
            rows.append({"rel_path": rel, "subject": f"{s:05d}"})
    index = pd.DataFrame(rows)
    index.insert(0, "id", np.arange(len(index)))
    index.to_csv(config.index_csv(dataset), index=False)
    subj = index.subject.to_numpy()
    i1, i2 = np.triu_indices(len(index), 1)
    pd.DataFrame(
        {
            "idx1": i1.astype(np.int32),
            "idx2": i2.astype(np.int32),
            "label": (subj[i1] == subj[i2]).astype(np.int8),
        }
    ).to_parquet(config.pairs_parquet(dataset), index=False)
    return index


class ColourEmbedder:
    """Deterministic toy matcher: a coarse colour/structure descriptor."""

    dim = 512
    name = "toy_matcher"

    def embed(self, crops):
        out = []
        for c in crops:
            c = np.asarray(c, np.float32) / 255.0
            small = c.reshape(8, 14, 8, 14, 3).mean(axis=(1, 3)).ravel()  # 192
            v = np.concatenate([small, np.zeros(self.dim - small.size, np.float32)])
            out.append(v - 0.5 * (v != 0))
        return np.stack(out).astype(np.float32)


@pytest.fixture
def toy_matcher():
    """Register :class:`ColourEmbedder` as ``toy_matcher`` for the test."""
    from face1kb import fr

    fr.register_embedder(
        "toy_matcher", lambda device=None: ColourEmbedder(), overwrite=True
    )
    yield "toy_matcher"
    fr.unregister("toy_matcher")
