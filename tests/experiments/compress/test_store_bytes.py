# SPDX-License-Identifier: MIT
"""compress.py re-creates stored bitstreams byte for byte (needs the data).

Runs with ``FACE1KB_DATA_ROOT`` pointing at the aligned crops and
``FACE1KB_WORK_ROOT`` at a tree of stored bitstreams (for example a copy of the
paper's data store with ``FACE1KB_LAYOUT=legacy``). The first index rows of a few
cells are encoded into a temporary work root and compared with the stored files;
cells without stored files are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from face1kb import config

from ._scripts import load

compress = load("compress")


def _reencode(tmp_path, monkeypatch, dataset, codecs, res, budget, rows=2):
    index = config.read_index(dataset).to_dict("records")[:rows]
    tasks = compress.build_tasks(dataset, index, codecs, [res], [budget])
    stored = [compress.output_path(t) for t in tasks]
    if not all(p.is_file() for p in stored):
        pytest.skip(f"no stored {codecs} files for {dataset} {res}/{budget}")
    monkeypatch.setattr(config, "WORK_ROOT", Path(tmp_path))
    opts = {"device": "cuda", "overwrite": False, "compressai_deterministic": "auto"}
    for t, ref in zip(tasks, stored):
        row = compress.process_task(t, opts)
        assert row["status"] == "encoded", row
        assert compress.output_path(t).read_bytes() == ref.read_bytes(), t


@pytest.mark.data
@pytest.mark.parametrize("dataset", config.DATASETS)
@pytest.mark.parametrize("res,budget", [(112, 1024), (224, 512)])
def test_classical_codecs(tmp_path, monkeypatch, dataset, res, budget):
    _reencode(
        tmp_path,
        monkeypatch,
        dataset,
        ["jpeg", "jpeg2000", "webp", "jpeg_xl", "avif", "heif", "jpeg_fzt"],
        res,
        budget,
    )


@pytest.mark.data
@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.parametrize("res,budget", [(112, 1024), (96, 512)])
def test_face1kb_codecs(tmp_path, monkeypatch, res, budget):
    _reencode(
        tmp_path, monkeypatch, "colorferet", ["ours_fast", "ours_accurate"], res, budget
    )
