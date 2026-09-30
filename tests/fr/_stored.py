# SPDX-License-Identifier: MIT
"""Embed the first aligned 112 px crops of both datasets and compare them with the
embedding arrays under ``FACE1KB_WORK_ROOT``.

Prints one JSON line ``{dataset: {...}}`` with, per dataset, the number of rows
compared, the maximum absolute difference, the number of bit-identical rows and the
minimum row cosine. Datasets without crops or without a stored array are reported
as ``null``::

    python tests/fr/_stored.py lvface_l [--n 64] [--batch 64] [--device cuda]
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Row-wise comparison of two ``(N, D)`` embedding arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = a / np.linalg.norm(a, axis=1, keepdims=True)
    nb = b / np.linalg.norm(b, axis=1, keepdims=True)
    return {
        "rows": int(len(a)),
        "max_abs_diff": float(np.abs(a - b).max()),
        "exact_rows": int((a == b).all(axis=1).sum()),
        "min_cos": float((na * nb).sum(axis=1).min()),
    }


def main() -> None:
    """Command-line entry point."""
    from PIL import Image

    from face1kb import config, fr

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model = fr.load(args.model, device=args.device, download=False)
    out: dict = {}
    for ds in config.DATASETS:
        ref_path = config.embeddings_path(ds, args.model, "aligned_112")
        if not ref_path.is_file() or not config.index_csv(ds).is_file():
            out[ds] = None
            continue
        idx = config.read_index(ds).sort_values("id").reset_index(drop=True)
        rel = list(idx.rel_path[: args.n])
        crops = [
            np.asarray(
                Image.open(config.aligned_dir(ds, 112) / p).convert("RGB"), np.uint8
            )
            for p in rel
        ]
        emb = np.concatenate(
            [
                model.embed(crops[i : i + args.batch])
                for i in range(0, len(crops), args.batch)
            ]
        )
        ref = np.array(np.load(ref_path, mmap_mode="r")[: len(rel)])
        out[ds] = compare(emb, ref)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
