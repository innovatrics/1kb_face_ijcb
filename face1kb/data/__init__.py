# SPDX-License-Identifier: MIT
"""Data layer: index, verification pairs, labels, crop geometry and preprocessing.

The public pipeline starts from aligned crops (``aligned_<res>/<subject>/<stem>.png``
under :func:`face1kb.config.dataset_dir`). Modules:

* :mod:`face1kb.data.index` -- canonical image index (``index.csv``) from a crop scan;
* :mod:`face1kb.data.pairs` -- complete verification-pair matrix (``pairs.parquet``);
* :mod:`face1kb.data.colorferet_labels` -- Color FERET labels from the NIST ground
  truth (``labels.csv``);
* :mod:`face1kb.data.attributes` -- estimated AI-Solutions-KK attributes
  (``attributes.csv``);
* :mod:`face1kb.data.alignment` -- ArcFace template, resolution scaling,
  crop-tightness variants;
* :mod:`face1kb.data.crops` -- derived crop folders (variants, other resolutions);
* :mod:`face1kb.data.preprocess` -- preprocessing operators A1-A4, B1, B2, C1, C2;
* :mod:`face1kb.data.stats` -- dataset statistics, figures, Color FERET table;
* :mod:`face1kb.data.release` -- export of the on-request crop release to folders;
* :mod:`face1kb.data.fetch` -- download/checksum helpers of the third-party models.

Heavy dependencies (OpenCV, torch, insightface, MediaPipe, ``stone``) are imported
lazily inside the functions that need them.
"""

from face1kb.data.alignment import (
    ARCFACE_DST_112,
    CROP_VARIANTS,
    derive_resolution,
    render_variant,
    template,
    variant_template,
)
from face1kb.data.index import (
    CF_POSE_NAMES,
    align_to_index,
    build_index,
    cf_pose,
    cf_pose_code,
    crop_rel_path,
    read_index_csv,
    scan_crops,
    write_index,
)
from face1kb.data.pairs import build_pairs, pair_counts, read_pairs, write_pairs

__all__ = [
    "ARCFACE_DST_112",
    "CF_POSE_NAMES",
    "CROP_VARIANTS",
    "align_to_index",
    "build_index",
    "build_pairs",
    "cf_pose",
    "cf_pose_code",
    "crop_rel_path",
    "derive_resolution",
    "pair_counts",
    "read_index_csv",
    "read_pairs",
    "render_variant",
    "scan_crops",
    "template",
    "variant_template",
    "write_index",
    "write_pairs",
]
