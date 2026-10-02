# SPDX-License-Identifier: MIT
"""Estimated demographic attributes of AI-Solutions-KK.

AI-Solutions-KK carries no demographic ground truth, so the fairness analysis uses
model estimates, computed per image and summarised per identity downstream:

* **age and gender** -- the insightface ``buffalo_l`` ``genderage`` model, run on
  the aligned 112 px crop with the ArcFace template as its five key points (the
  crop is already aligned, so no detector runs). The first ``per_subject_ga`` crops
  of each identity (index order) are used, 8 by default. The model runs on the CPU
  by default, which reproduces the paper's genderage outputs bit for bit given the
  same crops (the paper's age and gender estimates were made on another set of
  112 px crops than the released ones; see the known deviations in the dataset
  documentation); the CUDA execution provider differs in the last float bits, which
  can move an age that sits on a rounding boundary by one year.
* **Monk Skin Tone** (MST1-MST10) -- the ``stone`` package
  (``skin-tone-classifier``) with the 10-tone Monk palette on the aligned 224 px
  crop of the first ``per_subject_mst`` crop(s) of each identity, 1 by default
  (skin tone is identity-stable and ``stone`` is slow).

``stone`` is GPL-3.0 licensed; it is an optional dependency (``pip install
skin-tone-classifier``) that is imported only when MST is requested and is not
distributed with this package. The insightface pretrained models are for
non-commercial research use only; they are downloaded from the official model zoo.

``stone`` clusters skin colours with ``cv2.kmeans`` and random initial centres drawn
from OpenCV's global random generator, so an MST estimate depends on the sequence
of ``stone`` calls in the process. :func:`estimate_attributes` calls it once per
identity in identity order, as the paper's attribute table was built; run it over
the full dataset to reproduce that sequence.

The output table (``attributes.csv``) has one row per processed image::

    subject, image, rel_path, age, gender, mst_label, mst_index, skin_hex

``image`` is the crop file name (``<stem>.png``); ``gender`` is ``M`` / ``F``;
``mst_*`` and ``skin_hex`` are empty for the images without an MST estimate.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from face1kb.data.alignment import ARCFACE_DST_112

log = logging.getLogger(__name__)

#: Monk Skin Tone palette (MST1 lightest ... MST10 darkest).
MONK_PALETTE: tuple[str, ...] = (
    "#f6ede4",
    "#f3e7db",
    "#f7ead0",
    "#eadaba",
    "#d7bd96",
    "#a07e56",
    "#825c43",
    "#604134",
    "#3a312a",
    "#292420",
)
#: Monk Skin Tone labels.
MONK_LABELS: tuple[str, ...] = tuple(f"MST{i}" for i in range(1, 11))
#: Default number of crops per identity for age/gender and for MST.
GA_PER_SUBJECT = 8
MST_PER_SUBJECT = 1
#: Output columns.
COLUMNS: tuple[str, ...] = (
    "subject",
    "image",
    "rel_path",
    "age",
    "gender",
    "mst_label",
    "mst_index",
    "skin_hex",
)


class GenderAge:
    """insightface ``genderage`` on aligned 112 px crops (BGR uint8).

    Parameters
    ----------
    model_path : path, optional
        ``genderage.onnx``; fetched from the insightface model zoo when omitted.
    ctx_id : int
        ``-1`` (default): CPU, which reproduces the paper's genderage outputs bit
        for bit given the same crops; ``n >= 0``: CUDA device ``n`` among the
        visible devices (``CUDA_VISIBLE_DEVICES``), with the CPU as fallback.
    providers : sequence, optional
        onnxruntime execution providers; overrides the choice made by ``ctx_id``.
    """

    def __init__(self, model_path=None, ctx_id: int = -1, providers=None):
        from insightface.model_zoo import get_model  # noqa: PLC0415

        if model_path is None:
            from face1kb.data.fetch import genderage_path  # noqa: PLC0415

            model_path = genderage_path()
        options = None
        if providers is None:
            if ctx_id < 0:
                providers = ["CPUExecutionProvider"]
            else:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                options = [{"device_id": str(int(ctx_id))}, {}]
        self.model = get_model(
            str(model_path), providers=list(providers), provider_options=options
        )
        self.model.prepare(ctx_id=ctx_id)
        self.kps = ARCFACE_DST_112.astype(np.float32)

    def __call__(self, bgr: np.ndarray) -> tuple[int, str]:
        """Return ``(age, "M" | "F")`` for one aligned crop."""
        from insightface.app.common import Face  # noqa: PLC0415

        h, w = bgr.shape[:2]
        face = Face(
            bbox=np.array([0, 0, w - 1, h - 1], np.float32),
            kps=self.kps,
            det_score=1.0,
        )
        self.model.get(bgr, face)
        return int(face.age), ("M" if int(face.gender) == 1 else "F")


def monk_skin_tone(path: str | os.PathLike) -> tuple[str, int | str, str]:
    """Monk Skin Tone of the face in an image file with ``stone``.

    Returns
    -------
    tuple
        ``(label, index, skin_hex)``, e.g. ``("MST6", 6, "#A07E56")``; empty
        strings when no face or tone is found.
    """
    import stone  # noqa: PLC0415

    r = stone.process(
        str(path),
        image_type="color",
        tone_palette=list(MONK_PALETTE),
        tone_labels=list(MONK_LABELS),
        return_report_image=False,
    )
    fc = (r.get("faces") or [{}])[0]
    label = fc.get("tone_label", "")
    index = int(label[3:]) if str(label).startswith("MST") else ""
    return label, index, fc.get("skin_tone", "")


def estimate_attributes(
    index: pd.DataFrame,
    dir_112: str | os.PathLike,
    dir_224: str | os.PathLike | None,
    per_subject_ga: int = GA_PER_SUBJECT,
    per_subject_mst: int = MST_PER_SUBJECT,
    subjects: Sequence[str] | None = None,
    genderage: Callable[[np.ndarray], tuple[int, str]] | None = None,
    ctx_id: int = -1,
    strict: bool = True,
) -> pd.DataFrame:
    """Estimate age/gender (112 px) and MST (224 px) per identity.

    Parameters
    ----------
    index : DataFrame
        Dataset index (``rel_path``, ``subject``), canonical order.
    dir_112, dir_224 : path
        Aligned 112 / 224 px crop folders; ``dir_224=None`` or
        ``per_subject_mst=0`` skips MST.
    per_subject_ga : int
        Crops per identity for age and gender (the first ones in index order).
    per_subject_mst : int
        Crops per identity (among those) for MST.
    subjects : sequence of str, optional
        Restrict to these identities (default: all, in sorted order).
    genderage : callable, optional
        ``bgr_crop -> (age, "M" | "F")``, e.g. a loaded :class:`GenderAge`;
        created with ``ctx_id`` when omitted.
    ctx_id : int
        :class:`GenderAge` device when creating the model (``-1``: CPU).
    strict : bool
        Raise :class:`face1kb.data.crops.MissingCropsError` if a 112 px crop cannot
        be read (after processing all others); otherwise skip it with a warning.
        A crop on which genderage fails keeps its row with empty age and gender.

    Returns
    -------
    DataFrame
        Columns :data:`COLUMNS`, one row per processed crop.

    Raises
    ------
    ValueError
        If ``subjects`` names an identity that is not in the index, or if no row
        was produced.
    """
    import cv2  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    from face1kb.data.crops import check_missing  # noqa: PLC0415

    warnings.filterwarnings("ignore", module="stone")
    dir_112 = Path(dir_112)
    use_mst = dir_224 is not None and per_subject_mst > 0
    groups = {
        str(s): g["rel_path"].tolist()
        for s, g in index.groupby(index["subject"].astype(str), sort=True)
    }
    order = sorted(groups) if subjects is None else [str(s) for s in subjects]
    unknown = [s for s in order if s not in groups]
    if unknown:
        raise ValueError(
            f"{len(unknown)} identities not in the index, e.g. {unknown[0]!r}"
        )
    ga = genderage or GenderAge(ctx_id=ctx_id)
    rows = []
    missing: list[str] = []
    n_crops = 0
    for subject in tqdm(order, desc="attributes"):
        for j, rel in enumerate(groups[subject][:per_subject_ga]):
            n_crops += 1
            img = cv2.imread(str(dir_112 / rel))
            if img is None:
                log.warning("cannot read %s", dir_112 / rel)
                missing.append(str(dir_112 / rel))
                continue
            try:
                age, gender = ga(img)
            except Exception as exc:  # noqa: BLE001 - keep the row, as the paper did
                log.warning("genderage failed on %s: %s", rel, exc)
                age, gender = "", ""
            mst = ("", "", "")
            if use_mst and j < per_subject_mst:
                try:
                    mst = monk_skin_tone(Path(dir_224) / rel)
                except Exception as exc:  # noqa: BLE001
                    log.warning("stone failed on %s: %s", rel, exc)
            rows.append((subject, Path(rel).name, rel, age, gender, *mst))
    if strict:
        check_missing(missing, n_crops)
    if not rows:
        raise ValueError(f"no attributes estimated: no crop of {dir_112} was read")
    return pd.DataFrame(rows, columns=list(COLUMNS))


def write_attributes(df: pd.DataFrame, path: str | os.PathLike) -> Path:
    """Write the attribute table as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def read_attributes(path: str | os.PathLike) -> pd.DataFrame:
    """Read an attribute CSV with the public column names.

    Accepts tables whose identity column is named ``identity`` (renamed to
    ``subject``) and adds ``rel_path`` (``<subject>/<image>``) when missing.
    """
    df = pd.read_csv(path, dtype={"subject": str, "identity": str, "image": str})
    if "subject" not in df.columns and "identity" in df.columns:
        df = df.rename(columns={"identity": "subject"})
    if "rel_path" not in df.columns:
        df.insert(2, "rel_path", df["subject"] + "/" + df["image"])
    return df
