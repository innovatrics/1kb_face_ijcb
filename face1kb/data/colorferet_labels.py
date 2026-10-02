# SPDX-License-Identifier: MIT
"""Color FERET labels from the NIST ground truth.

The NIST Color FERET distribution ships per-subject and per-recording ground truth
for its 11,338 colour images of 994 subjects in two equivalent formats, on both
DVDs::

    dvd{1,2}/data/ground_truths/xml/<subject>/<subject>.xml        # Gender, YOB, Race
    dvd{1,2}/data/ground_truths/xml/<subject>/<image>.xml          # CaptureDate, Pose,
                                                                   # glasses, beard, ...
    dvd{1,2}/data/ground_truths/name_value/<subject>/<subject>.txt # same, key=value
    dvd{1,2}/data/ground_truths/name_value/<subject>/<image>.txt

:func:`parse_ground_truth` reads either format from an extracted copy of the
distribution (a ``dvd1``/``dvd2`` folder, or a folder with ``dvd1``/``dvd2`` at most
:data:`DVD_SEARCH_DEPTH` levels below it) or directly from the distribution tar
archive, and returns one row per recording with the columns

========================  ===========================================================
``subject``               NIST subject number, zero-padded (``00001``)
``image``                 image stem (``00001_930831_fa_a``)
``rel_path``              ``<subject>/<image>.png``, the crop path of the index
``pose_code``             NIST pose code (``fa``, ``fb``, ``hl``, ..., ``re``)
``pose``                  pose description (:data:`~face1kb.data.index.CF_POSE_NAMES`)
``gender``                ``male`` / ``female``
``race``                  race with the 7-class merge (:data:`RACE_MERGE`)
``race_nist``             race as given by NIST (9 classes)
``glasses``               ``yes`` / ``no``
``beard``, ``mustache``   ``1`` = no, ``2`` = yes
``age_from``, ``age_to``  age in years at capture: capture year minus year of birth
``year_of_birth``         NIST year of birth
``capture_date``          NIST capture date (``MM/DD/YYYY``)
``yaw``, ``pitch``,       NIST nominal pose angles (degrees; pitch and roll are 0
``roll``                  throughout the colour set)
========================  ===========================================================

Rows are sorted by ``(subject, image)``, which is the order the paper's attribute
table was built from. Landmark coordinates in the ground truth are not read.

The value encodings (lower-case gender and glasses, ``1``/``2`` facial hair, the
race merge) are those of the labels used in the paper's fairness and difficulty
analyses.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from face1kb.data.index import CF_POSE_NAMES

log = logging.getLogger(__name__)

#: NIST race categories merged into the 7 classes of the paper.
RACE_MERGE: dict[str, str] = {
    "Black-or-African-American": "Black",
    "Asian-Middle-Eastern": "Asian",
    "Asian-Southern": "Asian",
}
#: Encoding of the NIST beard / mustache flags.
HAIR_CODES: dict[str, int] = {"No": 1, "Yes": 2}
#: Column order of the labels table.
COLUMNS: tuple[str, ...] = (
    "subject",
    "image",
    "rel_path",
    "pose_code",
    "pose",
    "gender",
    "race",
    "race_nist",
    "glasses",
    "beard",
    "mustache",
    "age_from",
    "age_to",
    "year_of_birth",
    "capture_date",
    "yaw",
    "pitch",
    "roll",
)
#: Number of colour recordings in the NIST Color FERET release.
N_RECORDINGS = 11_338
#: Number of subjects in the NIST Color FERET release.
N_SUBJECTS = 994
#: How many folder levels below the given folder are searched for ``dvd1``/``dvd2``.
DVD_SEARCH_DEPTH = 3

_GT = re.compile(
    r"(?:^|/)dvd[12]/data/ground_truths/(xml|name_value)/(\d{5})/(\d{5}(?:_[^/]*)?)"
    r"\.(xml|txt)$"
)


# --------------------------------------------------------------------- sources
def _find_dvds(root: Path, max_depth: int = DVD_SEARCH_DEPTH) -> list[Path]:
    """``dvd1`` / ``dvd2`` folders at ``root`` or up to ``max_depth`` levels below."""
    if root.name in ("dvd1", "dvd2"):
        return [root]
    found: list[Path] = []
    level = [root]
    for _ in range(max_depth):
        nxt = []
        for d in level:
            try:
                children = sorted(p for p in d.iterdir() if p.is_dir())
            except OSError:
                continue
            for c in children:
                (found if c.name in ("dvd1", "dvd2") else nxt).append(c)
        if found:
            break
        level = nxt
    return found


def _iter_dir(root: Path, fmt: str) -> Iterator[tuple[str, str, bytes]]:
    for dvd in _find_dvds(root):
        base = dvd / "data" / "ground_truths" / fmt
        if not base.is_dir():
            continue
        for path in sorted(base.glob("[0-9]" * 5 + "/*")):
            m = _GT.search(path.as_posix())
            if m and m.group(1) == fmt:
                yield m.group(2), m.group(3), path.read_bytes()


def _iter_tar(path: Path, fmt: str) -> Iterator[tuple[str, str, bytes]]:
    with tarfile.open(path, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            m = _GT.search(member.name)
            if not m or m.group(1) != fmt:
                continue
            f = tar.extractfile(member)
            if f is not None:
                yield m.group(2), m.group(3), f.read()


def _iter_ground_truth(source: Path, fmt: str) -> Iterator[tuple[str, str, bytes]]:
    if source.is_dir():
        return _iter_dir(source, fmt)
    if source.is_file():
        return _iter_tar(source, fmt)
    raise FileNotFoundError(f"Color FERET ground truth not found: {source}")


# --------------------------------------------------------------------- parsers
def _parse_subject_xml(data: bytes) -> dict:
    root = ET.parse(io.BytesIO(data)).getroot()
    subj = root.find("Subject") if root.tag == "Subjects" else root
    return {
        "gender": subj.find("Gender").get("value"),
        "year_of_birth": subj.find("YOB").get("value"),
        "race_nist": subj.find("Race").get("value"),
    }


def _parse_recording_xml(data: bytes) -> dict:
    root = ET.parse(io.BytesIO(data)).getroot()
    rec = root.find("Recording") if root.tag == "Recordings" else root
    face = rec.find("Subject/Application/Face")
    pose = face.find("Pose")
    hair = face.find("Hair")
    return {
        "capture_date": rec.findtext("CaptureDate"),
        "pose_code": pose.get("name"),
        "yaw": pose.get("yaw"),
        "pitch": pose.get("pitch"),
        "roll": pose.get("roll"),
        "glasses": face.find("Wearing").get("glasses"),
        "beard": hair.get("beard"),
        "mustache": hair.get("mustache"),
    }


def _parse_name_value(data: bytes) -> dict[str, str]:
    out = {}
    for line in data.decode("utf-8", errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _subject_nv(data: bytes) -> dict:
    kv = _parse_name_value(data)
    return {
        "gender": kv["gender"],
        "year_of_birth": kv["yob"],
        "race_nist": kv["race"],
    }


def _recording_nv(data: bytes) -> dict:
    kv = _parse_name_value(data)
    return {
        "capture_date": kv["capture_date"],
        "pose_code": kv["pose"],
        "yaw": kv["yaw"],
        "pitch": kv["pitch"],
        "roll": kv["roll"],
        "glasses": kv["glasses"],
        "beard": kv["beard"],
        "mustache": kv["mustache"],
    }


_PARSERS = {
    "xml": (_parse_subject_xml, _parse_recording_xml),
    "name_value": (_subject_nv, _recording_nv),
}


def _age(year_of_birth: str, capture_date: str) -> int | None:
    try:
        return int(str(capture_date).split("/")[-1]) - int(year_of_birth)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------- public
def read_records(
    source: str | os.PathLike, fmt: str = "xml"
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Read the raw subject and recording records of one ground-truth format.

    Returns
    -------
    tuple[dict, dict]
        ``{subject: fields}`` and ``{image: fields | {"subject": subject}}``.
    """
    if fmt not in _PARSERS:
        raise ValueError(f"fmt must be one of {list(_PARSERS)}")
    parse_subject, parse_recording = _PARSERS[fmt]
    subjects: dict[str, dict] = {}
    records: dict[str, dict] = {}
    for subject, stem, data in _iter_ground_truth(Path(source), fmt):
        if stem == subject:
            subjects[subject] = parse_subject(data)
        else:
            rec = parse_recording(data)
            rec["subject"] = subject
            if stem in records:
                raise ValueError(f"duplicate ground truth for image {stem}")
            records[stem] = rec
    return subjects, records


def labels_from_records(
    subjects: dict[str, dict], records: dict[str, dict]
) -> pd.DataFrame:
    """Normalise raw NIST records into the labels table (see module docstring)."""
    rows = []
    for image, rec in records.items():
        subject = rec["subject"]
        if subject not in subjects:
            raise KeyError(f"no subject record for {subject} (image {image})")
        subj = subjects[subject]
        code = rec["pose_code"]
        if code not in CF_POSE_NAMES:
            raise ValueError(f"unknown pose code {code!r} for {image}")
        age = _age(subj["year_of_birth"], rec["capture_date"])
        rows.append(
            {
                "subject": subject,
                "image": image,
                "rel_path": f"{subject}/{image}.png",
                "pose_code": code,
                "pose": CF_POSE_NAMES[code],
                "gender": str(subj["gender"]).lower(),
                "race": RACE_MERGE.get(subj["race_nist"], subj["race_nist"]),
                "race_nist": subj["race_nist"],
                "glasses": str(rec["glasses"]).lower(),
                "beard": HAIR_CODES[rec["beard"]],
                "mustache": HAIR_CODES[rec["mustache"]],
                "age_from": age,
                "age_to": age,
                "year_of_birth": int(subj["year_of_birth"]),
                "capture_date": rec["capture_date"],
                "yaw": float(rec["yaw"]),
                "pitch": float(rec["pitch"]),
                "roll": float(rec["roll"]),
            }
        )
    if not rows:
        raise ValueError("no Color FERET recordings found")
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    df = df.sort_values(["subject", "image"], kind="stable").reset_index(drop=True)
    for col in ("age_from", "age_to"):
        df[col] = df[col].astype("Int64")
    return df


def parse_ground_truth(source: str | os.PathLike, fmt: str = "auto") -> pd.DataFrame:
    """Build the Color FERET labels table from the NIST ground truth.

    Parameters
    ----------
    source : path
        Extracted NIST distribution (a ``dvd1`` / ``dvd2`` folder, or a folder with
        ``dvd1`` / ``dvd2`` at most :data:`DVD_SEARCH_DEPTH` levels below it) or the
        distribution tar archive (searched at any depth).
    fmt : {"auto", "xml", "name_value"}
        Ground-truth format to read; ``auto`` uses the XML files and falls back to
        the name/value text files when no XML is found.

    Returns
    -------
    DataFrame
        One row per recording (11,338 for the complete release), columns
        :data:`COLUMNS`, sorted by ``(subject, image)``.
    """
    formats = ["xml", "name_value"] if fmt == "auto" else [fmt]
    for f in formats:
        subjects, records = read_records(source, f)
        if records:
            log.info(
                "read %d subjects / %d recordings (%s)", len(subjects), len(records), f
            )
            df = labels_from_records(subjects, records)
            if len(df) != N_RECORDINGS or df.subject.nunique() != N_SUBJECTS:
                log.warning(
                    "expected %d recordings of %d subjects, found %d of %d",
                    N_RECORDINGS,
                    N_SUBJECTS,
                    len(df),
                    df.subject.nunique(),
                )
            return df
    where = (
        f"under {source}"
        if not Path(source).is_dir()
        else f"in dvd1/dvd2 folders at most {DVD_SEARCH_DEPTH} levels below {source}"
    )
    raise FileNotFoundError(
        f"no Color FERET ground truth ({' or '.join(formats)}) {where}"
    )


def write_labels(df: pd.DataFrame, path: str | os.PathLike) -> Path:
    """Write the labels table as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def read_labels(path: str | os.PathLike) -> pd.DataFrame:
    """Read a Color FERET labels CSV, keeping ``subject`` as a zero-padded string.

    Works for tables written by :func:`write_labels` and for older tables with
    the same columns; only the columns of :data:`COLUMNS` that are present, plus
    any other columns, are returned unchanged.
    """
    df = pd.read_csv(path, dtype={"subject": str, "image": str, "rel_path": str})
    if "subject" in df.columns:
        df["subject"] = df["subject"].str.zfill(5)
    return df
