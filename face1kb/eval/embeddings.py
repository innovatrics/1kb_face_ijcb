# SPDX-License-Identifier: MIT
"""Embedding arrays: source tags, model selection, loading and the manifest.

An embedding array is stored per ``(dataset, model, source)`` as
``embeddings/<model>/<tag>.npy`` (see :func:`face1kb.config.embeddings_path`): shape
``(N, 512)`` float32, row ``i`` belongs to image id ``i`` of the dataset index, and a
row of NaN marks a crop that could not be embedded. Embeddings are stored raw; the
scoring code L2-normalises them.

Source tags follow the grammar ``aligned_<res><suffix>`` (the uncompressed aligned
crops) or ``<codec>_<res><suffix>_<budget>`` (a compressed cell), e.g.
``aligned_112``, ``webp_112_1024``, ``ours_accurate_112_adv_hfc_003_512``. The
suffix selects a crop variant (``_tight``/``_mid``/``_fill``, ``_A1`` .. ``_C2``,
``_adv_<attack>_<eps>``); the core grid has an empty suffix.

``embeddings/manifest.csv`` lists every array with its row counts
(:data:`MANIFEST_COLUMNS`); :func:`scan_manifest` rebuilds it from the arrays on
disk.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from face1kb import config

log = logging.getLogger(__name__)

#: Width of the stored embeddings; arrays of another width are ignored by the scorers.
EMB_DIM = 512

#: Compressed-source codec names recognised in source tags (result-file names).
CODECS: tuple[str, ...] = (
    "jpeg",
    "jpeg2000",
    "webp",
    "jpeg_xl",
    "avif",
    "heif",
    "jpeg_fzt",
    "jpeg_ai",
    "neural_bmshj2018",
    "neural_mbt2018_mean",
    "ours_fast",
    "ours_accurate",
)

#: The four anchor matchers of the study (bootstrap CIs, significance, fairness).
ANCHOR_MODELS: tuple[str, ...] = (
    "arcface_antelopev2",
    "lvface_l",
    "topofr_r100",
    "edgeface_xs",
)

#: Columns of ``embeddings/manifest.csv``.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "dataset",
    "model",
    "source_tag",
    "kind",
    "res",
    "codec",
    "budget",
    "n_images",
    "n_ok",
    "n_missing",
    "dim",
)


def parse_tag(
    tag: str, *, aligned_codec: str = "aligned", codecs: Sequence[str] = CODECS
) -> dict | None:
    """Split a source tag into its fields.

    Parameters
    ----------
    tag : str
        ``aligned_<res><suffix>`` or ``<codec>_<res><suffix>_<budget>``.
    aligned_codec : str
        Value of ``codec`` returned for aligned sources (``""`` in ``metrics.csv``,
        ``"aligned"`` in the significance and fairness tables).
    codecs : sequence of str
        Codec names to recognise. The longest matching name wins, so ``jpeg_xl``
        is not read as ``jpeg`` with suffix ``_xl``.

    Returns
    -------
    dict or None
        ``{"kind", "res", "suffix", "codec", "budget"}`` (``kind`` is ``"aligned"``
        or ``"compressed"``; ``budget`` is 0 for aligned sources), or ``None`` when
        the tag names no known codec.
    """
    if tag.startswith("aligned_"):
        m = re.match(r"(\d+)(.*)$", tag[len("aligned_") :])
        return {
            "kind": "aligned",
            "res": int(m.group(1)),
            "suffix": m.group(2),
            "codec": aligned_codec,
            "budget": 0,
        }
    for codec in sorted(codecs, key=len, reverse=True):
        if tag.startswith(codec + "_"):
            res_suffix, budget = tag[len(codec) + 1 :].rsplit("_", 1)
            m = re.match(r"(\d+)(.*)$", res_suffix)
            return {
                "kind": "compressed",
                "res": int(m.group(1)),
                "suffix": m.group(2),
                "codec": codec,
                "budget": int(budget),
            }
    return None


def models_present(dataset: str) -> list[str]:
    """Sorted names of the models that have an embeddings folder for ``dataset``."""
    # the embeddings root is the folder of the manifest (the layout lives in config)
    edir = config.embeddings_manifest(dataset).parent
    if not edir.is_dir():
        return []
    return sorted(p.name for p in edir.glob("*") if p.is_dir())


def resolve_models(spec: str | Iterable[str], dataset: str) -> list[str]:
    """Select models by ``"anchor"``, ``"all"`` or a comma list, keeping those on disk.

    Parameters
    ----------
    spec : str or iterable of str
        ``"anchor"`` (:data:`ANCHOR_MODELS`), ``"all"`` (every model folder present),
        a comma-separated list or an iterable of model names.
    dataset : str
        Dataset whose embeddings folder is inspected.

    Returns
    -------
    list of str
        The requested models that have embeddings for ``dataset``, in request
        order (``"all"``: sorted).
    """
    present = models_present(dataset)
    if isinstance(spec, str):
        if spec == "anchor":
            want = list(ANCHOR_MODELS)
        elif spec == "all":
            want = present
        else:
            want = [m.strip() for m in spec.split(",") if m.strip()]
    else:
        want = list(spec)
    return [m for m in want if m in present]


def list_sources(
    dataset: str, model: str, *, aligned_codec: str = "aligned"
) -> list[tuple[Path, dict]]:
    """Embedding arrays of one model, sorted by file name, with their parsed tags.

    Files whose name is not a recognised source tag (:func:`parse_tag`) are left
    out.
    """
    edir = config.embeddings_dir(dataset, model)
    if not edir.is_dir():
        return []
    out = []
    for path in sorted(edir.glob("*.npy")):
        meta = parse_tag(path.stem, aligned_codec=aligned_codec)
        if meta:
            out.append((path, meta))
    return out


def load_embeddings(
    dataset: str, model: str, tag: str, *, mmap: bool = False
) -> np.ndarray | None:
    """Load one embedding array, or ``None`` when it does not exist."""
    path = config.embeddings_path(dataset, model, tag)
    if not path.is_file():
        return None
    return np.load(path, mmap_mode="r" if mmap else None)


def check_rows(emb: np.ndarray, n_index: int, what: str = "embeddings") -> None:
    """Raise ``ValueError`` unless ``emb`` has one row per index entry."""
    if emb.shape[0] != n_index:
        raise ValueError(f"{what}: {emb.shape[0]} rows != index length {n_index}")


# ----------------------------------------------------------------------- manifest
def _manifest_meta(tag: str) -> dict:
    meta = parse_tag(tag, aligned_codec="")
    if meta is None:
        return {"kind": "?", "res": 0, "codec": "", "budget": 0}
    return {k: meta[k] for k in ("kind", "res", "codec", "budget")}


def manifest_row(dataset: str, model: str, path: Path) -> dict:
    """Manifest row of one embedding array (reads it memory-mapped)."""
    arr = np.load(path, mmap_mode="r")
    n, dim = int(arr.shape[0]), int(arr.shape[1])
    n_miss = int(np.isnan(arr).any(axis=1).sum())
    return {
        "dataset": dataset,
        "model": model,
        "source_tag": path.stem,
        **_manifest_meta(path.stem),
        "n_images": n,
        "n_ok": n - n_miss,
        "n_missing": n_miss,
        "dim": dim,
    }


def scan_manifest(dataset: str, models: Iterable[str] | None = None):
    """Build the manifest of ``dataset`` from the arrays on disk.

    Parameters
    ----------
    dataset : str
        Dataset name.
    models : iterable of str, optional
        Restrict the scan to these model folders (default: all present).

    Returns
    -------
    pandas.DataFrame
        One row per array (:data:`MANIFEST_COLUMNS`), sorted by model and tag.
    """
    import pandas as pd  # noqa: PLC0415

    names = models_present(dataset) if models is None else list(models)
    rows = []
    for model in names:
        edir = config.embeddings_dir(dataset, model)
        for npy in sorted(edir.glob("*.npy")) if edir.is_dir() else []:
            rows.append(manifest_row(dataset, model, npy))
    df = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    return df.sort_values(["model", "source_tag"]).reset_index(drop=True)


def read_manifest(dataset: str):
    """Read ``embeddings/manifest.csv`` of ``dataset`` (empty frame when missing)."""
    import pandas as pd  # noqa: PLC0415

    path = config.embeddings_manifest(dataset)
    if not path.is_file():
        return pd.DataFrame(columns=list(MANIFEST_COLUMNS))
    df = pd.read_csv(path)
    df["codec"] = df["codec"].fillna("").astype(str)
    return df


def write_manifest(df, dataset: str, *, merge: bool = True) -> Path:
    """Write ``df`` as the manifest of ``dataset``.

    With ``merge=True`` the rows are merged into an existing manifest, a new row
    replacing an old one with the same ``(model, source_tag)``.
    """
    import pandas as pd  # noqa: PLC0415

    path = config.embeddings_manifest(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = df
    if merge and path.is_file():
        old = read_manifest(dataset)
        new = (
            pd.concat([old, df])
            .drop_duplicates(subset=["model", "source_tag"], keep="last")
            .reset_index(drop=True)
        )
    new.sort_values(["model", "source_tag"]).to_csv(path, index=False)
    return path
