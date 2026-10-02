# SPDX-License-Identifier: MIT
"""Central configuration: roots, dataset names, grids and path helpers.

Every location is taken from an environment variable, with a default inside the
repository checkout, so that no module needs a hardcoded path:

* ``FACE1KB_DATA_ROOT`` (default ``<repo>/data``): inputs -- aligned crops, labels,
  index, pairs.
* ``FACE1KB_WORK_ROOT`` (``<repo>/work``): heavy intermediates -- compressed files,
  decoded caches, embeddings.
* ``FACE1KB_OUTPUT_ROOT`` (``<repo>/outputs``): aggregates, LaTeX tables, figures.
* ``FACE1KB_MODELS_ROOT`` (``<repo>/models``): third-party weights -- FR evaluators,
  JPEG-AI models, insightface, CLIP cache.
* ``FACE1KB_WEIGHTS_DIR`` (``<repo>/weights``): the released face1kb weights.
* ``FACE1KB_JPEGAI_DIR`` (``<repo>/third_party/jpeg-ai-reference-software``): the
  JPEG-AI reference software checkout.
* ``FACE1KB_LAYOUT`` (``public``): directory layout, see below.
* ``FACE1KB_FR_PLUGINS`` (empty): comma-separated module names or ``.py`` file paths
  of modules that register extra face-recognition models with
  ``face1kb.fr.register_onnx`` / ``register_torch`` / ``register_embedder``
  (:data:`FR_PLUGINS`); they are imported on demand by ``face1kb.fr.load``, so the
  models are also available in worker processes and CLIs.

The shipped paper results live in ``<repo>/results`` (:data:`RESULTS_ROOT`) with the
same file layout as :data:`OUTPUT_ROOT`; generators can render the paper tables from
them.

Public data layout::

    DATA_ROOT/<dataset>/aligned_<res>[<suffix>]/<subject>/<stem>.png
    DATA_ROOT/<dataset>/index.csv          # id, rel_path, subject[, pose]
    DATA_ROOT/<dataset>/pairs.parquet      # idx1, idx2, label
    DATA_ROOT/colorferet/labels.csv        # NIST ground-truth labels
    DATA_ROOT/kk/attributes.csv            # estimated attributes

    WORK_ROOT/<dataset>/compressed/<res>[<suffix>]px_<budget>B/<codec>/<rel_stem>.<ext>
    WORK_ROOT/<dataset>/decoded/<res>[<suffix>]px_<budget>B/<codec>/<rel_stem>.png
    WORK_ROOT/<dataset>/embeddings/<model>/<tag>.npy   # row i <-> index.csv id i
    WORK_ROOT/<dataset>/embeddings/manifest.csv

``index.csv`` holds the single canonical row order of a dataset: a directory scan of
the aligned crops sorted by ``(subject, stem)``. Embedding tags follow the grammar
``aligned_<res><suffix>`` or ``<codec>_<res><suffix>_<budget>`` (:func:`embedding_tag`).

``FACE1KB_LAYOUT=legacy`` maps the same helpers onto the directory layout of the
original research data store (``dat_aligned_<res>`` crop folders, an
``alignment/pairs_images.csv`` index, ``ai_solutions_kk_aligned`` as the KK folder,
compressed files and embeddings inside the dataset folder, decoded caches as
``<codec>_png`` siblings of the bitstream folders). It exists only so that a copy of
that store can be read without re-arranging it: point both ``FACE1KB_DATA_ROOT`` and
``FACE1KB_WORK_ROOT`` at its ``datasets`` folder. The mapping covers paths only; the
legacy AI-Solutions-KK index has the columns ``id, rel_path, identity, det_score,
residual_px`` with ``.jpg`` relative paths (the crops themselves are PNG). Read
indices with :func:`read_index`, which normalises both layouts to the public schema.

Attributes are resolved from the environment once, at import time. The helpers read
the module attributes on every call, so tests may monkeypatch, e.g.,
``face1kb.config.DATA_ROOT``.

The ``<repo>`` defaults assume the package is used from a repository checkout (an
editable install, ``pip install -e .``, or ``PYTHONPATH``). With a regular install
the package lives in ``site-packages``, which contains neither the weights nor the
data: set the ``FACE1KB_*`` variables explicitly in that case (at least
``FACE1KB_WEIGHTS_DIR``).
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repository checkout (parent of the ``face1kb`` package).
REPO_ROOT = Path(__file__).resolve().parents[1]
#: True when the package is imported from an installed copy (``site-packages``)
#: rather than from a repository checkout; the ``<repo>`` defaults are then unusable.
INSTALLED_COPY = any(
    part in ("site-packages", "dist-packages") for part in REPO_ROOT.parts
)


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else default


#: Inputs: aligned crops, labels, index and pairs.
DATA_ROOT = _env_path("FACE1KB_DATA_ROOT", REPO_ROOT / "data")
#: Heavy intermediates: compressed files, decoded caches, embeddings.
WORK_ROOT = _env_path("FACE1KB_WORK_ROOT", REPO_ROOT / "work")
#: Aggregates, LaTeX tables and figures produced by the experiments.
OUTPUT_ROOT = _env_path("FACE1KB_OUTPUT_ROOT", REPO_ROOT / "outputs")
#: Third-party weights (FR evaluators, JPEG-AI models, insightface, CLIP cache).
MODELS_ROOT = _env_path("FACE1KB_MODELS_ROOT", REPO_ROOT / "models")
#: Released face1kb weights (``*.safetensors``, tracked with git LFS).
WEIGHTS_DIR = _env_path("FACE1KB_WEIGHTS_DIR", REPO_ROOT / "weights")
#: Checkout of the JPEG-AI reference software.
JPEGAI_DIR = _env_path(
    "FACE1KB_JPEGAI_DIR", REPO_ROOT / "third_party" / "jpeg-ai-reference-software"
)
#: Shipped paper results (read-only reference, same layout as OUTPUT_ROOT).
RESULTS_ROOT = REPO_ROOT / "results"

#: Face-recognition plugin modules (``FACE1KB_FR_PLUGINS``: comma-separated module
#: names or ``.py`` paths), imported by ``face1kb.fr.load_plugins``. ``None`` when the
#: variable is unset at import time; ``face1kb.fr`` then reads it at call time.
FR_PLUGINS: str | None = os.environ.get("FACE1KB_FR_PLUGINS", "").strip() or None

#: Directory layout: ``"public"`` (default) or ``"legacy"`` (see module docstring).
LAYOUT = os.environ.get("FACE1KB_LAYOUT", "").strip().lower() or "public"
if LAYOUT not in ("public", "legacy"):
    raise ValueError(f"FACE1KB_LAYOUT must be 'public' or 'legacy', got {LAYOUT!r}")

# --------------------------------------------------------------------- constants
#: Evaluation datasets (Color FERET and AI-Solutions-KK).
DATASETS: tuple[str, ...] = ("colorferet", "kk")
#: Human-readable dataset names.
DATASET_LABELS: dict[str, str] = {"colorferet": "Color FERET", "kk": "AI-Solutions-KK"}
#: Square crop resolutions of the benchmark grid (px).
RESOLUTIONS: tuple[int, ...] = (64, 96, 112, 168, 224)
#: Byte budgets of the benchmark grid.
BUDGETS: tuple[int, ...] = (1024, 512)
#: Variants of the face1kb codec (names accepted by ``face1kb.load``).
CODEC_VARIANTS: tuple[str, ...] = ("fast", "accurate")
#: Codec names of the two face1kb variants in result files -> variant.
OURS_CODECS: dict[str, str] = {"ours_fast": "fast", "ours_accurate": "accurate"}
#: Row labels of the two face1kb variants in the paper tables.
OURS_LABELS: dict[str, str] = {
    "ours_fast": "Ours-FAST",
    "ours_accurate": "Ours-ACCURATE",
}
#: File extension of the face1kb container (bitstream files of the benchmark).
OURS_EXT = ".bin"
#: Released weight files, by variant.
WEIGHT_FILES: dict[str, str] = {
    "fast": "face1kb_fast.safetensors",
    "accurate": "face1kb_accurate.safetensors",
}

# Folder / file names of the original research data store (FACE1KB_LAYOUT=legacy).
_LEGACY_DATASET_DIRS = {"colorferet": "colorferet", "kk": "ai_solutions_kk_aligned"}
_LEGACY_ATTRIBUTES = {"kk": "ai_solutions_kk_attributes.csv"}


def _check_dataset(dataset: str) -> str:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    return dataset


def _legacy() -> bool:
    return LAYOUT == "legacy"


def _dataset_dirname(dataset: str) -> str:
    _check_dataset(dataset)
    return _LEGACY_DATASET_DIRS[dataset] if _legacy() else dataset


def cell_name(res: int, budget: int, suffix: str = "") -> str:
    """Name of a compression cell folder, e.g. ``112px_1024B`` or ``112_A1px_512B``."""
    return f"{int(res)}{suffix}px_{int(budget)}B"


# --------------------------------------------------------------------- inputs
def dataset_dir(dataset: str) -> Path:
    """Input root of ``dataset`` under :data:`DATA_ROOT`."""
    return DATA_ROOT / _dataset_dirname(dataset)


def aligned_dir(dataset: str, res: int, suffix: str = "") -> Path:
    """Folder of the aligned ``res`` px crops (``suffix`` selects a variant set)."""
    prefix = "dat_aligned" if _legacy() else "aligned"
    return dataset_dir(dataset) / f"{prefix}_{int(res)}{suffix}"


def index_csv(dataset: str) -> Path:
    """Canonical image index (``id, rel_path, subject[, pose]``)."""
    if _legacy():
        return dataset_dir(dataset) / "alignment" / "pairs_images.csv"
    return dataset_dir(dataset) / "index.csv"


def read_index(dataset: str):
    """Read :func:`index_csv` as a DataFrame with the public columns.

    Returns ``id, rel_path, subject[, pose]`` (plus any further columns of the
    file), with ``rel_path`` pointing at the ``.png`` crop relative to
    :func:`aligned_dir`. In the legacy layout the AI-Solutions-KK index names the
    subject column ``identity`` and lists ``.jpg`` paths, and the legacy Color FERET
    index stores the subject unpadded (``1``); all are normalised here. ``subject``
    is always a string equal to the crop folder name (e.g. ``00001``), as in
    :func:`face1kb.data.index.read_index_csv`.
    """
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(
        index_csv(dataset), dtype={"subject": str, "identity": str, "rel_path": str}
    )
    if "subject" not in df.columns and "identity" in df.columns:
        df = df.rename(columns={"identity": "subject"})
    if _legacy() and dataset == "colorferet":
        # The legacy index stores the subject unpadded (1); use the folder name.
        df["subject"] = [Path(p).parts[0] for p in df.rel_path]
    df["rel_path"] = [str(Path(p).with_suffix(".png").as_posix()) for p in df.rel_path]
    return df


def pairs_parquet(dataset: str) -> Path:
    """Verification pairs (``idx1, idx2, label``; indices into :func:`index_csv`)."""
    if _legacy():
        return dataset_dir(dataset) / "alignment" / "pairs.parquet"
    return dataset_dir(dataset) / "pairs.parquet"


def labels_csv(dataset: str = "colorferet") -> Path:
    """Ground-truth subject labels (Color FERET, parsed from the NIST release)."""
    return dataset_dir(dataset) / "labels.csv"


def attributes_csv(dataset: str = "kk") -> Path:
    """Estimated per-image attributes (AI-Solutions-KK)."""
    if _legacy() and dataset in _LEGACY_ATTRIBUTES:
        return DATA_ROOT.parent / "attributes" / _LEGACY_ATTRIBUTES[dataset]
    return dataset_dir(dataset) / "attributes.csv"


# --------------------------------------------------------------------- work
def work_dir(dataset: str) -> Path:
    """Work root of ``dataset`` under :data:`WORK_ROOT`."""
    return WORK_ROOT / _dataset_dirname(dataset)


def compressed_dir(
    dataset: str, res: int, budget: int, codec: str, suffix: str = ""
) -> Path:
    """Folder of the bitstreams of one (dataset, res, budget, codec) cell."""
    return work_dir(dataset) / "compressed" / cell_name(res, budget, suffix) / codec


def decoded_dir(
    dataset: str, res: int, budget: int, codec: str, suffix: str = ""
) -> Path:
    """Folder of the decoded PNG cache of one cell (``<rel_stem>.png``)."""
    if _legacy():
        cell = work_dir(dataset) / "compressed" / cell_name(res, budget, suffix)
        return cell / f"{codec}_png"
    return work_dir(dataset) / "decoded" / cell_name(res, budget, suffix) / codec


def embedding_tag(
    res: int, suffix: str = "", codec: str | None = None, budget: int | None = None
) -> str:
    """Return ``aligned_<res><suffix>`` or ``<codec>_<res><suffix>_<budget>``."""
    if codec is None or codec == "aligned":
        return f"aligned_{int(res)}{suffix}"
    if budget is None:
        raise ValueError("budget is required for a compressed-source tag")
    return f"{codec}_{int(res)}{suffix}_{int(budget)}"


def embeddings_dir(dataset: str, model: str) -> Path:
    """Folder of the embedding arrays of ``model`` on ``dataset``."""
    return work_dir(dataset) / "embeddings" / model


def embeddings_path(dataset: str, model: str, tag: str) -> Path:
    """Embedding array ``(N, D)`` float32; row ``i`` is image id ``i`` of the index."""
    return embeddings_dir(dataset, model) / f"{tag}.npy"


def embeddings_manifest(dataset: str) -> Path:
    """Manifest CSV of the embedding arrays of ``dataset``."""
    return work_dir(dataset) / "embeddings" / "manifest.csv"


# --------------------------------------------------------------------- outputs
def output_dir(*parts: str) -> Path:
    """Folder under :data:`OUTPUT_ROOT` (e.g. ``output_dir("accuracy")``)."""
    return OUTPUT_ROOT.joinpath(*parts)


def results_dir(*parts: str) -> Path:
    """Folder under :data:`RESULTS_ROOT` (the shipped paper results)."""
    return RESULTS_ROOT.joinpath(*parts)


# --------------------------------------------------------------------- weights
def weights_dir_hint() -> str:
    """Explain how to point at the weights when the ``<repo>`` default is unusable.

    Returns an empty string when the package runs from a repository checkout or
    ``FACE1KB_WEIGHTS_DIR`` is set.
    """
    if not INSTALLED_COPY or os.environ.get("FACE1KB_WEIGHTS_DIR", "").strip():
        return ""
    return (
        "face1kb is imported from an installed copy, which does not contain the "
        "weights; set FACE1KB_WEIGHTS_DIR to the weights/ folder of a checkout "
        "(after `git lfs pull`), pass weights_dir=..., or use an editable install "
        "(pip install -e .)"
    )


def weights_path(variant: str) -> Path:
    """Released safetensors file of a face1kb codec variant (``fast``/``accurate``)."""
    if variant not in WEIGHT_FILES:
        raise ValueError(f"unknown variant {variant!r}; choose from {CODEC_VARIANTS}")
    return WEIGHTS_DIR / WEIGHT_FILES[variant]


def models_dir(*parts: str) -> Path:
    """Folder under :data:`MODELS_ROOT` for downloaded third-party weights."""
    return MODELS_ROOT.joinpath(*parts)
