"""Central configuration of the replication pipeline.

Every path can be overridden through an environment variable so the pipeline
runs on any machine without editing the sources. Defaults keep all data and
outputs inside the repository checkout.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Root directory for datasets and intermediate artifacts.
DATA_ROOT = Path(os.environ.get("FACE1KB_DATA_ROOT", REPO_ROOT / "data"))

#: Root directory for computed outputs (embeddings, metrics, reports).
OUTPUT_ROOT = Path(
    os.environ.get("FACE1KB_OUTPUT_ROOT", REPO_ROOT / "outputs")
)

#: HuggingFace dataset with the aligned face crops used in the paper.
HF_DATASET_ID = os.environ.get(
    "FACE1KB_HF_DATASET", "Cha53c/1kb-face-aligned"
)

#: Local checkout of the JPEG-AI reference software (see docs/JPEG_AI.md).
JPEGAI_REPO_DIR = Path(
    os.environ.get(
        "JPEGAI_REPO_DIR",
        REPO_ROOT / "third_party" / "jpeg-ai-reference-software",
    )
)

#: Directory with the proprietary ONNX face-recognition models. The models are
#: NOT distributed with this repository; a mockup with an identical interface
#: is generated instead (see face1kb/embeddings/make_mockup_model.py).
PROPRIETARY_MODELS_DIR = Path(
    os.environ.get(
        "FACE1KB_PROPRIETARY_MODELS", REPO_ROOT / "models" / "proprietary"
    )
)

#: Path of the generated mockup embedding model.
MOCKUP_MODEL_PATH = REPO_ROOT / "models" / "mockup" / "mockup_512d.onnx"

#: Face-crop resolutions evaluated in the paper.
RESOLUTIONS = (112, 224)

#: Per-image compression budget (bytes) imposed by 2D-barcode capacity.
MAX_SIZE_BYTES = 1024

#: Compression algorithms evaluated in the paper.
CODECS = ("jpeg", "jpeg2000", "jpeg_xl", "webp", "jpeg_fzt", "jpeg_ai")

#: File extension of the stored bitstream per codec.
CODEC_EXTENSIONS = {
    "jpeg": ".jpg",
    "jpeg2000": ".jp2",
    "jpeg_xl": ".jxl",
    "webp": ".webp",
    "jpeg_fzt": ".fzt",
    "jpeg_ai": ".jpegai",
}

#: Open-source face-recognition models, evaluated through DeepFace.
DEEPFACE_MODELS = (
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepID",
    "ArcFace",
    "SFace",
)

#: Proprietary model identifiers (Innovatrics); ONNX files are not distributed.
PROPRIETARY_MODELS = ("inno-fast", "inno-balanced", "inno-accurate")

#: FAR operating points at which FRR is reported.
FAR_LEVELS = (0.1, 0.01, 0.001, 0.0001, 0.00001)
FAR_LEVEL_LABELS = ("1:10", "1:100", "1:1000", "1:10000", "1:100000")


def aligned_dir(resolution: int) -> Path:
    """Directory with aligned original PNG images at *resolution*."""
    return DATA_ROOT / f"aligned_{resolution}"


def compressed_dir(resolution: int) -> Path:
    """Directory with per-codec compressed bitstreams at *resolution*."""
    return DATA_ROOT / f"compressed_{resolution}"


def decompressed_dir(resolution: int) -> Path:
    """Directory with per-codec decompressed PNG images at *resolution*."""
    return DATA_ROOT / f"decompressed_{resolution}"


def embeddings_dir(resolution: int) -> Path:
    """Directory with the computed face embeddings at *resolution*."""
    return OUTPUT_ROOT / f"embeddings_{resolution}"


def metrics_dir() -> Path:
    """Directory with the computed metrics, tables and reports."""
    return OUTPUT_ROOT / "metrics"


#: CSV with one row per image: relative file name and identity label.
NAMES_CSV = DATA_ROOT / "names.csv"
