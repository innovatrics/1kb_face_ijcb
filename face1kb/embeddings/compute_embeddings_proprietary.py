"""Compute embeddings with the proprietary ONNX models (or the mockup).

The paper additionally evaluates three proprietary Innovatrics models
(``inno-fast``, ``inno-balanced``, ``inno-accurate``; 512-D embeddings).
Those models are **not distributed** with this repository. This script looks
for them under ``models/proprietary/<name>.onnx`` and, when one is absent,
falls back to that model's own deterministic mockup
(``models/mockup/<name>_mockup_512d.onnx``, generated on first use by
``face1kb.embeddings.make_mockup_model``) - printing a prominent warning,
because mockup embeddings carry no biometric meaning.

Preprocessing matches the paper's implementation exactly: the image is read
in BGR channel order, resized to 112x112, scaled to ``[-1, 1]`` and fed as
``(1, 3, 112, 112)`` through input ``input.1``.

Usage
-----
    python -m face1kb.embeddings.compute_embeddings_proprietary \\
        --resolution 112 --source original
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm

from face1kb import config
from face1kb.data.pairs import load_names
from face1kb.embeddings.compute_embeddings import (
    output_path,
    source_image_dir,
)

MOCKUP_WARNING = """
{border}
WARNING: proprietary model '{name}' not found at
  {expected}
Falling back to its MOCKUP model
  {mockup}
Its embeddings carry NO biometric meaning; the resulting metrics are
placeholders only. To reproduce the paper's proprietary-model results,
obtain the models from Innovatrics.
{border}
"""


def resolve_model_path(model_name: str) -> Path:
    """Return the ONNX path for *model_name*, falling back to its mockup."""
    real_path = config.PROPRIETARY_MODELS_DIR / f"{model_name}.onnx"
    if real_path.exists():
        return real_path

    mockup_path = config.mockup_model_path(model_name)
    print(
        MOCKUP_WARNING.format(
            border="=" * 70,
            name=model_name,
            expected=real_path,
            mockup=mockup_path,
        )
    )
    if not mockup_path.exists():
        from face1kb.embeddings import make_mockup_model

        make_mockup_model.export(model_name, mockup_path)
    return mockup_path


def preprocess(image_path: Path) -> np.ndarray:
    """Load an image as the ``(1, 3, 112, 112)`` BGR tensor in [-1, 1]."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    image = cv2.resize(image, (112, 112))
    tensor = ((image.astype(np.float32) / 255.0) - 0.5) * 2.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.expand_dims(tensor, 0)


def compute_for_model(
    model_path: Path, names: list[str], image_dir: Path, desc: str
) -> np.ndarray:
    """Embed all images with one ONNX model, keeping canonical order."""
    session = onnxruntime.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    embeddings = []
    for name in tqdm(names, desc=desc):
        tensor = preprocess(image_dir / name)
        embedding = session.run(None, {"input.1": tensor})[0][0]
        embeddings.append(embedding)
    return np.stack(embeddings)


def main() -> None:
    """Compute proprietary (or mockup) embeddings on one image source."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        choices=config.RESOLUTIONS,
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=("original", *config.CODECS),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(config.PROPRIETARY_MODELS),
        choices=config.PROPRIETARY_MODELS,
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    names, _ = load_names()
    image_dir = source_image_dir(args.resolution, args.source)
    if not image_dir.is_dir():
        raise SystemExit(f"Input directory not found: {image_dir}")

    for model_name in args.models:
        out_path = output_path(args.resolution, args.source, model_name)
        if args.skip_existing and out_path.exists():
            print(f"skipping {model_name} ({out_path.name} exists)")
            continue
        model_path = resolve_model_path(model_name)
        embeddings = compute_for_model(model_path, names, image_dir, desc=model_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, embeddings)
        print(f"{model_name}: saved {embeddings.shape} to {out_path}")


if __name__ == "__main__":
    main()
