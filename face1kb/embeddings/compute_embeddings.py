"""Compute open-source face embeddings with DeepFace.

Runs the seven open-source recognition models of the paper (VGG-Face,
FaceNet, FaceNet512, OpenFace, DeepID, ArcFace, SFace) over either the
aligned originals or the decoded images of one codec, in the canonical
``names.csv`` order, and stores one ``(N, D)`` float array per model::

    outputs/embeddings_<resolution>/<Model>_embeddings.npy            # orig
    outputs/embeddings_<resolution>/<Model>_<codec>_embeddings.npy    # codec

Images are already tightly aligned face crops; DeepFace is called with
``enforce_detection=False`` so images where its internal detector finds no
face are still embedded (identical to the paper's setup).

Usage
-----
    python -m face1kb.embeddings.compute_embeddings --resolution 112 \\
        --source original
    python -m face1kb.embeddings.compute_embeddings --resolution 112 \\
        --source jpeg_ai
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from face1kb import config
from face1kb.data.pairs import load_names


def source_image_dir(resolution: int, source: str) -> Path:
    """Directory with the input PNGs for *source* at *resolution*."""
    if source == "original":
        return config.aligned_dir(resolution)
    return config.decompressed_dir(resolution) / source


def output_path(resolution: int, source: str, model_name: str) -> Path:
    """NPY path for the embeddings of *model_name* on *source*."""
    safe_model = model_name.replace("-", "_").replace(" ", "_")
    suffix = "" if source == "original" else f"_{source}"
    directory = config.embeddings_dir(resolution)
    return directory / f"{safe_model}{suffix}_embeddings.npy"


def compute_for_model(
    model_name: str, names: list[str], image_dir: Path
) -> np.ndarray:
    """Embed all images with one DeepFace model, keeping canonical order."""
    from deepface import DeepFace

    embeddings = []
    for name in tqdm(names, desc=model_name):
        result = DeepFace.represent(
            img_path=str(image_dir / name),
            model_name=model_name,
            enforce_detection=False,
        )
        if not result:
            raise RuntimeError(
                f"No embedding returned for {name}; aborting to keep "
                f"the canonical image ordering intact."
            )
        embeddings.append(np.asarray(result[0]["embedding"]))
    return np.stack(embeddings)


def main() -> None:
    """Compute embeddings for all requested models on one image source."""
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
        help="Embed the aligned originals or one codec's decoded images.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(config.DEEPFACE_MODELS),
        choices=config.DEEPFACE_MODELS,
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models whose output NPY already exists.",
    )
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
        embeddings = compute_for_model(model_name, names, image_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, embeddings)
        print(f"{model_name}: saved {embeddings.shape} to {out_path}")


if __name__ == "__main__":
    main()
