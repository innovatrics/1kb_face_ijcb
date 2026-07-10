"""Build and upload the aligned face dataset to the HuggingFace Hub.

Packages the aligned 112x112 and 224x224 face crops (produced from the
public `AI-Solutions-KK/face_recognition_dataset
<https://huggingface.co/datasets/AI-Solutions-KK/face_recognition_dataset>`_
with a proprietary face detector and ArcFace-style five-landmark alignment)
into one Hub dataset with two configs, ``aligned_112`` and ``aligned_224``.
Each record holds:

- ``image``      - the aligned PNG crop,
- ``identity``   - identity label (105 unique values),
- ``file_name``  - source-relative file name (defines directory layout and
  the canonical evaluation ordering),
- ``resolution`` - crop resolution in pixels (112 or 224).

Requires ``hf auth login`` with write access to the target namespace.

Usage
-----
    python tools/build_hf_dataset.py \\
        --source-112 /path/to/dat_aligned_112 \\
        --source-224 /path/to/dat_aligned_224 \\
        --repo-id <namespace>/1kb-face-aligned [--private]
"""

import argparse
import sys
from pathlib import Path

from datasets import Dataset, Features, Image, Value
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face1kb import config  # noqa: E402

DATASET_CARD = """\
---
license: mit
task_categories:
- image-classification
tags:
- face-recognition
- biometrics
- image-compression
pretty_name: 1 kB Face - aligned verification crops (IJCB 2026)
---

# Aligned face crops for *Face Recognition at One Kilobyte* (IJCB 2026)

Aligned face crops used by the IJCB 2026 paper *Face Recognition at One
Kilobyte: Evaluating Image Compression Algorithms for Barcode-Constrained
Biometric Verification*. Derived from the MIT-licensed
[AI-Solutions-KK/face_recognition_dataset][source] (105 identities,
17,534 images) by face detection with a proprietary detector followed by
ArcFace-style five-landmark alignment, exported at two resolutions as the
configs `aligned_112` and `aligned_224`.

[source]: https://huggingface.co/datasets/AI-Solutions-KK/face_recognition_dataset

Each record: `image` (PNG crop), `identity` (105 unique labels),
`file_name` (source-relative name defining the canonical evaluation
ordering), `resolution` (112 or 224).

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", "aligned_112", split="train")
```

Replication code: https://github.com/innovatrics/1kb_face_ijcb
"""


def build_split(source_dir: Path, resolution: int) -> Dataset:
    """Create one resolution config from a directory of identity folders."""
    image_paths = sorted(source_dir.rglob("*.png"))
    if not image_paths:
        raise SystemExit(f"No PNG images found under {source_dir}")
    print(f"aligned_{resolution}: {len(image_paths)} images")

    def generate():
        for path in image_paths:
            rel_path = path.relative_to(source_dir)
            yield {
                "image": str(path),
                "identity": rel_path.parts[0].removeprefix("pins_"),
                "file_name": str(rel_path),
                "resolution": resolution,
            }

    features = Features(
        {
            "image": Image(),
            "identity": Value("string"),
            "file_name": Value("string"),
            "resolution": Value("int32"),
        }
    )
    return Dataset.from_generator(generate, features=features)


def main() -> None:
    """Build both configs and push them to the Hub."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source-112", required=True, type=Path)
    parser.add_argument("--source-224", required=True, type=Path)
    parser.add_argument("--repo-id", default=config.HF_DATASET_ID)
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repository as private.",
    )
    args = parser.parse_args()

    sources = {112: args.source_112, 224: args.source_224}
    for resolution, source_dir in sources.items():
        dataset = build_split(source_dir, resolution)
        dataset.push_to_hub(
            args.repo_id,
            config_name=f"aligned_{resolution}",
            split="train",
            private=args.private,
        )
        print(f"Pushed aligned_{resolution} to {args.repo_id}")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=DATASET_CARD.format(repo_id=args.repo_id).encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded dataset card to {args.repo_id}")


if __name__ == "__main__":
    main()
