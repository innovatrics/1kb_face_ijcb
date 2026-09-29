"""Download the aligned face dataset from HuggingFace and export it to disk.

The dataset (see ``face1kb.config.HF_DATASET_ID``) contains the face crops of
the publicly available `AI-Solutions-KK/face_recognition_dataset
<https://huggingface.co/datasets/AI-Solutions-KK/face_recognition_dataset>`_
after face detection and ArcFace-style five-landmark alignment, stored at the
two resolutions evaluated in the paper (112x112 and 224x224 pixels). Each
record carries the identity label and the source-relative file name, so the
export reproduces the exact directory layout used by the experiments::

    data/aligned_<resolution>/<identity_dir>/<image>.png

The script also writes ``data/names.csv`` (one row per image: relative file
name and identity), which defines the canonical image ordering used by all
later pipeline stages.

The dataset is private and available on request from the authors (contact
details are in the paper). Once access has been granted to your HuggingFace
account, authenticate with ``hf auth login`` (``huggingface-cli login`` in
older ``huggingface_hub`` releases) or set the ``HF_TOKEN`` environment
variable before running the script.

Usage
-----
    python -m face1kb.data.download_dataset [--resolutions 112 224]
"""

import argparse
import csv

from datasets import Dataset, load_dataset
from datasets.exceptions import DatasetNotFoundError
from tqdm import tqdm

from face1kb import config

ACCESS_HINT = """\
Cannot access the HuggingFace dataset '{dataset_id}'.

The aligned face crops are available on request from the authors (contact
details are in the paper). Once access has been granted to your HuggingFace
account, authenticate with `hf auth login` (or set the HF_TOKEN environment
variable) and run this command again."""


def export_resolution(dataset: Dataset, resolution: int) -> list[tuple[str, str]]:
    """Export the crops of one resolution to PNG files on disk.

    Parameters
    ----------
    dataset : Dataset
        The full HF dataset holding both resolutions (``resolution`` column).
    resolution : int
        Aligned crop resolution (112 or 224).

    Returns
    -------
    list of (str, str)
        ``(relative_file_name, identity)`` per exported image, sorted.
    """
    out_dir = config.aligned_dir(resolution)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter on the resolution column only, so the image bytes are not decoded
    # for the discarded rows.
    subset = dataset.filter(
        lambda res: res == resolution, input_columns="resolution"
    )

    names = []
    for record in tqdm(subset, desc=f"aligned_{resolution}"):
        rel_name = record["file_name"]
        target = out_dir / rel_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            record["image"].save(target, "PNG")
        names.append((rel_name, record["identity"]))

    names.sort()
    return names


def write_names_csv(names: list[tuple[str, str]]) -> None:
    """Write the canonical image list to ``data/names.csv``."""
    config.NAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(config.NAMES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "identity"])
        writer.writerows(names)
    print(f"Wrote {len(names)} image names to {config.NAMES_CSV}")


def main() -> None:
    """Download all requested resolutions and write the canonical name list."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(config.RESOLUTIONS),
        choices=config.RESOLUTIONS,
        help="Aligned crop resolutions to download.",
    )
    args = parser.parse_args()

    try:
        dataset = load_dataset(config.HF_DATASET_ID, split="train")
    except DatasetNotFoundError as exc:
        raise SystemExit(ACCESS_HINT.format(dataset_id=config.HF_DATASET_ID)) from exc

    names_per_resolution = {}
    for resolution in args.resolutions:
        names_per_resolution[resolution] = export_resolution(dataset, resolution)
        print(f"Exported {len(names_per_resolution[resolution])} images "
              f"to {config.aligned_dir(resolution)}")

    name_sets = {tuple(n) for n in names_per_resolution.values()}
    if len(name_sets) > 1:
        raise RuntimeError(
            "Resolution configs contain different image sets; "
            "names.csv would be ambiguous."
        )

    write_names_csv(next(iter(names_per_resolution.values())))


if __name__ == "__main__":
    main()
