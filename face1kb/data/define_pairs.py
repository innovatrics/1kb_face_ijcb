"""Report and optionally export the verification pairs used by the paper.

The pair protocol itself is defined in ``face1kb.data.pairs`` (all
non-redundant image pairs, mated when both images share an identity). This
CLI prints the resulting pair statistics and can export a human-readable
sample or, on request, the complete pair list.

Usage
-----
    python -m face1kb.data.define_pairs [--sample-csv PATH --sample-size 1000]
    python -m face1kb.data.define_pairs --full-csv PATH   # tens of GB!
"""

import argparse
import csv
from itertools import combinations

import numpy as np

from face1kb import config
from face1kb.data.pairs import load_names, pair_counts


def export_sample(names: list[str], identity_ids: np.ndarray, path: str,
                  sample_size: int) -> None:
    """Write a deterministic sample of mated and non-mated pairs to CSV."""
    rng = np.random.default_rng(seed=42)
    n = len(names)
    rows = []
    while len(rows) < sample_size:
        i, j = sorted(rng.integers(0, n, size=2))
        if i == j:
            continue
        label = int(identity_ids[i] == identity_ids[j])
        rows.append([names[i], names[j], label])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name1", "name2", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sampled pairs to {path}")


def export_full(names: list[str], identity_ids: np.ndarray, path: str) -> None:
    """Write the complete pair list to CSV (very large; ~150 M rows)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name1", "name2", "label"])
        for i, j in combinations(range(len(names)), 2):
            writer.writerow(
                [names[i], names[j], int(identity_ids[i] == identity_ids[j])]
            )
    print(f"Wrote full pair list to {path}")


def main() -> None:
    """Print pair statistics and run the requested exports."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample-csv", help="Write a sampled pair CSV here.")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument(
        "--full-csv",
        help="Write the COMPLETE pair list here (tens of GB).",
    )
    args = parser.parse_args()

    names, identity_ids = load_names()
    n_mated, n_nonmated = pair_counts(identity_ids)
    n_identities = len(np.unique(identity_ids))

    print(f"Images:          {len(names)}")
    print(f"Identities:      {n_identities}")
    print(f"Mated pairs:     {n_mated:,}")
    print(f"Non-mated pairs: {n_nonmated:,}")
    print(f"Budget:          {config.MAX_SIZE_BYTES} B/image")

    if args.sample_csv:
        export_sample(names, identity_ids, args.sample_csv, args.sample_size)
    if args.full_csv:
        export_full(names, identity_ids, args.full_csv)


if __name__ == "__main__":
    main()
