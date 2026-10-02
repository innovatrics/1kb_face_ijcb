# SPDX-License-Identifier: MIT
r"""Recompression study: a face compressed once and then compressed again.

For one dataset, crop resolution and byte budget this script runs the two-pass chain

    aligned crop --(source codec @ budget)--> decode --(second codec @ budget)--> decode

over the 8 x 8 matrix of the six classical codecs, JPEG-FzT and Ours-ACCURATE, scores
every chain with one face matcher and writes one aggregate CSV per (dataset, budget)::

    OUTPUT_ROOT/recompression/recompression_<dataset>_<budget>.csv
    columns: dataset, res, budget, source_codec, second_codec, eer,
             delta_eer_vs_singlepass

for the default matcher ``edgeface_xs``; another matcher (``--model``) writes
``recompression_<dataset>_<model>.csv`` with the rows of every budget.

Three stages, selected with ``--stages``:

``produce`` (CPU pool)
    Writes the doubly compressed files of the 7 x 7 classical/JPEG-FzT block to
    ``<recompressed root>/<src>__<dst>_<res>_<budget>/<subject>/<stem>.<ext>`` (plus a
    per-cell manifest parquet under ``manifest/``). Files that exist are skipped, so
    the stage resumes. The first pass encodes the aligned PNG; the decoded first-pass
    image is the input of the second pass. As in the paper, that image keeps the
    decoder's metadata: a decoded JPEG XL file carries a 536-byte ICC profile, which
    the AVIF and HEIF encoders embed in their output, so those chains spend half of
    a 1 kB budget on it (``--strip-metadata`` drops it). Both passes use the
    benchmark binary search (JPEG-FzT over the even qualities 2..94).
``score`` (1 GPU)
    Embeds every cell of the block with the matcher, computes the EER over all mated
    pairs plus ``--sample-nonmated`` seeded non-mated pairs of ``pairs.parquet``
    (crops missing from a cell are NaN rows and drop out), and the single-pass
    reference EER of the second codec from the benchmark's compressed files
    (``compressed/<res>px_<budget>B/<codec>``, all index rows).
``ours`` (1 GPU)
    The 15 cells with Ours-ACCURATE as source and/or second codec. The codec runs on
    the GPU, so these cells are scored on a deterministic subject-spread sample of
    ``--ours-n`` crops (at most ``--ours-max-per-subject`` per subject) with the pairs
    restricted to the sample; the single-pass references of these rows are computed
    on the same sample (Ours-ACCURATE from the stored ``.bin`` files when present,
    otherwise by encoding). A failed face1kb decode gives a black frame.

Rows of the output CSV that this run computes replace existing rows with the same
(res, budget, source_codec, second_codec); cells already present are skipped unless
``--rescore`` is given.

The paper settings: 112 px, 1024 and 512 B, ``edgeface_xs`` (plus the
``arcface_antelopev2`` rescore on Color FERET), 5,000,000 non-mated pairs, seed 0;
all 11,335 Color FERET crops and the first 2,500 AI-Solutions-KK crops of the index
(``--limit 2500``) for the classical block; 120 crops for the Ours-ACCURATE cells.

Examples
--------
    python experiments/recompression/recompress.py --dataset colorferet --budget 1024
    python experiments/recompression/recompress.py --dataset kk --budget 512 \
        --limit 2500
    python experiments/recompression/recompress.py --dataset colorferet --budget 1024 \
        --model arcface_antelopev2 --stages score,ours
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from face1kb import config
from face1kb.baselines import classical, jpeg_fzt
from face1kb.baselines import extension as codec_extension
from face1kb.data.cli_utils import parse_list, setup_logging
from face1kb.eval import verification

log = logging.getLogger("recompress")

#: First- and second-pass codecs of the CPU block (paper order).
CLASSICAL: tuple[str, ...] = (*classical.CODECS, "jpeg_fzt")
#: The learned codec of the matrix.
OURS = "ours_accurate"
#: Matrix order of the tables (rows = source, columns = second codec).
MATRIX: tuple[str, ...] = (*CLASSICAL, OURS)
#: Default scoring matcher (its CSV carries no model suffix).
DEFAULT_MODEL = "edgeface_xs"
#: Columns of the output CSV.
COLUMNS: tuple[str, ...] = (
    "dataset",
    "res",
    "budget",
    "source_codec",
    "second_codec",
    "eer",
    "delta_eer_vs_singlepass",
)
KEYS = ["res", "budget", "source_codec", "second_codec"]
#: Embedding batch (the paper embedded in index blocks of 256 rows).
EMBED_BATCH = 256
#: Embedding width assumed for matchers without a ``dim`` attribute.
EMB_DIM = 512
STAGES = ("produce", "score", "ours")


# ----------------------------------------------------------------------- paths
def recompressed_root(dataset: str) -> Path:
    """Return the default folder of the recompressed cells (under WORK_ROOT)."""
    return config.work_dir(dataset) / "recompressed"


def cell_dir(root: Path, src: str, dst: str, res: int, budget: int) -> Path:
    """Folder of one chain cell: ``<root>/<src>__<dst>_<res>_<budget>``."""
    return Path(root) / f"{src}__{dst}_{int(res)}_{int(budget)}"


def output_csv(out_dir: Path, dataset: str, budget: int, model: str) -> Path:
    """Aggregate CSV: ``recompression_<ds>_<budget>.csv`` for the default matcher.

    Another matcher writes ``recompression_<ds>_<model>.csv``, one file for all
    budgets (the layout of the shipped ArcFace rescore).
    """
    if model == DEFAULT_MODEL:
        return Path(out_dir) / f"recompression_{dataset}_{int(budget)}.csv"
    return Path(out_dir) / f"recompression_{dataset}_{model}.csv"


# ----------------------------------------------------------------------- codecs
def encode(img, codec: str, budget: int, strip_metadata: bool = False):
    """One budget-fitted pass of a classical codec or JPEG-FzT (binary search).

    Returns ``(bytes, info)``. ``strip_metadata`` drops the ``info`` (ICC profile,
    comment) of a PIL input before a classical encode.
    """
    if codec == "jpeg_fzt":
        return jpeg_fzt.encode_to_budget(img, budget, qualities=jpeg_fzt.QUALITIES_EVEN)
    return classical.encode_to_budget(img, codec, budget, strip_metadata=strip_metadata)


def decode_chain(data: bytes, codec: str, size: tuple[int, int]):
    """Decode a pass for the next pass: PIL image (classical) or array (JPEG-FzT)."""
    if codec == "jpeg_fzt":
        return jpeg_fzt.decode(data, size)
    return classical.decode_pil(data)


def roundtrip(arr: np.ndarray, codec: str, budget: int) -> np.ndarray:
    """``uint8`` crop -> codec @ budget -> decoded ``uint8`` crop (no metadata)."""
    data, _ = encode(arr, codec, budget)
    if codec == "jpeg_fzt":
        return jpeg_fzt.decode(data, (arr.shape[1], arr.shape[0]))
    return classical.decode(data)


# ----------------------------------------------------------------------- produce
def _process(task: tuple) -> dict:
    """Recompress one crop through ``src`` then ``dst``; write the final file."""
    in_path, out_path, rel_path, src, dst, res, budget, strip = task
    from PIL import Image  # noqa: PLC0415

    subject, image = str(Path(rel_path).parent), Path(rel_path).stem
    base = {
        "image": image,
        "subject": subject,
        "source_codec": src,
        "second_codec": dst,
        "res": res,
        "budget": budget,
    }
    out_path = Path(out_path)
    if out_path.exists():
        nbytes = out_path.stat().st_size
        return {
            **base,
            "bytes_first": float("nan"),
            "bytes_final": int(nbytes),
            "fitted": bool(nbytes <= budget),
            "encode_ms": float("nan"),
            "skipped": True,
        }
    try:
        with Image.open(in_path) as im:
            img = im.convert("RGB")
        size = img.size
        t0 = time.perf_counter()
        data1, info1 = encode(img, src, budget)
        mid = decode_chain(data1, src, size)
        data2, info2 = encode(mid, dst, budget, strip_metadata=strip)
        ms = (time.perf_counter() - t0) * 1000.0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".part")
        tmp.write_bytes(data2)
        os.replace(tmp, out_path)
        return {
            **base,
            "bytes_first": int(info1["size"]),
            "bytes_final": int(info2["size"]),
            "fitted": bool(info1["fitted"] and info2["fitted"]),
            "encode_ms": round(ms, 2),
            "skipped": False,
        }
    except Exception as e:  # noqa: BLE001 - one bad crop must not stop the pool
        return {
            **base,
            "bytes_first": -1,
            "bytes_final": -1,
            "fitted": False,
            "encode_ms": -1.0,
            "skipped": False,
            "error": str(e)[:200],
        }


def produce(
    dataset: str,
    rel_paths: list[str],
    sources: list[str],
    seconds: list[str],
    res: int,
    budget: int,
    root: Path,
    workers: int,
    strip_metadata: bool = False,
) -> pd.DataFrame:
    """Write the recompressed files of every (source, second) cell."""
    from tqdm import tqdm  # noqa: PLC0415

    src_dir = config.aligned_dir(dataset, res)
    tasks = []
    for rel in rel_paths:
        for src in sources:
            for dst in seconds:
                out = cell_dir(root, src, dst, res, budget) / Path(rel).with_suffix(
                    codec_extension(dst)
                )
                tasks.append(
                    (
                        str(src_dir / rel),
                        str(out),
                        rel,
                        src,
                        dst,
                        res,
                        budget,
                        strip_metadata,
                    )
                )
    log.info(
        "[%s] produce: %d crops x %d x %d cells = %d tasks, %d workers",
        dataset,
        len(rel_paths),
        len(sources),
        len(seconds),
        len(tasks),
        workers,
    )
    if workers <= 1:
        records = [_process(t) for t in tqdm(tasks, unit="task")]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            records = list(
                tqdm(ex.map(_process, tasks, chunksize=16), total=len(tasks))
            )
    df = pd.DataFrame(records)
    mdir = Path(root) / "manifest"
    for (src, dst), g in df.groupby(["source_codec", "second_codec"]):
        if g["skipped"].all():
            continue  # nothing new in this cell: keep its manifest as it is
        mdir.mkdir(parents=True, exist_ok=True)
        shard = mdir / f"{src}__{dst}_{res}_{budget}.parquet"
        if shard.exists():
            old = pd.read_parquet(shard)
            new_keys = set(zip(g.subject, g.image, strict=True))
            keep = [
                (s, i) not in new_keys
                for s, i in zip(old.subject, old.image, strict=True)
            ]
            g = pd.concat([old[keep], g], ignore_index=True)
        g.to_parquet(shard, index=False)
    errors = int((df.bytes_final < 0).sum())
    if errors:
        log.warning(
            "[%s] %d recompression tasks failed (manifest 'error')", dataset, errors
        )
    return df


# ----------------------------------------------------------------------- scoring
def to_model_input(arr: np.ndarray) -> np.ndarray:
    """Bilinear resize of an RGB crop to the 112 x 112 matcher input."""
    if arr.shape[0] != 112 or arr.shape[1] != 112:
        import cv2  # noqa: PLC0415

        arr = cv2.resize(arr, (112, 112), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(arr, dtype=np.uint8)


def embed_dir(model, folder: Path, ext: str, rel_paths: list[str]) -> np.ndarray:
    """Embed ``<folder>/<rel_stem><ext>`` for every index row; NaN rows = missing.

    Rows are processed in index blocks of :data:`EMBED_BATCH`; the crops of a block
    that exist are embedded in one call.
    """
    from face1kb.baselines import decode_file  # noqa: PLC0415

    n = len(rel_paths)
    out = np.full((n, getattr(model, "dim", EMB_DIM)), np.nan, dtype=np.float32)
    for start in range(0, n, EMBED_BATCH):
        arrs, ok = [], []
        for i in range(start, min(start + EMBED_BATCH, n)):
            fp = Path(folder) / Path(rel_paths[i]).with_suffix(ext)
            if not fp.exists():
                continue
            try:
                arrs.append(to_model_input(decode_file(fp)))
                ok.append(i)
            except Exception:  # noqa: BLE001 - an undecodable file is a missing row
                continue
        if arrs:
            out[ok] = np.asarray(model.embed(arrs), dtype=np.float32)
    return out


def eer_of(emb: np.ndarray, pos_idx, neg_idx) -> float:
    """EER over the shared pair set (NaN scores of missing rows are dropped)."""
    pos = verification.cosine_scores(emb, *pos_idx)
    neg = verification.cosine_scores(emb, *neg_idx)
    return verification.eer(pos, neg)


def score_block(
    dataset: str,
    model,
    rel_paths: list[str],
    sources: list[str],
    seconds: list[str],
    res: int,
    budget: int,
    root: Path,
    pairs,
    done: set,
    on_row,
) -> None:
    """Score the classical/JPEG-FzT cells; ``on_row(row)`` receives each result."""
    pos_idx, neg_idx = pairs
    single: dict[str, float] = {}
    for src in sources:
        for dst in seconds:
            if (src, dst) in done:
                log.info("  skip %s -> %s (already scored)", src, dst)
                continue
            cell = cell_dir(root, src, dst, res, budget)
            if not cell.is_dir():
                log.warning("  skip %s -> %s: %s does not exist", src, dst, cell)
                continue
            ext = codec_extension(dst)
            eer = eer_of(embed_dir(model, cell, ext, rel_paths), pos_idx, neg_idx)
            if dst not in single:
                sdir = config.compressed_dir(dataset, res, budget, dst)
                single[dst] = (
                    eer_of(embed_dir(model, sdir, ext, rel_paths), pos_idx, neg_idx)
                    if sdir.is_dir()
                    else float("nan")
                )
            row = _row(dataset, res, budget, src, dst, eer, eer - single[dst])
            log.info(
                "  %-9s -> %-9s eer=%.4f single(%s)=%.4f delta=%+.4f",
                src,
                dst,
                eer,
                dst,
                single[dst],
                row["delta_eer_vs_singlepass"],
            )
            on_row(row)


def _row(dataset, res, budget, src, dst, eer, delta) -> dict:
    return {
        "dataset": dataset,
        "res": int(res),
        "budget": int(budget),
        "source_codec": src,
        "second_codec": dst,
        "eer": float(eer),
        "delta_eer_vs_singlepass": float(delta),
    }


# ----------------------------------------------------------------------- ours
def subject_spread_sample(
    rel_paths: list[str], n: int, max_per_subject: int
) -> np.ndarray:
    """Deterministic image sample spread over the subjects (sorted index ids).

    Subjects (folder names, sorted) are taken with a stride
    ``max(1, n_subjects * max_per_subject // n)``, up to ``max_per_subject`` crops
    each in index order, until ``n`` crops are collected; the remaining subjects top
    the sample up if the strided pass falls short.
    """
    subj = [str(Path(p).parent) for p in rel_paths]
    order: dict[str, list[int]] = {}
    for i, s in enumerate(subj):
        order.setdefault(s, []).append(i)
    subj_ids = sorted(order)
    stride = max(1, len(subj_ids) * max_per_subject // max(n, 1))
    chosen = subj_ids[::stride] or subj_ids
    picked: list[int] = []
    for s in chosen:
        picked.extend(order[s][:max_per_subject])
        if len(picked) >= n:
            break
    if len(picked) < n:
        seen = set(chosen)
        for s in subj_ids:
            if s in seen:
                continue
            picked.extend(order[s][:max_per_subject])
            if len(picked) >= n:
                break
    return np.array(sorted(set(picked[:n])))


def restrict_pairs(idx1, idx2, label, sample: np.ndarray, n_index: int):
    """Mated and non-mated pairs whose two members are both in ``sample``."""
    inside = np.zeros(n_index, bool)
    inside[sample] = True
    both = inside[idx1] & inside[idx2]
    pos = both & (label == 1)
    neg = both & (label == 0)
    return (idx1[pos], idx2[pos]), (idx1[neg], idx2[neg])


def ours_cells(sources: list[str], seconds: list[str]) -> list[tuple[str, str]]:
    """List the Ours-ACCURATE cells of a matrix, in the order of the paper CSVs.

    First every source into Ours-ACCURATE, then Ours-ACCURATE into every other second
    codec (both in matrix order).
    """
    srcs = sorted(set(sources), key=MATRIX.index)
    dsts = sorted(set(seconds), key=MATRIX.index)
    cells = [(s, OURS) for s in srcs] if OURS in dsts else []
    if OURS in srcs:
        cells += [(OURS, d) for d in dsts if d != OURS]
    return cells


def score_ours(
    dataset: str,
    model,
    rel_paths: list[str],
    sources: list[str],
    seconds: list[str],
    res: int,
    budget: int,
    n: int,
    max_per_subject: int,
    device: str,
    done: set,
    on_row,
) -> None:
    """Score the Ours-ACCURATE cells on a subject-spread sample of crops."""
    from PIL import Image  # noqa: PLC0415

    import face1kb  # noqa: PLC0415

    cells = [c for c in ours_cells(sources, seconds) if c not in done]
    if not cells:
        log.info("[%s %dB] Ours-ACCURATE cells already scored", dataset, budget)
        return
    n_all = len(rel_paths)
    sample = subject_spread_sample(rel_paths, n, max_per_subject)
    pos_idx, neg_idx = restrict_pairs(*verification.load_pairs(dataset), sample, n_all)
    log.info(
        "[%s %dB] Ours sample n=%d subjects=%d mated=%d non-mated=%d",
        dataset,
        budget,
        len(sample),
        len({str(Path(rel_paths[i]).parent) for i in sample}),
        pos_idx[0].size,
        neg_idx[0].size,
    )
    if pos_idx[0].size == 0 or neg_idx[0].size == 0:
        raise SystemExit("the Ours sample has no mated or no non-mated pair")
    codec = face1kb.load("accurate", device=device)

    def ours_rt(arr):
        return codec.decode(codec.encode(arr, budget, paper_compat=True))

    src_dir = config.aligned_dir(dataset, res)
    aligned = {}
    for i in sample:
        with Image.open(src_dir / rel_paths[i]) as im:
            aligned[i] = np.asarray(im.convert("RGB"), dtype=np.uint8)
    bins = config.compressed_dir(dataset, res, budget, OURS)

    def embed_eer(crops: dict[int, np.ndarray]) -> float:
        emb = np.full((n_all, getattr(model, "dim", EMB_DIM)), np.nan, np.float32)
        ids = sorted(crops)
        emb[ids] = np.asarray(
            model.embed([to_model_input(crops[i]) for i in ids]), dtype=np.float32
        )
        return eer_of(emb, pos_idx, neg_idx)

    needed_dst = sorted({d for _, d in cells}, key=MATRIX.index)
    needed_src = {s for s, _ in cells} | ({OURS} if OURS in needed_dst else set())
    t0 = time.perf_counter()
    first: dict[str, dict[int, np.ndarray]] = {}
    for src in sorted(needed_src, key=MATRIX.index):
        d = {}
        for i in sample:
            if src == OURS:
                binp = bins / Path(rel_paths[i]).with_suffix(config.OURS_EXT)
                d[i] = (
                    codec.decode(binp.read_bytes())
                    if binp.exists()
                    else ours_rt(aligned[i])
                )
            else:
                d[i] = roundtrip(aligned[i], src, budget)
        first[src] = d
    log.info("  first passes done in %.0f s", time.perf_counter() - t0)
    single: dict[str, float] = {}
    for dst in needed_dst:
        ref = (
            first[OURS]
            if dst == OURS
            else {i: roundtrip(aligned[i], dst, budget) for i in sample}
        )
        single[dst] = embed_eer(ref)
    log.info("  single-pass refs: %s", {k: round(v, 4) for k, v in single.items()})
    for src, dst in cells:
        tc = time.perf_counter()
        final = {
            i: ours_rt(first[src][i])
            if dst == OURS
            else roundtrip(first[src][i], dst, budget)
            for i in sample
        }
        eer = embed_eer(final)
        row = _row(dataset, res, budget, src, dst, eer, eer - single[dst])
        log.info(
            "  %-13s -> %-13s eer=%.4f single(%s)=%.4f delta=%+.4f [%.0f s]",
            src,
            dst,
            eer,
            dst,
            single[dst],
            row["delta_eer_vs_singlepass"],
            time.perf_counter() - tc,
        )
        on_row(row)


# ----------------------------------------------------------------------- output
def merge_rows(existing: pd.DataFrame | None, rows: list[dict]) -> pd.DataFrame:
    """Replace/insert ``rows`` by key and sort into matrix order."""
    new = pd.DataFrame(rows, columns=list(COLUMNS))
    if existing is not None and len(existing):
        keys_new = set(map(tuple, new[KEYS].astype(str).to_numpy()))
        keep = [tuple(k) not in keys_new for k in existing[KEYS].astype(str).to_numpy()]
        new = pd.concat([existing.loc[keep, list(COLUMNS)], new], ignore_index=True)
    rank = {c: i for i, c in enumerate(MATRIX)}
    ours_last = (new.source_codec == OURS) | (new.second_codec == OURS)
    new = new.assign(
        _o=ours_last.astype(int),
        _s=new.source_codec.map(rank).fillna(len(rank)),
        _d=new.second_codec.map(rank).fillna(len(rank)),
    )
    new = new.sort_values(["res", "budget", "_o", "_s", "_d"], kind="stable")
    return new.drop(columns=["_o", "_s", "_d"]).reset_index(drop=True)


class CsvWriter:
    """Merge rows into the output CSV as they are computed (resumable runs)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.df = pd.read_csv(self.path) if self.path.exists() else None

    def done(self, res: int, budget: int) -> set:
        if self.df is None:
            return set()
        d = self.df[(self.df.res == res) & (self.df.budget == budget)]
        return set(zip(d.source_codec, d.second_codec, strict=True))

    def add(self, row: dict) -> None:
        self.df = merge_rows(self.df, [row])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".part")
        self.df.to_csv(tmp, index=False)
        os.replace(tmp, self.path)


# ----------------------------------------------------------------------- main
def _check_codecs(names: list[str], allowed: tuple[str, ...], what: str) -> None:
    bad = [c for c in names if c not in allowed]
    if bad:
        raise SystemExit(f"unknown {what} codec(s) {bad}; choose from {list(allowed)}")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=config.DATASETS, default="colorferet")
    ap.add_argument("--res", type=int, default=112)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--sources", default=",".join(MATRIX), help="first-pass codecs")
    ap.add_argument("--seconds", default=",".join(MATRIX), help="second-pass codecs")
    ap.add_argument(
        "--stages", default=",".join(STAGES), help="comma list of produce,score,ours"
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="use only the first N index rows for the classical block (0 = all; "
        "the paper used 2500 on AI-Solutions-KK)",
    )
    ap.add_argument(
        "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 8) - 4))
    )
    ap.add_argument(
        "--strip-metadata",
        action="store_true",
        help="drop the decoded first pass's ICC profile / comment before the second "
        "pass (not the paper setting; use a separate --recompressed-root/--out-dir)",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help="scoring matcher")
    ap.add_argument("--device", default=None, help="torch device (default: cuda)")
    ap.add_argument("--sample-nonmated", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ours-n", type=int, default=120, help="Ours-ACCURATE sample")
    ap.add_argument("--ours-max-per-subject", type=int, default=10)
    ap.add_argument(
        "--recompressed-root",
        default=None,
        help="folder of the recompressed cells (default: WORK_ROOT/<ds>/recompressed)",
    )
    ap.add_argument(
        "--out-dir", default=None, help="default: OUTPUT_ROOT/recompression"
    )
    ap.add_argument(
        "--rescore", action="store_true", help="recompute cells already in the CSV"
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    stages = parse_list(args.stages)
    _check_codecs(stages, STAGES, "stage")
    sources, seconds = parse_list(args.sources), parse_list(args.seconds)
    _check_codecs(sources + seconds, MATRIX, "matrix")
    if args.recompressed_root:
        root = Path(args.recompressed_root)
    elif "produce" in stages and config.LAYOUT == "legacy":
        raise SystemExit(
            "FACE1KB_LAYOUT=legacy: pass --recompressed-root to write recompressed "
            "files outside the legacy store"
        )
    else:
        root = recompressed_root(args.dataset)
    out_dir = Path(args.out_dir) if args.out_dir else config.output_dir("recompression")

    index = config.read_index(args.dataset)
    rel_all = [str(p) for p in index.rel_path]
    rel_block = rel_all[: args.limit] if args.limit else rel_all
    cls_src = [c for c in sources if c != OURS]
    cls_dst = [c for c in seconds if c != OURS]

    if "produce" in stages and cls_src and cls_dst:
        produce(
            args.dataset,
            rel_block,
            cls_src,
            cls_dst,
            args.res,
            args.budget,
            root,
            args.workers,
            args.strip_metadata,
        )
    if not {"score", "ours"} & set(stages):
        return 0

    from face1kb import fr  # noqa: PLC0415

    device = args.device or "cuda"
    model = fr.load(args.model, device=device)
    writer = CsvWriter(output_csv(out_dir, args.dataset, args.budget, args.model))
    done = set() if args.rescore else writer.done(args.res, args.budget)
    if "score" in stages and cls_src and cls_dst:
        pairs = verification.pair_indices(args.dataset, args.sample_nonmated, args.seed)
        log.info(
            "[%s] mated=%d non-mated=%d",
            args.dataset,
            pairs[0][0].size,
            pairs[1][0].size,
        )
        # the single-pass references and all cells embed the full index; crops
        # beyond --limit are simply absent from the recompressed cells
        score_block(
            args.dataset,
            model,
            rel_all,
            cls_src,
            cls_dst,
            args.res,
            args.budget,
            root,
            pairs,
            done,
            writer.add,
        )
    if "ours" in stages and (OURS in sources or OURS in seconds):
        score_ours(
            args.dataset,
            model,
            rel_all,
            sources,
            seconds,
            args.res,
            args.budget,
            args.ours_n,
            args.ours_max_per_subject,
            device,
            done,
            writer.add,
        )
    log.info("wrote %s", writer.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
