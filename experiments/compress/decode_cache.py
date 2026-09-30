# SPDX-License-Identifier: MIT
r"""Decode stored bitstreams once into a PNG cache.

The embedding and quality scripts decode every stored crop once per face matcher.
For the classical codecs a decode costs nothing, but a JPEG-AI decode takes about
0.5 s and a face1kb decode a GPU forward pass, and the 14-matcher roster would pay
that for each matcher. This script decodes each bitstream exactly once into

    WORK_ROOT/<dataset>/decoded/<res>[<suffix>]px_<budget>B/<codec>/<rel_stem>.png

(:func:`face1kb.config.decoded_dir`; ``FACE1KB_LAYOUT=legacy`` uses the sibling
``<codec>_png`` folders). :func:`face1kb.baselines.decode_file` with ``cache=True``
reads a cached PNG when it exists and decodes the bitstream otherwise, so the cache
changes run time, never results: a missing or deleted cache only makes consumers
slower.

Decoding goes through :func:`face1kb.baselines.decode_file` (JPEG-AI with the HOP
profile; face1kb containers with ``(x * 255).round()`` to ``uint8``). Existing PNGs
are skipped (resumable) and every PNG is written to a temporary name and renamed. The
bitstreams are distributed round-robin over ``--gpus``, one process per GPU; each
process loads its decoders once. A malformed JPEG-AI stream is rejected by the
decoder's structure check and counted as failed. Each decode is bounded by a
``--timeout`` (default 60 s) enforced with ``SIGALRM``, so a stream that sends the
reference decoder into a loop costs one file, not the whole run; a timed-out file
counts as failed. The timeout is Unix-only (on platforms without ``SIGALRM`` decodes
are unbounded).

Examples
--------
    python experiments/compress/decode_cache.py --datasets colorferet,kk --gpus 0,1
    python experiments/compress/decode_cache.py --datasets kk --codecs ours_fast \
        --resolutions 224 --check
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import signal
from pathlib import Path

from face1kb import config

log = logging.getLogger("decode_cache")

#: Codecs whose decode is slow enough to cache (the benchmark's cached codecs).
DEFAULT_CODECS = ("jpeg_ai", "ours_fast", "ours_accurate")

#: Per-file decode timeout in seconds (``SIGALRM``; Unix only).
PER_FILE_TIMEOUT_S = 60


class _DecodeTimeout(Exception):
    """Raised by the ``SIGALRM`` handler when one decode exceeds the timeout."""


def _on_alarm(signum, frame):  # noqa: ARG001 - signal handler signature
    raise _DecodeTimeout


def cell_bitstreams(
    dataset: str, res: int, budget: int, codec: str, suffix: str = ""
) -> list[Path]:
    """Sorted bitstream files of one cell (empty if the cell does not exist)."""
    from face1kb.baselines import extension  # noqa: PLC0415

    root = config.compressed_dir(dataset, res, budget, codec, suffix)
    if not root.is_dir():
        return []
    return sorted(root.rglob(f"*{extension(codec)}"))


def plan(datasets, codecs, resolutions, budgets, suffix="") -> list[tuple]:
    """``(dataset, res, budget, codec, files, todo)`` for every existing cell."""
    from face1kb.baselines import decoded_cache_path  # noqa: PLC0415

    cells = []
    for ds in datasets:
        for codec in codecs:
            for res in resolutions:
                for budget in budgets:
                    files = cell_bitstreams(ds, res, budget, codec, suffix)
                    if not files:
                        continue
                    todo = [f for f in files if not decoded_cache_path(f).exists()]
                    cells.append((ds, res, budget, codec, files, todo))
    return cells


def _worker(
    gpu: str | None,
    files: list[str],
    env: dict,
    timeout_s: int = PER_FILE_TIMEOUT_S,
) -> tuple[int, int]:
    """Decode ``files`` on one GPU; return ``(n_written, n_failed)``.

    Each decode is bounded by ``timeout_s`` seconds via ``SIGALRM`` where the
    platform has it (``timeout_s <= 0`` disables the bound).
    """
    os.environ.update(env)
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from face1kb.baselines import decode_file, decoded_cache_path  # noqa: PLC0415

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    use_alarm = timeout_s > 0 and hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _on_alarm)
    done = failed = 0
    for k, f in enumerate(files):
        src = Path(f)
        dst = decoded_cache_path(src)
        if dst.exists():
            continue
        try:
            if use_alarm:
                signal.alarm(timeout_s)
            try:
                arr = decode_file(src, device="cuda")
            finally:
                if use_alarm:
                    signal.alarm(0)
        except _DecodeTimeout:
            log.warning("decode of %s timed out after %d s", src, timeout_s)
            failed += 1
            continue
        except Exception as exc:  # noqa: BLE001 - one bad stream must not stop the run
            log.warning("decode of %s failed: %s", src, exc)
            failed += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + f".part{os.getpid()}")
        Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(tmp, format="PNG")
        os.replace(tmp, dst)
        done += 1
        if (k + 1) % 1000 == 0:
            log.info("[gpu %s] %d/%d", gpu, k + 1, len(files))
    return done, failed


def _guard_legacy_store() -> None:
    if config.LAYOUT == "legacy" and (
        Path(config.WORK_ROOT).resolve() == Path(config.DATA_ROOT).resolve()
    ):
        raise SystemExit(
            "FACE1KB_LAYOUT=legacy with FACE1KB_WORK_ROOT == FACE1KB_DATA_ROOT: "
            "refusing to write into the original data store; pass --work-root"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument("--codecs", default=",".join(DEFAULT_CODECS))
    ap.add_argument("--resolutions", default=",".join(map(str, config.RESOLUTIONS)))
    ap.add_argument("--budgets", default=",".join(map(str, config.BUDGETS)))
    ap.add_argument("--align-suffix", default="")
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids")
    ap.add_argument(
        "--timeout",
        type=int,
        default=PER_FILE_TIMEOUT_S,
        help="per-file decode timeout in s (SIGALRM, Unix only; 0 disables)",
    )
    ap.add_argument("--check", action="store_true", help="report coverage only")
    ap.add_argument("--work-root", default=None, help="override FACE1KB_WORK_ROOT")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    env = {}
    if args.work_root:
        env["FACE1KB_WORK_ROOT"] = args.work_root
        config.WORK_ROOT = Path(args.work_root)

    datasets = [d for d in args.datasets.split(",") if d]
    codecs = [c for c in args.codecs.split(",") if c]
    resolutions = [int(r) for r in args.resolutions.split(",") if r]
    budgets = [int(b) for b in args.budgets.split(",") if b]
    cells = plan(datasets, codecs, resolutions, budgets, args.align_suffix)
    todo: list[str] = []
    for ds, res, budget, codec, files, missing in cells:
        log.info(
            "%-10s %-14s %3d%spx %4dB  cached %d/%d",
            ds,
            codec,
            res,
            args.align_suffix,
            budget,
            len(files) - len(missing),
            len(files),
        )
        todo += [str(f) for f in missing]
    if args.check or not todo:
        log.info("%d file(s) to decode", len(todo))
        return 0
    _guard_legacy_store()
    gpus = [g for g in args.gpus.split(",") if g] or [None]
    shards = [todo[i :: len(gpus)] for i in range(len(gpus))]
    log.info("decoding %d file(s) on GPU(s) %s", len(todo), gpus)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(gpus)) as pool:
        results = pool.starmap(
            _worker, [(g, s, env, args.timeout) for g, s in zip(gpus, shards)]
        )
    done = sum(r[0] for r in results)
    failed = sum(r[1] for r in results)
    log.info("done: %d written, %d failed", done, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
