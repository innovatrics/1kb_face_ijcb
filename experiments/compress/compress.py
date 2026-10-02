# SPDX-License-Identifier: MIT
r"""Compress aligned crops to a byte budget with the baselines and the face1kb codecs.

Every crop of the dataset index (``index.csv``, in its canonical order) is encoded
with each requested codec at each ``(resolution, budget)`` cell and stored as

    WORK_ROOT/<dataset>/compressed/<res>[<suffix>]px_<budget>B/<codec>/<rel_stem>.<ext>

Codecs and their budget search (all from :mod:`face1kb.baselines` and
:func:`face1kb.load`):

* ``jpeg``, ``jpeg2000``, ``webp``, ``jpeg_xl``, ``avif``, ``heif``: binary search
  over the quality (JPEG 2000: compression ratio) grid; CPU.
* ``jpeg_fzt``: binary search over the JPEG quality 1, 3, ..., 95; CPU.
* ``jpeg_ai``: analytic target-bpp fit (HOP profile, tools off); GPU. A crop for
  which the reference software produces no plausible bitstream is recorded as
  ``failed`` and no file is written.
* ``neural_bmshj2018``, ``neural_mbt2018_mean``: binary search over the CompressAI
  quality 1..8 on the entropy-coded size; GPU. The 768 B and 960 B cells use cuDNN's
  deterministic algorithms (``--compressai-deterministic auto``), as the paper's
  streams of those cells were encoded that way.
* ``ours_fast``, ``ours_accurate``: :meth:`face1kb.codec.api.Codec.encode` with
  ``paper_compat=True`` (the paper's budget accounting and over-budget fallback);
  GPU.

When nothing fits a budget, the smallest-output setting is stored anyway (the
paper's "failure to compress"); budget compliance is always judged from the size of
the stored file.

Runs are resumable: an existing output file is never re-encoded (unless
``--overwrite``); its row is recorded from the on-disk size with status
``existing``. ``--shard I:N`` processes the index rows ``I, I+N, I+2N, ...`` so that
independent processes can share a grid; ``--offset``/``--limit`` then select a
contiguous slice of those rows (the paper ran CompressAI with ``--limit 300``, the
first 300 index rows). Files are written atomically (temporary name, then rename).

Each run writes one manifest (Parquet) to
``WORK_ROOT/<dataset>/compressed/manifest/<tag>.parquet`` with one row per task:
``dataset, image, subject, rel_path, pose, codec, res, suffix, budget, setting,
search_bytes, bytes, fitted, search_fitted, encode_ms, status, error``. ``bytes`` is
the size of the stored file and ``fitted`` is ``bytes <= budget``; ``search_bytes``
and ``search_fitted`` are what the codec's search compared against the budget (the
CompressAI search counts the entropy-coded bytes, not the ``.ptci`` container).
``status`` is ``encoded``, ``existing``, ``missing`` (no source crop), ``failed``
(JPEG-AI produced no plausible stream) or ``error``.

CPU codecs run in a process pool (``--workers``); GPU codecs run sequentially in
the main process on the device selected with ``--gpu`` / ``CUDA_VISIBLE_DEVICES``.
Byte-identical reproduction of the paper streams needs the pinned codec libraries
(``requirements/paper.txt``) and, for the GPU codecs, the same GPU type and software
stack (see ``docs/reproduce_compress.md``).

Examples
--------
    # classical codecs + JPEG-FzT, full Color FERET grid, 16 CPU workers
    python experiments/compress/compress.py --dataset colorferet --codecs cpu \
        --workers 16
    # the face1kb codecs on one GPU, shard 0 of 4
    python experiments/compress/compress.py --dataset kk --codecs ours --gpu 0 \
        --shard 0:4
    # CompressAI on the first 300 crops (paper scope)
    python experiments/compress/compress.py --dataset colorferet \
        --codecs neural_bmshj2018,neural_mbt2018_mean --limit 300 --gpu 0
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from face1kb import config

log = logging.getLogger("compress")

#: Codec groups accepted by ``--codecs``.
CPU_GROUP = ("jpeg", "jpeg2000", "webp", "jpeg_xl", "avif", "heif", "jpeg_fzt")
COMPRESSAI = ("neural_bmshj2018", "neural_mbt2018_mean")
OURS = tuple(config.OURS_CODECS)
GPU_GROUP = ("jpeg_ai", *COMPRESSAI, *OURS)
GROUPS = {
    "cpu": CPU_GROUP,
    "classical": CPU_GROUP,
    "gpu": GPU_GROUP,
    "compressai": COMPRESSAI,
    "ours": OURS,
    "baselines": CPU_GROUP + ("jpeg_ai",) + COMPRESSAI,
    "all": CPU_GROUP + GPU_GROUP,
}
#: Budgets whose CompressAI streams the paper encoded with deterministic cuDNN.
COMPRESSAI_DETERMINISTIC_BUDGETS = (768, 960)
#: Manifest columns.
COLUMNS = (
    "dataset",
    "image",
    "subject",
    "rel_path",
    "pose",
    "codec",
    "res",
    "suffix",
    "budget",
    "setting",
    "search_bytes",
    "bytes",
    "fitted",
    "search_fitted",
    "encode_ms",
    "status",
    "error",
)


# ----------------------------------------------------------------- task building
def parse_codecs(spec: str) -> list[str]:
    """Expand a comma-separated codec list (names or :data:`GROUPS`), keep order."""
    from face1kb.baselines import ALL_EXTENSIONS  # noqa: PLC0415

    out: list[str] = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        names = GROUPS.get(tok, (tok,))
        for name in names:
            if name not in ALL_EXTENSIONS:
                raise SystemExit(
                    f"unknown codec {name!r}; choose from {sorted(ALL_EXTENSIONS)} "
                    f"or a group {sorted(GROUPS)}"
                )
            if name not in out:
                out.append(name)
    return out


def select_rows(rows: list, shard: tuple[int, int], offset: int, limit: int) -> list:
    """Rows ``rows[i::n]`` of shard ``(i, n)``, then ``[offset:offset + limit]``.

    ``limit=0`` keeps every row after ``offset``.
    """
    idx, count = shard
    if count < 1 or not 0 <= idx < count:
        raise ValueError(f"invalid shard {idx}:{count}")
    rows = rows[idx::count][offset:]
    return rows[:limit] if limit else rows


def build_tasks(
    dataset: str,
    index_rows: list[dict],
    codecs: list[str],
    resolutions: list[int],
    budgets: list[int],
    suffix: str = "",
) -> list[tuple]:
    """Cartesian ``(row x codec x res x budget)`` tasks.

    A task is ``(dataset, rel_path, subject, pose, codec, res, budget, suffix)``;
    ``rel_path`` is the crop path relative to the aligned folder, and the output file
    is ``<rel_path without extension>.<codec extension>`` inside the cell folder.
    ``subject`` is the crop's folder name (the index's ``subject`` column may be
    numeric).
    """
    tasks = []
    for r in index_rows:
        subject = Path(r["rel_path"]).parent.as_posix()
        for codec in codecs:
            for res in resolutions:
                for budget in budgets:
                    tasks.append(
                        (
                            dataset,
                            r["rel_path"],
                            subject,
                            r.get("pose", ""),
                            codec,
                            int(res),
                            int(budget),
                            suffix,
                        )
                    )
    return tasks


def output_path(task: tuple) -> Path:
    """Return the stored bitstream path of a task."""
    from face1kb.baselines import extension  # noqa: PLC0415

    dataset, rel_path, _s, _p, codec, res, budget, suffix = task
    cell = config.compressed_dir(dataset, res, budget, codec, suffix)
    return cell / Path(rel_path).with_suffix(extension(codec))


def source_path(task: tuple) -> Path:
    """Return the aligned crop of a task."""
    dataset, rel_path, _s, _p, _c, res, _b, suffix = task
    return config.aligned_dir(dataset, res, suffix) / Path(rel_path).with_suffix(".png")


def manifest_dir(dataset: str) -> Path:
    """Folder of the compression manifests of ``dataset``."""
    return config.work_dir(dataset) / "compressed" / "manifest"


def manifest_tag(
    codecs: list[str],
    resolutions: list[int],
    budgets: list[int],
    suffix: str = "",
    shard: tuple[int, int] = (0, 1),
    offset: int = 0,
    limit: int = 0,
) -> str:
    """Manifest file stem, e.g. ``jpeg_webp_112_1024_512``.

    ``_sh<I>of<N>`` is appended for a shard and ``_o<offset>n<limit>`` for a row
    slice, so that runs over different rows write different manifests.
    """
    tag = (
        "_".join(codecs)[:40]
        + "_"
        + "_".join(map(str, resolutions))
        + suffix
        + "_"
        + "_".join(map(str, budgets))
    )
    if shard[1] > 1:
        tag += f"_sh{shard[0]}of{shard[1]}"
    if offset or limit:
        tag += f"_o{offset}n{limit}"
    return tag


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temporary file in the same folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".part{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ----------------------------------------------------------------- encoding
_OURS_CODECS: dict = {}


def _ours_codec(codec: str, device: str):
    """Load the face1kb codec of ``ours_fast`` / ``ours_accurate`` once per process."""
    key = (codec, device)
    if key not in _OURS_CODECS:
        import face1kb  # noqa: PLC0415

        _OURS_CODECS[key] = face1kb.load(config.OURS_CODECS[codec], device=device)
    return _OURS_CODECS[key]


def encode_crop(
    codec: str, src: Path, budget: int, device: str, deterministic: bool
) -> tuple[bytes, dict]:
    """Encode one crop file; return ``(data, info)`` (info: setting, size, fitted)."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    with Image.open(src) as im:
        img = im.convert("RGB")
    if codec in OURS:
        data, info = _ours_codec(codec, device).encode(
            np.asarray(img, dtype=np.uint8),
            budget,
            paper_compat=True,
            return_info=True,
        )
        return data, {
            "setting": int(info["rate_index"]),
            "size": int(info["bytes"]),
            "fitted": bool(info["fitted"]),
        }
    from face1kb.baselines import get_codec  # noqa: PLC0415

    entry = get_codec(codec)
    if entry.family == "compressai":
        return entry.encode_to_budget(
            np.asarray(img, dtype=np.uint8),
            budget,
            device=device,
            deterministic=deterministic,
        )
    # Classical codecs, JPEG-FzT and JPEG-AI take the PIL image as the paper did.
    return entry.encode_to_budget(img, budget)


def process_task(task: tuple, opts: dict) -> dict:
    """Compress one ``(crop, codec, res, budget)`` task and return its manifest row."""
    dataset, rel_path, subject, pose, codec, res, budget, suffix = task
    row = {
        "dataset": dataset,
        "image": Path(rel_path).stem,
        "subject": subject,
        "rel_path": rel_path,
        "pose": pose,
        "codec": codec,
        "res": res,
        "suffix": suffix,
        "budget": budget,
        "setting": float("nan"),
        "search_bytes": -1,
        "bytes": -1,
        "fitted": False,
        "search_fitted": False,
        "encode_ms": float("nan"),
        "status": "",
        "error": "",
    }
    out = output_path(task)
    if out.exists() and not opts["overwrite"]:
        size = out.stat().st_size
        row.update(bytes=size, fitted=size <= budget, status="existing")
        return row
    src = source_path(task)
    if not src.is_file():
        row.update(status="missing", error=f"no source crop {src.name}")
        return row
    deterministic = {
        "on": True,
        "off": False,
        "auto": budget in COMPRESSAI_DETERMINISTIC_BUDGETS,
    }[opts["compressai_deterministic"]]
    try:
        t0 = time.perf_counter()
        data, info = encode_crop(codec, src, budget, opts["device"], deterministic)
        ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001 - one bad crop must not stop the grid
        from face1kb.baselines.jpeg_ai import JpegAIError  # noqa: PLC0415

        status = "failed" if isinstance(exc, JpegAIError) else "error"
        row.update(status=status, error=f"{type(exc).__name__}: {exc}"[:300])
        return row
    if not data:
        row.update(status="failed", error="empty bitstream")
        return row
    write_atomic(out, data)
    row.update(
        setting=float(info["setting"]),
        search_bytes=int(info["size"]),
        bytes=len(data),
        fitted=len(data) <= budget,
        search_fitted=bool(info["fitted"]),
        encode_ms=round(ms, 2),
        status="encoded",
    )
    return row


_POOL_OPTS: dict = {}


def _pool_init(opts: dict) -> None:
    _POOL_OPTS.update(opts)


def _pool_task(task: tuple) -> dict:
    return process_task(task, _POOL_OPTS)


def run_tasks(tasks: list[tuple], opts: dict, workers: int) -> list[dict]:
    """Run CPU tasks in a process pool and GPU tasks sequentially; return the rows."""
    from tqdm import tqdm  # noqa: PLC0415

    cpu = [t for t in tasks if t[4] in CPU_GROUP]
    gpu = [t for t in tasks if t[4] not in CPU_GROUP]
    rows: list[dict] = []
    if cpu:
        # CPU work first: the pool forks before any CUDA context exists.
        if workers > 1:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_pool_init, initargs=(opts,)
            ) as ex:
                rows += list(
                    tqdm(
                        ex.map(_pool_task, cpu, chunksize=32),
                        total=len(cpu),
                        unit="task",
                        desc="cpu",
                    )
                )
        else:
            rows += [process_task(t, opts) for t in tqdm(cpu, unit="task", desc="cpu")]
    if gpu:
        rows += [process_task(t, opts) for t in tqdm(gpu, unit="task", desc="gpu")]
    return rows


def summarize(df) -> str:
    """Per-codec summary (counts by status, median bytes, on-disk fit rate)."""
    lines = []
    for (codec, res, budget), g in df.groupby(["codec", "res", "budget"], sort=False):
        ok = g[g.bytes >= 0]
        counts = ", ".join(f"{k}={v}" for k, v in g.status.value_counts().items())
        med = int(ok.bytes.median()) if len(ok) else -1
        fit = 100.0 * ok.fitted.mean() if len(ok) else float("nan")
        lines.append(
            f"  {codec:20s} {res:4d}px {budget:5d}B  median={med:5d} B  "
            f"fit={fit:6.2f}%  [{counts}]"
        )
    return "\n".join(lines)


def _guard_legacy_store() -> None:
    """Refuse to write into a legacy data store (outputs must go elsewhere)."""
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
    ap.add_argument("--dataset", choices=config.DATASETS, default="colorferet")
    ap.add_argument(
        "--codecs",
        default="cpu",
        help=f"comma-separated codec names or groups {sorted(GROUPS)}",
    )
    ap.add_argument("--resolutions", default=",".join(map(str, config.RESOLUTIONS)))
    ap.add_argument("--budgets", default=",".join(map(str, config.BUDGETS)))
    ap.add_argument(
        "--align-suffix",
        "--suffix",
        dest="align_suffix",
        default="",
        help="crop-set suffix, e.g. _tight reads aligned_112_tight and writes "
        "112_tightpx_<budget>B cells",
    )
    ap.add_argument("--shard", default="0:1", help="I:N -- index rows I, I+N, ...")
    ap.add_argument("--offset", type=int, default=0, help="skip rows of the shard")
    ap.add_argument("--limit", type=int, default=0, help="at most N rows (0 = all)")
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="processes for the CPU codecs",
    )
    ap.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES")
    ap.add_argument("--device", default="cuda", help="torch device of the GPU codecs")
    ap.add_argument(
        "--compressai-deterministic",
        choices=("auto", "on", "off"),
        default="auto",
        help="cuDNN deterministic algorithms for CompressAI (auto: only for the "
        f"{COMPRESSAI_DETERMINISTIC_BUDGETS} B budgets)",
    )
    ap.add_argument("--overwrite", action="store_true", help="re-encode existing files")
    ap.add_argument("--work-root", default=None, help="override FACE1KB_WORK_ROOT")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.work_root:
        os.environ["FACE1KB_WORK_ROOT"] = args.work_root  # for worker processes
        config.WORK_ROOT = Path(args.work_root)
    _guard_legacy_store()

    import pandas as pd  # noqa: PLC0415

    codecs = parse_codecs(args.codecs)
    resolutions = [int(r) for r in args.resolutions.split(",") if r]
    budgets = [int(b) for b in args.budgets.split(",") if b]
    shard = tuple(int(v) for v in args.shard.split(":"))
    index = config.read_index(args.dataset)
    rows = select_rows(index.to_dict("records"), shard, args.offset, args.limit)
    tasks = build_tasks(
        args.dataset, rows, codecs, resolutions, budgets, args.align_suffix
    )
    log.info(
        "%s: %d of %d index rows x codecs %s x res %s x budgets %s -> %d tasks",
        args.dataset,
        len(rows),
        len(index),
        codecs,
        resolutions,
        budgets,
        len(tasks),
    )
    opts = {
        "device": args.device,
        "overwrite": args.overwrite,
        "compressai_deterministic": args.compressai_deterministic,
    }
    records = run_tasks(tasks, opts, args.workers)
    if not records:
        log.info("nothing to do")
        return 0
    df = pd.DataFrame(records, columns=list(COLUMNS))
    out = manifest_dir(args.dataset) / (
        manifest_tag(
            codecs,
            resolutions,
            budgets,
            args.align_suffix,
            shard,
            args.offset,
            args.limit,
        )
        + ".parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("wrote %d rows -> %s\n%s", len(df), out, summarize(df))
    bad = df.status.isin(["error"]).sum()
    if bad:
        log.warning("%d task(s) raised an error (see the manifest 'error' column)", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
