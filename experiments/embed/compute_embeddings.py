# SPDX-License-Identifier: MIT
r"""Embed every aligned and compressed crop set with the face-recognition roster.

For each dataset, source and model this writes one ``(N, 512)`` float32 array of raw
embeddings, row ``i`` belonging to image id ``i`` of the dataset index
(``config.read_index``); a row of NaN marks a crop that is missing or could not be
decoded. A *source* is one crop folder found on disk:

* aligned: ``config.aligned_dir(dataset, res, suffix)/<rel_path>.png``, tag
  ``aligned_<res><suffix>``;
* compressed: ``config.compressed_dir(dataset, res, budget, codec, suffix)/<rel_stem>
  .<ext>``, tag ``<codec>_<res><suffix>_<budget>``.

Arrays go to ``config.embeddings_path(dataset, model, tag)`` and are listed in
``config.embeddings_manifest(dataset)`` (rebuilt from the arrays on disk at the end of
every run, :func:`face1kb.eval.embeddings.scan_manifest`).

Decoding and embedding
----------------------
* Bitstreams are decoded with :func:`face1kb.baselines.decode_file` (JPEG-FzT at the
  source resolution, CompressAI and face1kb containers on CUDA). A decoded PNG cache
  (``config.decoded_dir``) is used when present: for JPEG-AI and the face1kb codecs
  the reconstruction is then read instead of decoded.
* A JPEG-AI source is embedded only from its decoded cache. When the cache folder
  exists, a bitstream without a cached PNG counts as a missing crop (the reference
  decoder failed on it). A source without a cache folder is skipped unless
  ``--jpegai-live`` is given (the reference decoder costs about 0.5 s per crop and
  would run once per model).
* Crops that are not 112 px are resized bilinearly to 112 px
  (:func:`face1kb.fr.embedders.resize_batch`) and embedded with
  ``face1kb.fr.load(model).embed``. Image ids are processed in chunks of ``--batch``
  (default 256); the decodable crops of one chunk form one forward batch. The
  published arrays were computed this way, but the chunk size of a stored array
  (256, 128 or 64) varies by matcher and cell; the matching ``--batch`` reproduces an
  array bit-exactly (e.g. ``--batch 64`` for TopoFR-R100 on the Color FERET 96 px
  JPEG-XL/AVIF cells, while LVFace-S Color FERET 96 px arrays match at the default
  256). Other chunk sizes change the embeddings by float noise only.

Execution
---------
Each model runs in its own spawned worker process (one model per process, pinned to
one of ``--gpus``), which loads the model once and embeds all of its pending sources,
cheapest decode first. Existing arrays are skipped (``--overwrite`` recomputes them),
arrays are written atomically, so an interrupted run resumes where it stopped. Model
weights are fetched in the parent process before the workers start
(``--offline`` requires them to be present). User-registered evaluators
(``FACE1KB_FR_PLUGINS``, see ``docs/models.md``) are accepted by ``--models``.

Examples
--------
    # the core grid (5 resolutions x 2 budgets, 12 codecs) for all 14 matchers
    python experiments/embed/compute_embeddings.py --models all --gpus 0,1
    # the four anchors on the preprocessing and crop-tightness variants
    python experiments/embed/compute_embeddings.py --models anchor --resolutions 112 \
        --suffixes _A1,_A2,_A3,_A4,_B1,_B2,_C1,_C2,_tight,_mid,_fill
    # list what would be computed
    python experiments/embed/compute_embeddings.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from face1kb import config

log = logging.getLogger("compute_embeddings")

#: Width of the stored embeddings.
EMB_DIM = 512
#: Input size of every roster model.
MODEL_INPUT = 112
#: Codecs whose files decode with Pillow (thread-safe, no GPU state).
PILLOW_CODECS = ("jpeg", "jpeg2000", "webp", "jpeg_xl", "avif", "heif")
#: Codecs whose decoded PNG cache replaces decoding when present.
CACHED_CODECS = ("jpeg_ai", "ours_fast", "ours_accurate")


# ----------------------------------------------------------------- sources
def discover_sources(datasets, kinds, resolutions, suffixes, codecs, budgets) -> list:
    """List every requested source whose crop folder exists on disk."""
    from face1kb import baselines  # noqa: PLC0415

    sources = []
    for ds in datasets:
        for res in resolutions:
            for suffix in suffixes:
                if "aligned" in kinds:
                    d = config.aligned_dir(ds, res, suffix)
                    if d.is_dir():
                        sources.append(
                            {
                                "dataset": ds,
                                "kind": "aligned",
                                "res": res,
                                "suffix": suffix,
                                "codec": None,
                                "budget": None,
                                "tag": config.embedding_tag(res, suffix),
                                "dir": str(d),
                                "ext": ".png",
                                "cache": None,
                            }
                        )
                if "compressed" not in kinds:
                    continue
                for codec in codecs:
                    for budget in budgets:
                        d = config.compressed_dir(ds, res, budget, codec, suffix)
                        if not d.is_dir():
                            continue
                        cache = config.decoded_dir(ds, res, budget, codec, suffix)
                        sources.append(
                            {
                                "dataset": ds,
                                "kind": "compressed",
                                "res": res,
                                "suffix": suffix,
                                "codec": codec,
                                "budget": budget,
                                "tag": config.embedding_tag(res, suffix, codec, budget),
                                "dir": str(d),
                                "ext": baselines.extension(codec),
                                "cache": str(cache) if cache.is_dir() else None,
                            }
                        )
    return sources


def decode_cost(source: dict) -> int:
    """Rank sources by decode cost (0 = aligned PNG ... 4 = CompressAI)."""
    codec = source["codec"]
    if not codec:
        return 0
    if codec in CACHED_CODECS and source["cache"]:
        return 1
    if codec == "jpeg_ai":
        return 5
    if codec == "jpeg_fzt":
        return 2
    if codec.startswith("ours"):
        return 3
    if codec.startswith("neural"):
        return 4
    return 1


def parallel_reads(source: dict) -> bool:
    """Whether the files of a source may be read by several threads at once."""
    codec = source["codec"]
    return codec is None or codec in PILLOW_CODECS or bool(source["cache"])


# ------------------------------------------------------------------ worker
class _Loader:
    """Read and decode the crop of one image id of a source (``None`` if missing)."""

    def __init__(self, source: dict, rel_paths):
        self.src = source
        self.dir = Path(source["dir"])
        self.rel_paths = rel_paths
        self.gpu_lock = threading.Lock()  # serialises live GPU decodes
        self.count_lock = threading.Lock()
        self.failed = 0  # files that exist but raised while decoding

    def _decode(self, path: Path):
        from face1kb import baselines  # noqa: PLC0415

        codec, res = self.src["codec"], self.src["res"]
        if codec is None or codec in PILLOW_CODECS:
            return baselines.decode_file(path)
        if codec in CACHED_CODECS and self.src["cache"]:
            cached = baselines.decoded_cache_path(path)
            if cached is not None and cached.is_file():
                return baselines.decode_file(cached)
            if codec == "jpeg_ai":
                return None  # the cache has no reconstruction of this bitstream
        with self.gpu_lock:
            return baselines.decode_file(path, res=res, device="cuda")

    def __call__(self, i: int):
        from face1kb.fr.embedders import resize_batch  # noqa: PLC0415

        path = self.dir / Path(self.rel_paths[i]).with_suffix(self.src["ext"])
        if not path.is_file():
            return i, None
        try:
            arr = self._decode(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            log.debug("decode failed for %s: %s", path, exc)
            with self.count_lock:
                self.failed += 1
            arr = None
        if arr is None:
            return i, None
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        return i, resize_batch(arr[None], MODEL_INPUT)[0]


def embed_source(model, source, rel_paths, batch, io_workers, limit):
    """Embed one source with a loaded model; return ``(array, n_ok, n_missing)``.

    ``limit`` > 0 embeds only the first ``limit`` image ids; the other rows stay NaN.
    Files that exist but fail to decode count as missing and are reported in a
    warning.
    """
    n_all = len(rel_paths)
    out = np.full((n_all, EMB_DIM), np.nan, dtype=np.float32)
    n = min(n_all, limit) if limit else n_all
    load = _Loader(source, rel_paths)
    threads = io_workers if parallel_reads(source) else 1
    pool = ThreadPoolExecutor(max_workers=threads) if threads > 1 else None
    n_miss = 0
    try:
        for start in range(0, n, batch):
            ids = range(start, min(start + batch, n))
            loaded = pool.map(load, ids) if pool else map(load, ids)
            arrs, ok_ids = [], []
            for i, arr in loaded:
                if arr is None:
                    n_miss += 1
                else:
                    arrs.append(arr)
                    ok_ids.append(i)
            if arrs:
                out[ok_ids] = np.asarray(model.embed(arrs), dtype=np.float32)
    finally:
        if pool:
            pool.shutdown()
    if load.failed:
        log.warning(
            "%s/%s: %d files could not be decoded (rows left NaN)",
            source["dataset"],
            source["tag"],
            load.failed,
        )
    return out, n - n_miss, n_miss


def atomic_save(path: Path, arr: np.ndarray) -> None:
    """Write ``arr`` as ``.npy`` through a temporary file and a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def run_model(model_name, gpu, sources, opts) -> list[dict]:
    """Worker entry point: pin the GPU, load one model, embed its pending sources."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # before torch is imported
    logging.basicConfig(level=opts["log_level"], format="%(message)s")
    from face1kb import fr  # noqa: PLC0415

    todo = [
        s
        for s in sources
        if opts["overwrite"]
        or not config.embeddings_path(s["dataset"], model_name, s["tag"]).exists()
    ]
    todo.sort(key=decode_cost)
    if not todo:
        return []
    model = fr.load(model_name, device="cuda", download=False)
    index_cache: dict = {}
    done = []
    for s in todo:
        ds = s["dataset"]
        if ds not in index_cache:
            idx = config.read_index(ds).sort_values("id").reset_index(drop=True)
            index_cache[ds] = idx["rel_path"].to_numpy()
        out, n_ok, n_miss = embed_source(
            model,
            s,
            index_cache[ds],
            opts["batch"],
            opts["io_workers"],
            opts["limit"],
        )
        atomic_save(config.embeddings_path(ds, model_name, s["tag"]), out)
        done.append({"dataset": ds, "tag": s["tag"]})
        log.info(
            "[%s @ gpu %s] %s/%s: %d ok, %d missing",
            model_name,
            gpu,
            ds,
            s["tag"],
            n_ok,
            n_miss,
        )
    return done


# -------------------------------------------------------------------- CLI
def resolve_models(spec: str) -> list[str]:
    """``all`` (the 14-model roster), ``anchor`` (4 anchors) or a comma list."""
    from face1kb import fr  # noqa: PLC0415

    if spec == "all":
        return list(fr.ROSTER)
    if spec == "anchor":
        return list(fr.ANCHORS)
    names = [m.strip() for m in spec.split(",") if m.strip()]
    known = set(fr.available())
    bad = [m for m in names if m not in known]
    if bad:
        raise SystemExit(f"unknown model(s) {bad}; choose from {sorted(known)}")
    return names


def fetch_models(models, offline: bool) -> None:
    """Make sure the weights of the built-in models are present before spawning."""
    from face1kb import fr  # noqa: PLC0415

    for m in models:
        spec = fr.REGISTRY.get(m)
        if spec is None or not spec.builtin or fr.present(m):
            continue
        if offline:
            raise SystemExit(
                f"weights of {m} are missing and --offline is set; run "
                "scripts/fetch_models.py first"
            )
        fr.fetch(m)


def rebuild_manifest(datasets) -> None:
    """Rewrite ``embeddings/manifest.csv`` of each dataset from the arrays on disk."""
    from face1kb.eval.embeddings import scan_manifest, write_manifest  # noqa: PLC0415

    for ds in datasets:
        df = scan_manifest(ds)
        if len(df):
            path = write_manifest(df, ds, merge=False)
            log.info("%s: manifest rebuilt -> %s (%d arrays)", ds, path, len(df))


def _csv(spec: str, cast=str) -> list:
    return [cast(x.strip()) for x in spec.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    from face1kb.eval.embeddings import CODECS  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument("--models", default="all", help="'all', 'anchor' or a comma list")
    ap.add_argument("--kind", default="both", choices=["aligned", "compressed", "both"])
    ap.add_argument("--resolutions", default=",".join(map(str, config.RESOLUTIONS)))
    ap.add_argument("--codecs", default=",".join(CODECS))
    ap.add_argument("--budgets", default=",".join(map(str, config.BUDGETS)))
    ap.add_argument(
        "--suffixes",
        default="",
        help="comma list of crop-variant suffixes, e.g. '_tight,_A1'; an empty item "
        "selects the base crops (default: base crops only)",
    )
    ap.add_argument("--gpus", default="0", help="comma list of GPU ids")
    ap.add_argument("--workers", type=int, default=0, help="0: one per GPU")
    ap.add_argument("--batch", type=int, default=256, help="image ids per forward")
    ap.add_argument("--io-workers", type=int, default=16, help="read threads/worker")
    ap.add_argument(
        "--limit", type=int, default=0, help="embed only the first N ids (0: all)"
    )
    ap.add_argument("--overwrite", action="store_true", help="recompute done arrays")
    ap.add_argument(
        "--jpegai-live",
        action="store_true",
        help="decode JPEG-AI sources without a decoded cache with the reference "
        "software (slow)",
    )
    ap.add_argument("--offline", action="store_true", help="never download weights")
    ap.add_argument("--dry-run", action="store_true", help="list the plan only")
    ap.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="only rewrite embeddings/manifest.csv from the arrays on disk",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - CLI entry point
    """Discover the crop sets and embed them with the requested models."""
    args = build_parser().parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")
    datasets = _csv(args.datasets)
    if args.rebuild_manifest:
        rebuild_manifest(datasets)
        return 0
    kinds = ["aligned", "compressed"] if args.kind == "both" else [args.kind]
    suffixes = [s.strip() for s in args.suffixes.split(",")] or [""]
    gpus = _csv(args.gpus)
    models = resolve_models(args.models)
    sources = discover_sources(
        datasets,
        kinds,
        _csv(args.resolutions, int),
        suffixes,
        _csv(args.codecs),
        _csv(args.budgets, int),
    )
    skipped = [
        s
        for s in sources
        if s["codec"] == "jpeg_ai" and not s["cache"] and not args.jpegai_live
    ]
    for s in skipped:
        log.warning(
            "skip %s/%s: no decoded JPEG-AI cache under %s (decode the cell first, "
            "or pass --jpegai-live)",
            s["dataset"],
            s["tag"],
            config.decoded_dir(
                s["dataset"], s["res"], s["budget"], "jpeg_ai", s["suffix"]
            ),
        )
    sources = [s for s in sources if s not in skipped]
    log.info("%d models x %d sources (%s)", len(models), len(sources), datasets)

    tasks = []
    for k, model in enumerate(models):
        todo = [
            s
            for s in sources
            if args.overwrite
            or not config.embeddings_path(s["dataset"], model, s["tag"]).exists()
        ]
        if todo:
            tasks.append((model, gpus[k % len(gpus)], todo))
    log.info(
        "%d arrays to compute in %d model tasks on GPUs %s",
        sum(len(t) for _, _, t in tasks),
        len(tasks),
        gpus,
    )
    if args.dry_run or not tasks:
        for model, gpu, todo in tasks:
            log.info("  %s @ gpu %s: %d sources", model, gpu, len(todo))
            if args.verbose:
                for s in todo:
                    log.info("    %s/%s", s["dataset"], s["tag"])
        log.info("dry run: nothing computed" if args.dry_run else "nothing to do")
        return 0

    fetch_models([m for m, _, _ in tasks], args.offline)
    opts = {
        "batch": args.batch,
        "io_workers": args.io_workers,
        "limit": args.limit,
        "overwrite": args.overwrite,
        "log_level": level,
    }
    workers = min(args.workers or len(gpus), len(tasks))
    failed = []
    n_done = 0
    # spawn: a clean CUDA context per worker, and one model per process (the two
    # CVLface snapshots cannot share a process). Each model task gets its own
    # single-worker process pool (ProcessPoolExecutor's max_tasks_per_child needs
    # Python 3.11); ``workers`` threads bound how many run at once.
    ctx = mp.get_context("spawn")

    def run_isolated(model: str, gpu: str, todo: list, opts: dict) -> list:
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as proc:
            return proc.submit(run_model, model, gpu, todo, opts).result()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_isolated, m, g, t, opts): m for m, g, t in tasks}
        for fut in as_completed(futs):
            model = futs[fut]
            try:
                n_done += len(fut.result())
            except Exception as exc:  # noqa: BLE001 - one model must not stop the rest
                log.error("model %s failed: %s: %s", model, type(exc).__name__, exc)
                failed.append(model)
    log.info("computed %d arrays", n_done)
    rebuild_manifest(datasets)
    if failed:
        log.error("failed models: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
