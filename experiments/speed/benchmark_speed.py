# SPDX-License-Identifier: MIT
r"""Encode / decode speed of the codecs at their budget-fitting setting.

Metric: single-shot encode at the budget-hitting setting and single-shot decode, per
crop, with models and settings loaded beforehand (one untimed warm-up of each call
per crop). This is the per-call codec cost; the hard-budget search multiplies the
encode by its number of trial encodes (about 6 for the binary searches), which is not
included. Statistics per ``(codec, device, res, budget)``: median, IQR, mean and
standard deviation in ms, plus the median size of the timed streams.

Codecs and settings:

* classical codecs and JPEG-FzT (CPU): the setting is fitted once, on the first crop,
  by a linear scan that keeps the last setting whose output fits and stops at the
  first overflow; every crop is then timed at that setting. JPEG-FzT is timed as a
  whole: encode = F-transform downsampling plus one in-memory JPEG encode, decode =
  :func:`face1kb.baselines.jpeg_fzt.decode` (JPEG decode plus the inverse
  F-transform). The paper's table timed the JPEG-FzT decode without the inverse
  F-transform (see ``docs/reproduce_compress.md``).
* CompressAI (CPU or GPU): quality 1, the lowest rate point (the sub-1 kB regime;
  the realised size is reported).
* face1kb FAST / ACCURATE (CPU or GPU): the gain is the one the budget search
  selects on the first crop (``paper_compat=True``); encode = one
  ``net.compress`` at that gain on the padded crop, decode =
  :meth:`~face1kb.codec.api.Codec.decode_tensor` of the crop's own container. The
  rate index comes from the release encode path
  (:meth:`~face1kb.codec.api.Codec.encode`, whose ACCURATE side channel is
  computed on the unpadded crop, as in the stored bitstreams); the paper's benchmark
  passed a side channel of the padded crop to the budget search instead. The
  ACCURATE side-channel encode is not timed.

Crops: ``--n`` aligned Color FERET crops per resolution, stride-sampled from the
sorted file list. For the CPU rows pin the process to one core, e.g.
``taskset -c 0``; thread pools are limited to one thread. JPEG-AI is timed by
``benchmark_jpegai_speed.py``.

Output: ``OUTPUT_ROOT/quality/speed_benchmark.csv`` (or ``--out``); rows of other
``(codec, device, res, budget)`` keys already in the file are kept. The paper used
``speed_cpu_classical.csv``, ``speed_gpu_learned.csv`` and ``speed_cpu_learned.csv``.

Examples
--------
    taskset -c 0 python experiments/speed/benchmark_speed.py --device cpu \
        --codecs jpeg,webp,jpeg_xl,avif,heif,jpeg2000,jpeg_fzt --n 200 \
        --out outputs/quality/speed_cpu_classical.csv
    python experiments/speed/benchmark_speed.py --device gpu --gpu 0 \
        --codecs ours_fast,ours_accurate,neural_bmshj2018,neural_mbt2018_mean --n 60 \
        --out outputs/quality/speed_gpu_learned.csv
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    # one thread per library; must be set before numpy / torch are imported
    for _v in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(_v, "1")

import argparse
import csv
import io
import logging
import statistics as st
import time
from pathlib import Path

from face1kb import config

log = logging.getLogger("benchmark_speed")

CLASSICAL = ("jpeg", "webp", "jpeg_xl", "avif", "heif", "jpeg2000", "jpeg_fzt")
NEURAL = ("neural_bmshj2018", "neural_mbt2018_mean")
OURS = tuple(config.OURS_CODECS)
#: CSV columns (the schema of the shipped ``speed_*.csv``).
FIELDS = (
    "codec",
    "device",
    "res",
    "budget",
    "setting",
    "median_bytes",
    "enc_median_ms",
    "enc_iqr_ms",
    "enc_mean_ms",
    "enc_std_ms",
    "dec_median_ms",
    "dec_iqr_ms",
    "dec_mean_ms",
    "dec_std_ms",
    "n",
)
#: CompressAI quality timed (the lowest rate point).
NEURAL_QUALITY = 1


def crop_files(dataset: str, res: int, n: int) -> list[str]:
    """Up to ``n`` crop paths at ``res``, stride-sampled from the sorted listing."""
    files = sorted(str(p) for p in config.aligned_dir(dataset, res).rglob("*.png"))
    if len(files) > n:
        files = files[:: len(files) // n][:n]
    return files


def load_crops(dataset: str, res: int, n: int) -> list:
    """Load the crops of :func:`crop_files` as RGB PIL images."""
    from PIL import Image  # noqa: PLC0415

    out = []
    for f in crop_files(dataset, res, n):
        with Image.open(f) as im:
            out.append(im.convert("RGB"))
    return out


def stats(xs: list[float]) -> dict:
    """Median / IQR / mean / std of seconds, in ms (rounded to 4 decimals)."""
    xs = sorted(xs)
    if not xs:
        return {}
    q1 = xs[len(xs) // 4]
    q3 = xs[(3 * len(xs)) // 4]
    return {
        "median_ms": round(st.median(xs) * 1e3, 4),
        "iqr_ms": round((q3 - q1) * 1e3, 4),
        "mean_ms": round(st.fmean(xs) * 1e3, 4),
        "std_ms": round(st.pstdev(xs) * 1e3, 4) if len(xs) > 1 else 0.0,
        "n": len(xs),
    }


# ------------------------------------------------------------------ classical
def classical_encoder(codec: str, img):
    """``(encode(setting) -> bytes, settings)`` of a CPU codec for one image."""
    from face1kb.baselines import classical, jpeg_fzt  # noqa: PLC0415

    if codec == "jpeg_fzt":
        # the encode is the F-transform downsampling plus the JPEG encode
        return (lambda q: jpeg_fzt.encode_quality(jpeg_fzt.downsample(img), q)), list(
            jpeg_fzt.QUALITIES
        )
    return (lambda s: classical.encode_setting(img, codec, s)), classical.settings(
        codec
    )


def classical_decoder(codec: str):
    """Decode callable of a CPU codec (full JPEG-FzT decode).

    The Pillow codecs are timed as ``Image.open(...).load()``; their plugins are
    registered by the first encode (:func:`face1kb.baselines.classical.encode_setting`).
    """
    from PIL import Image  # noqa: PLC0415

    from face1kb.baselines import jpeg_fzt  # noqa: PLC0415

    if codec == "jpeg_fzt":
        return jpeg_fzt.decode
    return lambda data: Image.open(io.BytesIO(data)).load()


def fit_setting(encode, settings, budget: int):
    """Last setting whose output fits ``budget``, scanning up to the first overflow."""
    from face1kb.baselines import scan_fit, search  # noqa: PLC0415

    return scan_fit(
        search.sized(encode), settings, budget, keep="last", stop_at_overflow=True
    ).setting


def bench_classical(codec: str, crops: list, budget: int):
    """Time encode / decode of a CPU codec at the setting fitted on ``crops[0]``."""
    enc0, settings = classical_encoder(codec, crops[0])
    setting = fit_setting(enc0, settings, budget)
    decode = classical_decoder(codec)
    enc_t, dec_t, sizes = [], [], []
    for img in crops:
        enc, _ = classical_encoder(codec, img)
        enc(setting)  # warm-up
        t0 = time.perf_counter()
        data = enc(setting)
        t1 = time.perf_counter()
        decode(data)  # warm-up
        t2 = time.perf_counter()
        decode(data)
        t3 = time.perf_counter()
        enc_t.append(t1 - t0)
        dec_t.append(t3 - t2)
        sizes.append(len(data))
    return enc_t, dec_t, sizes, setting


# ------------------------------------------------------------------ CompressAI
def bench_neural(codec: str, crops: list, device: str):
    """Time ``compress`` / ``decompress`` of a CompressAI model at quality 1."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from face1kb.baselines import compressai_codecs as cai  # noqa: PLC0415

    net = cai.get_net(codec, NEURAL_QUALITY, device)
    cuda = str(device).startswith("cuda")
    enc_t, dec_t, sizes = [], [], []
    for img in crops:
        x, _ = cai.to_tensor(np.asarray(img, dtype=np.uint8), device)
        with torch.no_grad():
            out = net.compress(x)  # warm-up
            _sync(cuda)
            t0 = time.perf_counter()
            out = net.compress(x)
            _sync(cuda)
            t1 = time.perf_counter()
            net.decompress(out["strings"], out["shape"])  # warm-up
            _sync(cuda)
            t2 = time.perf_counter()
            net.decompress(out["strings"], out["shape"])
            _sync(cuda)
            t3 = time.perf_counter()
        enc_t.append(t1 - t0)
        dec_t.append(t3 - t2)
        sizes.append(cai.entropy_size(out["strings"]))
    return enc_t, dec_t, sizes, NEURAL_QUALITY


def _sync(cuda: bool) -> None:
    if cuda:
        import torch  # noqa: PLC0415

        torch.cuda.synchronize()


# ------------------------------------------------------------------ face1kb
def bench_ours(codec: str, crops: list, budget: int, device: str):
    """Time the face1kb codec at the gain its budget search picks on ``crops[0]``."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    import face1kb  # noqa: PLC0415
    from face1kb.codec.budget import gain_table  # noqa: PLC0415
    from face1kb.codec.resize import pad_to_multiple  # noqa: PLC0415

    model = face1kb.load(config.OURS_CODECS[codec], device=device)
    net = model.net
    table = gain_table(net)
    cuda = str(device).startswith("cuda")

    def padded(img):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(model.device)
        return pad_to_multiple(x)[0]

    def container(img):
        return model.encode(np.asarray(img, dtype=np.uint8), budget, paper_compat=True)

    _, info0 = model.encode(
        np.asarray(crops[0], dtype=np.uint8),
        budget,
        paper_compat=True,
        return_info=True,
    )
    rate_index = int(info0["rate_index"])
    gain = table[rate_index]
    enc_t, dec_t, sizes = [], [], []
    for img in crops:
        xp = padded(img)
        with torch.no_grad():
            net.compress(xp, inputscale=gain, res=img.size[0])  # warm-up
            _sync(cuda)
            t0 = time.perf_counter()
            net.compress(xp, inputscale=gain, res=img.size[0])
            _sync(cuda)
            t1 = time.perf_counter()
            packed = container(img)  # a real container for the decode timing
            model.decode_tensor(packed)  # warm-up
            _sync(cuda)
            t2 = time.perf_counter()
            model.decode_tensor(packed)
            _sync(cuda)
            t3 = time.perf_counter()
        enc_t.append(t1 - t0)
        dec_t.append(t3 - t2)
        sizes.append(len(packed))
    return enc_t, dec_t, sizes, rate_index


# ------------------------------------------------------------------ output
def merge_rows(path: Path, rows: list[dict]) -> list[dict]:
    """Rows of ``path`` whose key is not measured again, followed by ``rows``."""
    if not path.exists():
        return rows
    new = {(r["codec"], r["device"], str(r["res"]), str(r["budget"])) for r in rows}
    with open(path, newline="") as f:
        old = [
            r
            for r in csv.DictReader(f)
            if (r["codec"], r["device"], r["res"], r["budget"]) not in new
        ]
    return old + rows


def row_for(codec, device, res, budget, enc, dec, sizes, setting) -> dict:
    """CSV row of one measured cell."""
    es, ds = stats(enc), stats(dec)
    return {
        "codec": codec,
        "device": device,
        "res": res,
        "budget": budget,
        "setting": setting,
        "median_bytes": int(st.median(sizes)) if sizes else -1,
        "enc_median_ms": es.get("median_ms"),
        "enc_iqr_ms": es.get("iqr_ms"),
        "enc_mean_ms": es.get("mean_ms"),
        "enc_std_ms": es.get("std_ms"),
        "dec_median_ms": ds.get("median_ms"),
        "dec_iqr_ms": ds.get("iqr_ms"),
        "dec_mean_ms": ds.get("mean_ms"),
        "dec_std_ms": ds.get("std_ms"),
        "n": es.get("n"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES (gpu)")
    ap.add_argument("--codecs", default=",".join(CLASSICAL))
    ap.add_argument("--resolutions", default=",".join(map(str, config.RESOLUTIONS)))
    ap.add_argument("--budgets", default=",".join(map(str, config.BUDGETS)))
    ap.add_argument("--n", type=int, default=200, help="crops per resolution")
    ap.add_argument("--dataset", choices=config.DATASETS, default="colorferet")
    ap.add_argument("--out", default=None, help="CSV (default: quality/speed_...)")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if args.device == "gpu":
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        import torch  # noqa: PLC0415

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    dev_label = "gpu" if device.startswith("cuda") else "cpu"
    out = (
        Path(args.out)
        if args.out
        else config.output_dir("quality") / "speed_benchmark.csv"
    )
    codecs = [c for c in args.codecs.split(",") if c]
    unknown = [c for c in codecs if c not in CLASSICAL + NEURAL + OURS]
    if unknown:
        raise SystemExit(f"unknown codec(s) {unknown}")
    resolutions = [int(r) for r in args.resolutions.split(",") if r]
    budgets = [int(b) for b in args.budgets.split(",") if b]

    rows = []
    for codec in codecs:
        for res in resolutions:
            crops = load_crops(args.dataset, res, args.n)
            if not crops:
                log.warning("no %d px crops found; skipped", res)
                continue
            for budget in budgets:
                try:
                    if codec in CLASSICAL:
                        r = bench_classical(codec, crops, budget)
                    elif codec in NEURAL:
                        r = bench_neural(codec, crops, device)
                    else:
                        r = bench_ours(codec, crops, budget, device)
                except Exception as exc:  # noqa: BLE001 - keep the other cells
                    log.error(
                        "%s %d %d: %s: %s", codec, res, budget, type(exc).__name__, exc
                    )
                    continue
                row = row_for(codec, dev_label, res, budget, *r)
                rows.append(row)
                log.info(
                    "%-20s %s %3dpx %4dB  enc=%sms dec=%sms (n=%s, %s B, setting %s)",
                    codec,
                    dev_label,
                    res,
                    budget,
                    row["enc_median_ms"],
                    row["dec_median_ms"],
                    row["n"],
                    row["median_bytes"],
                    row["setting"],
                )
    if not rows:
        log.error("nothing measured")
        return 1
    all_rows = merge_rows(out, rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS), extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    log.info("wrote %s (%d rows)", out, len(all_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
