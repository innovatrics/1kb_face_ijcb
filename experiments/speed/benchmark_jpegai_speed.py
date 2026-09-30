# SPDX-License-Identifier: MIT
r"""JPEG-AI encode / decode speed per resolution, on the CPU or one GPU.

Times the JPEG-AI reference software (HOP profile, tools off; see
:mod:`face1kb.baselines.jpeg_ai`) on ``--n`` aligned Color FERET crops per
resolution, stride-sampled from the sorted file list, and writes one row per
resolution in the schema of ``benchmark_speed.py``, plus ``loadavg`` (the 1-minute
load average during the run) and ``device_name``.

* Rate setting: target bpp x 100 = ``round(1024 * 8 / px * 100)`` capped to 2..65,
  i.e. 65 at 64-112 px, 29 at 168 px and 16 at 224 px. The cap of 65 is above the
  benchmark's fit range (2..50); the realised size is reported per row (e.g. about
  1130 B at 112 px).
* Encode time: one call of the reference encoder on the crop's PNG
  (:func:`face1kb.baselines.jpeg_ai.encode_file`, which also writes the
  reconstruction); decode time: :func:`face1kb.baselines.jpeg_ai.decode` of the
  bitstream. The models are loaded and one crop is coded before timing.
* ``--device cpu`` hides the GPUs (``CUDA_VISIBLE_DEVICES=""``) before torch is
  imported; pin the process to one core with ``taskset -c 0`` for the paper's
  single-core rows. The run refuses to start when the load average exceeds
  :data:`MAX_LOADAVG_PER_CORE` per core (``--allow-load`` overrides).

Output: ``OUTPUT_ROOT/quality/speed_<device>_jpegai.csv`` (or ``--out``).

Examples
--------
    taskset -c 0 python experiments/speed/benchmark_jpegai_speed.py --device cpu --n 10
    python experiments/speed/benchmark_jpegai_speed.py --device gpu --gpu 0 \
        --resolutions 112 --n 38
"""

from __future__ import annotations

import os

if __name__ == "__main__":
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
import logging
import statistics as st
import tempfile
import time
from pathlib import Path

from face1kb import config

log = logging.getLogger("benchmark_jpegai_speed")

#: Budget the rate setting is derived from.
BUDGET = 1024
#: Upper limit of the target bpp x 100.
MAX_BPP_M100 = 65
#: Refuse to measure above this 1-minute load average per core.
MAX_LOADAVG_PER_CORE = 0.35
FIELDS = (
    "codec",
    "device",
    "device_name",
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
    "loadavg",
)


def bpp_setting(res: int) -> int:
    """Target bpp x 100 of a ``res`` px crop: the analytic 1024 B value, 2..65."""
    return int(max(2, min(MAX_BPP_M100, round(BUDGET * 8 / (res * res) * 100))))


def crop_files(dataset: str, res: int, n: int) -> list[str]:
    """Up to ``n`` crop paths at ``res``, stride-sampled from the sorted listing."""
    files = sorted(str(p) for p in config.aligned_dir(dataset, res).rglob("*.png"))
    if len(files) > n:
        files = files[:: len(files) // n][:n]
    return files


def stats_ms(xs: list[float], prefix: str) -> dict:
    """Median / IQR / mean / std of millisecond values under ``prefix``."""
    if not xs:
        return {f"{prefix}_{k}_ms": None for k in ("median", "iqr", "mean", "std")}
    xs = sorted(xs)
    q1 = xs[len(xs) // 4]
    q3 = xs[(3 * len(xs)) // 4] if len(xs) >= 4 else xs[-1]
    return {
        f"{prefix}_median_ms": round(st.median(xs), 4),
        f"{prefix}_iqr_ms": round(q3 - q1, 4),
        f"{prefix}_mean_ms": round(st.fmean(xs), 4),
        f"{prefix}_std_ms": round(st.pstdev(xs) if len(xs) > 1 else 0.0, 4),
    }


def device_name(device: str) -> str:
    """``cpu`` or ``gpu:<CUDA device name>``."""
    if device != "gpu":
        return device
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return f"gpu:{torch.cuda.get_device_name(0)}"
    except Exception:  # noqa: BLE001 - the name is informational
        pass
    return device


def bench_res(dataset: str, res: int, n: int, device: str) -> dict | None:
    """Time encode and decode of ``n`` crops at ``res``; return one CSV row."""
    from PIL import Image  # noqa: PLC0415

    from face1kb.baselines import jpeg_ai  # noqa: PLC0415

    files = crop_files(dataset, res, n)
    if not files:
        log.warning("res=%d: no crops found, skipped", res)
        return None
    setting = bpp_setting(res)
    enc_ms: list[float] = []
    dec_ms: list[float] = []
    sizes: list[int] = []
    with tempfile.TemporaryDirectory(prefix=f"jaispeed_{res}_") as td:
        tmp = Path(td)

        def save(i, f):
            png = tmp / f"{i}.png"
            with Image.open(f) as im:
                im.convert("RGB").save(png)
            return png

        # warm-up: loads the encoder and decoder models
        warm = save("warm", files[0])
        if jpeg_ai.encode_file(warm, tmp / "warm.bin", setting, rec_path=tmp / "w.png"):
            jpeg_ai.decode((tmp / "warm.bin").read_bytes())
        for i, f in enumerate(files):
            png, bits, rec = save(i, f), tmp / f"{i}.bin", tmp / f"{i}_rec.png"
            t0 = time.perf_counter()
            ok = jpeg_ai.encode_file(png, bits, setting, rec_path=rec)
            t1 = time.perf_counter()
            if not ok or not bits.exists():
                continue
            enc_ms.append((t1 - t0) * 1e3)
            data = bits.read_bytes()
            sizes.append(len(data))
            t0 = time.perf_counter()
            out = jpeg_ai.decode(data)
            t1 = time.perf_counter()
            if out is not None:
                dec_ms.append((t1 - t0) * 1e3)
    row = {
        "codec": "jpeg_ai",
        "device": device,
        "device_name": device_name(device),
        "res": res,
        "budget": BUDGET,
        "setting": setting,
        "median_bytes": int(st.median(sizes)) if sizes else None,
        "n": len(enc_ms),
        "loadavg": round(os.getloadavg()[0], 1),
        **stats_ms(enc_ms, "enc"),
        **stats_ms(dec_ms, "dec"),
    }
    log.info(
        "res=%3d setting=%3d n=%3d enc=%s ms dec=%s ms bytes=%s",
        res,
        setting,
        row["n"],
        row["enc_median_ms"],
        row["dec_median_ms"],
        row["median_bytes"],
    )
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--device", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--gpu", default="0", help="GPU id with --device gpu")
    ap.add_argument("--resolutions", default=",".join(map(str, config.RESOLUTIONS)))
    ap.add_argument("--n", type=int, default=10, help="crops timed per resolution")
    ap.add_argument("--dataset", choices=config.DATASETS, default="colorferet")
    ap.add_argument("--out", default=None, help="output CSV")
    ap.add_argument(
        "--allow-load",
        action="store_true",
        help="measure on a busy host (timings are then not comparable)",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    load, cores = os.getloadavg()[0], os.cpu_count() or 1
    if load > MAX_LOADAVG_PER_CORE * cores and not args.allow_load:
        log.error(
            "load average %.1f on %d cores exceeds %.2f per core; wait for a quiet "
            "host or pass --allow-load",
            load,
            cores,
            MAX_LOADAVG_PER_CORE,
        )
        return 2
    # must be set before the reference software imports torch
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.device == "cpu" else str(args.gpu)
    out = (
        Path(args.out)
        if args.out
        else config.output_dir("quality") / f"speed_{args.device}_jpegai.csv"
    )
    rows = [
        r
        for res in (int(x) for x in args.resolutions.split(",") if x)
        if (r := bench_res(args.dataset, res, args.n, args.device)) is not None
    ]
    if not rows:
        log.error("no rows measured")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s (%d rows)", out, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
