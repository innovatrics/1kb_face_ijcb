# SPDX-License-Identifier: MIT
r"""Small-sample rate--identity comparison of the codecs (the id-cos of Section 5.1).

A seeded sample of ``--n`` aligned crops per dataset (64, ``random.Random(0).sample``
over the sorted ``aligned_<res>/*/*.png`` files) is compressed on the fly to each
budget (1024 and 512 B) with

* the classical codecs JPEG, WebP, AVIF, HEIF, JPEG XL and JPEG 2000: a linear scan
  of the benchmark grid keeping the largest file that fits (settings that fail to
  encode are skipped; if nothing fits, the smallest setting);
* JPEG-FzT: a scan of the even JPEG qualities 2..94 that stops at the first file
  above the budget;
* JPEG-AI (reference software, HOP profile): the analytic bpp target with up to two
  step-down re-encodes (:func:`face1kb.baselines.jpeg_ai.step_down_fit`); skipped
  with ``--no-jpegai`` or when the reference software is not installed;
* face1kb-FAST and face1kb-ACCURATE (``paper_compat=True``).

``--codecs`` restricts the run to some of these (names as in the output); the
sample does not depend on the selection, so a partial run gives the same rows as
the corresponding rows of a full run. With ``--codecs`` an existing
``comparison.json`` is updated in place: only the recomputed (dataset, codec,
budget) rows are replaced or added, every other row is kept. A run without
``--codecs`` rewrites the file.

Each reconstruction is compared with its crop:

* ``id_cos`` -- cosine similarity of the two embeddings of the identity model
  ``--id-model`` at its 112 px input (crops of other sizes are resized with an
  antialiased bilinear filter and rounded to uint8);
* ``psnr``, ``ssim`` (``pytorch_msssim``, 11 px Gaussian window) and ``lpips``
  (AlexNet) at the native resolution -- diagnostics only; the tables take their
  fidelity columns from the full quality measurement;
* ``bytes`` -- the emitted size.

The medians per (dataset, codec, budget) are written to
``<out-dir>/comparison.json`` (default ``$FACE1KB_OUTPUT_ROOT/codec_comparison/
res<res>``), a list of ``{dataset, codec, budget, n, id_cos, psnr, ssim, lpips,
bytes, id_model}``.

**Identity model.** The paper's id-cos values were measured with a proprietary face
matcher that is not distributed, so they cannot be regenerated; the shipped
``results/codec_comparison/res*/comparison.json`` files are their only source. The
default here is the public held-out matcher ``cvlface_ir101``. Any model known to
:mod:`face1kb.fr` can be named, including one registered by a plugin
(``FACE1KB_FR_PLUGINS``), and ``--id-onnx PATH`` registers an ONNX model on the fly.
Numbers from a different matcher are not comparable with the published id-cos.

Needs a CUDA GPU for the face1kb codecs and JPEG-AI.

Examples
--------
    python experiments/codec_comparison/evaluate.py --res 112
    python experiments/codec_comparison/evaluate.py --res 224 --no-jpegai \
        --id-onnx my_matcher.onnx --id-onnx-color BGR
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np

import face1kb
from face1kb import config, fr
from face1kb.baselines import classical, jpeg_ai, jpeg_fzt
from face1kb.codec.resize import resize112
from face1kb.data.cli_utils import parse_list, setup_logging
from face1kb.eval.verification import paired_cosines

log = logging.getLogger("codec_comparison.evaluate")

#: Classical codecs in evaluation order.
CLASSICAL = ("jpeg", "webp", "avif", "heif", "jpeg_xl", "jpeg2000")
#: Every codec name ``--codecs`` accepts, in output order.
CODECS = (*CLASSICAL, "jpeg_fzt", "jpeg_ai", "ours_fast", "ours_accurate")


def _read(path: Path) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def to01(img):
    """``HxWx3`` uint8 -> ``3xHxW`` float32 tensor in ``[0, 1]``."""
    import torch  # noqa: PLC0415

    return torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)


class Encoders:
    """Encode + decode one crop with every codec: ``name -> (img01, bytes)``."""

    def __init__(
        self,
        device: str,
        ours: list[str],
        use_jpegai: bool,
        codecs: set[str] | None = None,
    ):
        self.device = device
        self.codecs = codecs
        self.ours = {
            f"ours_{v}": face1kb.load(v, device=device)
            for v in ours
            if self._on(f"ours_{v}")
        }
        self.use_jpegai = use_jpegai and self._on("jpeg_ai")

    def _on(self, name: str) -> bool:
        """Whether ``name`` is selected (every codec when no selection is given)."""
        return self.codecs is None or name in self.codecs

    def classical(self, img, codec: str, budget: int):
        data, info = classical.encode_to_budget(
            img, codec, budget, search="scan", keep="largest", skip_errors=True
        )
        return to01(classical.decode(data)), info["size"]

    def fzt(self, img, budget: int):
        data, info = jpeg_fzt.encode_to_budget(
            img,
            budget,
            qualities=jpeg_fzt.QUALITIES_EVEN,
            search="scan",
            keep="largest",
            stop_at_overflow=True,
        )
        return to01(jpeg_fzt.decode(data, img.shape[0])), info["size"]

    def jpegai(self, img, budget: int):
        def enc(bpp):
            data = jpeg_ai.encode_bpp(img, bpp)
            return None if data is None else (len(data), data)

        fit = jpeg_ai.step_down_fit(enc, budget, img.shape[0] * img.shape[1])
        if fit is None:
            return None, 0
        dec = jpeg_ai.decode(fit.payload)
        return (to01(dec) if dec is not None else None), fit.size

    def face1kb(self, name: str, img, budget: int):
        codec = self.ours[name]
        data, info = codec.encode(img, budget, paper_compat=True, return_info=True)
        xhat, h = codec.decode_tensor(data)
        if xhat is None:  # identity-only container: black frame
            import torch  # noqa: PLC0415

            return torch.zeros(3, h.res, h.res), info["bytes"]
        return xhat[0, :, : h.res, : h.res].float().cpu(), info["bytes"]

    def all(self, img, budget: int, first: bool) -> dict:
        outs = {c: self.classical(img, c, budget) for c in CLASSICAL if self._on(c)}
        if self._on("jpeg_fzt"):
            outs["jpeg_fzt"] = self.fzt(img, budget)
        if self.use_jpegai:
            try:
                outs["jpeg_ai"] = self.jpegai(img, budget)
            except Exception as exc:  # noqa: BLE001 - environment: skip the codec
                outs["jpeg_ai"] = (None, 0)
                if first:
                    log.warning("  jpeg_ai skipped: %s", type(exc).__name__)
        for name in self.ours:
            outs[name] = self.face1kb(name, img, budget)
        return outs


class Metrics:
    """id-cos at the 112 px matcher input plus native-resolution diagnostics."""

    def __init__(self, embedder, device: str):
        import lpips  # noqa: PLC0415

        self.embedder = embedder
        self.device = device
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @staticmethod
    def _u8(x) -> np.ndarray:
        """``1x3xHxW`` float in ``[0, 1]`` -> ``HxWx3`` uint8."""
        arr = x[0].permute(1, 2, 0).cpu().numpy()
        return (arr * 255.0).round().clip(0, 255).astype(np.uint8)

    def __call__(self, clean01, dec01) -> dict:
        import torch  # noqa: PLC0415
        from pytorch_msssim import ssim as ssim_fn  # noqa: PLC0415

        c = clean01.unsqueeze(0).to(self.device).clamp(0, 1)
        d = dec01.unsqueeze(0).to(self.device).clamp(0, 1)
        mse = torch.mean((c - d) ** 2).item()
        psnr = 99.0 if mse < 1e-9 else -10 * np.log10(mse)
        s = ssim_fn(c, d, data_range=1.0).item()
        with torch.no_grad():
            lp = self.lpips(c * 2 - 1, d * 2 - 1).item()
            c112, d112 = resize112(c).clamp(0, 1), resize112(d).clamp(0, 1)
        emb = self.embedder.embed(np.stack([self._u8(c112), self._u8(d112)]))
        cos = float(paired_cosines(emb[:1], emb[1:])[0])
        return {"id_cos": cos, "psnr": psnr, "ssim": s, "lpips": lp}


def sample_crops(dataset: str, res: int, n: int, seed: int) -> list[Path]:
    """``random.Random(seed).sample`` of the sorted aligned crop files."""
    root = config.aligned_dir(dataset, res)
    files = sorted(root.glob("*/*.png"))
    if not files:
        raise SystemExit(f"no aligned crops under {root} (set FACE1KB_DATA_ROOT)")
    return random.Random(seed).sample(files, min(n, len(files)))


def run(args) -> list[dict]:
    """Evaluate every (dataset, budget, crop, codec); return the median rows."""
    import torch  # noqa: PLC0415

    if args.id_onnx:
        fr.register_onnx(
            args.id_model,
            args.id_onnx,
            color=args.id_onnx_color,
            normalize=args.id_onnx_normalize,
            overwrite=True,
        )
    embedder = fr.load(args.id_model, device=args.device)
    use_jpegai = not args.no_jpegai
    if use_jpegai and not jpeg_ai.available():
        log.warning("JPEG-AI reference software not installed: JPEG-AI skipped")
        use_jpegai = False
    codecs = set(parse_list(args.codecs)) if args.codecs else None
    if codecs is not None:
        unknown = codecs - set(CODECS)
        if unknown:
            raise SystemExit(f"unknown codec(s) {sorted(unknown)}; pick from {CODECS}")
    enc = Encoders(args.device, parse_list(args.ours), use_jpegai, codecs)
    metrics = Metrics(embedder, args.device)
    rows = []
    for ds in parse_list(args.datasets):
        crops = sample_crops(ds, args.res, args.n, args.seed)
        log.info("[%s] %d crops", ds, len(crops))
        for budget in parse_list(args.budgets, int):
            agg: dict[str, list[dict]] = {}
            for k, fp in enumerate(crops):
                img = _read(fp)
                clean01 = to01(img)
                for c, (dec01, nb) in enc.all(img, budget, first=k == 0).items():
                    if dec01 is None:
                        continue
                    m = metrics(clean01, dec01)
                    m["bytes"] = nb
                    agg.setdefault(c, []).append(m)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if (k + 1) % 16 == 0:
                    log.info("  B=%d: %d/%d crops", budget, k + 1, len(crops))
            for c, ms in agg.items():
                med = {key: float(np.median([x[key] for x in ms])) for key in ms[0]}
                rows.append(
                    {
                        "dataset": ds,
                        "codec": c,
                        "budget": budget,
                        "n": len(ms),
                        **med,
                        "id_model": args.id_model,
                    }
                )
                log.info(
                    "  B=%d %-14s id_cos=%.3f psnr=%.1f ssim=%.3f lpips=%.3f "
                    "bytes=%.0f",
                    budget,
                    c,
                    med["id_cos"],
                    med["psnr"],
                    med["ssim"],
                    med["lpips"],
                    med["bytes"],
                )
    return rows


def _key(row: dict) -> tuple:
    return (row["dataset"], row["codec"], int(row["budget"]))


def merge_rows(existing: list[dict], new: list[dict]) -> list[dict]:
    """Replace the (dataset, codec, budget) rows of ``existing`` found in ``new``.

    Rows of ``existing`` without a counterpart in ``new`` are kept in place; the
    remaining rows of ``new`` are appended in their order.
    """
    fresh = {_key(r): r for r in new}
    out = [fresh.pop(_key(r), r) for r in existing]
    return out + [r for r in new if _key(r) in fresh]


def main(argv: list[str] | None = None) -> int:
    """Run the comparison and write comparison.json."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--res", type=int, default=112, help="crop resolution (112, 224)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets", default="colorferet,kk")
    ap.add_argument("--budgets", default="1024,512")
    ap.add_argument("--ours", default="fast,accurate", help="face1kb variants")
    ap.add_argument(
        "--codecs",
        default=None,
        help="comma list of codecs to evaluate (default: all; e.g. jpeg_ai,ours_fast)",
    )
    ap.add_argument("--no-jpegai", action="store_true")
    ap.add_argument("--id-model", default=fr.IDCOS_DEFAULT)
    ap.add_argument(
        "--id-onnx",
        type=Path,
        default=None,
        help="register this ONNX file as --id-model",
    )
    ap.add_argument("--id-onnx-color", choices=("RGB", "BGR"), default="RGB")
    ap.add_argument(
        "--id-onnx-normalize", default="[-1,1]", choices=("[-1,1]", "[0,1]", "none")
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    setup_logging()

    rows = run(args)
    out_dir = args.out_dir or config.output_dir("codec_comparison", f"res{args.res}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "comparison.json"
    if args.codecs and out.exists():
        existing = json.loads(out.read_text())
        models = {r.get("id_model") for r in existing} - {args.id_model}
        if models:
            log.warning(
                "%s has rows of identity model(s) %s; merging rows of %s",
                out,
                sorted(map(str, models)),
                args.id_model,
            )
        log.info("updating %d of %d existing rows", len(rows), len(existing))
        rows = merge_rows(existing, rows)
    out.write_text(json.dumps(rows, indent=1))
    log.info(
        "wrote %s (%d rows; id model %s)",
        out_dir / "comparison.json",
        len(rows),
        args.id_model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
