# SPDX-License-Identifier: MIT
"""Per-image compression difficulty: which crops lose the most identity.

For a random sample of aligned crops of each dataset, every crop is compressed to
each byte budget with every codec, decoded and scored per image:

* ``id_cos`` -- cosine between the matcher embeddings of the reconstruction and of
  the clean crop (``--id-model``; default ``cvlface_ir101``, the public held-out
  matcher). Difficulty is ``1 - id_cos``;
* ``psnr`` (on [0, 1] images, 99 for an exact copy), ``ssim``
  (``pytorch_msssim.ssim``, data range 1) and ``lpips`` (``lpips`` AlexNet, inputs
  scaled to [-1, 1]) against the clean crop;
* ``bytes`` -- the emitted size.

Each row also carries eight codec-independent descriptors of the clean crop
(Laplacian variance, gradient magnitude, Canny edge density, high-frequency DCT
share, grey-level entropy, brightness, contrast, colorfulness) and the dataset's
attributes: Color FERET from ``labels.csv`` (pose, yaw, pitch, |yaw|, glasses,
beard, mustache, race, gender, age), AI-Solutions-KK from ``attributes.csv`` (age,
gender, Monk skin-tone index), joined on (subject, image stem); crops without an
attribute row get empty cells.

Sample: ``random.Random(seed).sample`` of the sorted ``aligned_<res>/*/*.png`` files,
``--n`` per dataset. Codecs:

* the six classical codecs with a linear scan of the quality grid in steps of
  ``--quality-step`` (JPEG 2000 ratios in steps of 3x that), keeping the largest
  file within the budget (the smallest setting when none fits; settings that fail
  to encode are skipped);
* JPEG-AI (``--no-jpegai`` skips it): the analytic bpp target with up to three
  downward corrections; crops without a decodable stream are left out;
* face1kb-FAST and face1kb-ACCURATE (``paper_compat=True``; an identity-only
  container scores a black frame).

Reconstructions are compared as float images in [0, 1]; for the matcher they are
rounded to ``uint8`` (and resized to 112 px with the antialiased bilinear resize of
the codec's identity loss when ``--res`` is not 112).

The output ``OUTPUT_ROOT/difficulty/sample_difficulty.csv`` is per-image data tied to
dataset identities and attributes. Keep it local: it is not redistributable.
``analyze.py`` reduces it to the shipped aggregates.

The paper: ``--n 300 --res 112 --budgets 1024,512 --quality-step 4``, seed 0, with a
proprietary held-out matcher for ``id_cos`` (not distributed), so a public rerun
gives different difficulty values.

Examples
--------
    python experiments/difficulty/compute.py
    python experiments/difficulty/compute.py --datasets colorferet --n 20 --no-jpegai
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

import numpy as np

from face1kb import config
from face1kb.baselines import classical
from face1kb.data.cli_utils import parse_list, setup_logging

log = logging.getLogger("sample_difficulty")

OURS = ("ours_fast", "ours_accurate")
CODECS: tuple[str, ...] = (*classical.CODECS, "jpeg_ai", *OURS)
DESCRIPTORS: tuple[str, ...] = (
    "lap_var",
    "grad_mag",
    "edge_density",
    "hf_share",
    "entropy",
    "brightness",
    "contrast",
    "colorfulness",
)
FIELDS: tuple[str, ...] = (
    "dataset",
    "subject",
    "image",
    "rel_path",
    "budget",
    "codec",
    "bytes",
    "id_cos",
    "psnr",
    "ssim",
    "lpips",
    *DESCRIPTORS,
    "pose",
    "yaw",
    "pitch",
    "abs_yaw",
    "glasses",
    "beard",
    "mustache",
    "race",
    "gender",
    "age",
    "mst_index",
    "id_model",
)
OUT_NAME = "sample_difficulty.csv"


# ------------------------------------------------------------------ sample
def sample_crops(dataset: str, res: int, n: int, seed: int) -> list[Path]:
    """``random.Random(seed).sample`` of the sorted crop files of ``dataset``."""
    files = sorted(config.aligned_dir(dataset, res).glob("*/*.png"))
    return random.Random(seed).sample(files, min(n, len(files)))


# ------------------------------------------------------------------ descriptors
def to01(rgb_u8: np.ndarray):
    """``uint8`` HxWx3 -> float tensor 3xHxW in [0, 1]."""
    import torch  # noqa: PLC0415

    a = np.asarray(rgb_u8, np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def descriptors(img01) -> dict[str, float]:
    """Codec-independent descriptors of a clean crop (3xHxW float in [0, 1])."""
    import cv2  # noqa: PLC0415

    rgb = (img01.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = float(np.sqrt(gx * gx + gy * gy).mean())
    edges = cv2.Canny(gray.astype(np.uint8), 80, 160)
    edge_density = float((edges > 0).mean())
    d = cv2.dct(gray / 255.0)
    e = d * d
    e[0, 0] = 0.0
    total = float(e.sum()) + 1e-9
    h, w = e.shape
    hf_share = float((total - e[: h // 4, : w // 4].sum()) / total)
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    r, g, b = (rgb[..., k].astype(np.float32) for k in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )
    return {
        "lap_var": lap_var,
        "grad_mag": grad_mag,
        "edge_density": edge_density,
        "hf_share": hf_share,
        "entropy": entropy,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "colorfulness": colorfulness,
    }


# ------------------------------------------------------------------ attributes
def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return ""


def _float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_attributes() -> dict[str, dict]:
    """``{dataset: {(subject, stem): attributes}}`` from the label/attribute files."""
    out: dict[str, dict] = {"colorferet": {}, "kk": {}}
    fp = config.labels_csv("colorferet")
    if fp.exists():
        from face1kb.data.colorferet_labels import read_labels  # noqa: PLC0415

        for r in read_labels(fp).fillna("").to_dict("records"):
            yaw, pitch = _float(r.get("yaw")), _float(r.get("pitch"))
            if np.isnan(yaw) or np.isnan(pitch):
                yaw = pitch = float("nan")
            out["colorferet"][(r["subject"], Path(str(r["rel_path"])).stem)] = {
                "pose": r.get("pose", ""),
                "yaw": yaw,
                "pitch": pitch,
                "abs_yaw": abs(yaw),
                "glasses": 1
                if str(r.get("glasses", "")).strip().lower() == "yes"
                else 0,
                "beard": _to_int(r.get("beard", "")),
                "mustache": _to_int(r.get("mustache", "")),
                "race": r.get("race", ""),
                "gender": r.get("gender", ""),
                "age": _to_int(r.get("age_from", "")),
            }
    else:
        log.warning("no Color FERET labels at %s", fp)
    fp = config.attributes_csv("kk")
    if fp.exists():
        from face1kb.data.attributes import read_attributes  # noqa: PLC0415

        for r in read_attributes(fp).fillna("").to_dict("records"):
            out["kk"][(r["subject"], Path(str(r["image"])).stem)] = {
                "age": _to_int(r.get("age", "")),
                "gender": r.get("gender", ""),
                "mst_index": _to_int(r.get("mst_index", "")),
            }
    else:
        log.warning("no AI-Solutions-KK attributes at %s", fp)
    return out


# ------------------------------------------------------------------ codecs
class Codecs:
    """Encode + decode of one crop with every requested codec at a budget."""

    def __init__(self, codecs: list[str], res: int, qstep: int, device: str):
        import face1kb  # noqa: PLC0415

        self.codecs, self.res, self.device = codecs, res, device
        self.grids = {
            c: classical.settings(c, quality_step=qstep, ratio_step=3 * qstep)
            for c in codecs
            if c in classical.CODECS
        }
        self.ours = {
            c: face1kb.load(c.split("_", 1)[1], device=device)
            for c in codecs
            if c in OURS
        }

    def classical(self, pil, codec: str, budget: int):
        data, _ = classical.encode_to_budget(
            pil, codec, budget, grid=self.grids[codec], search="scan", skip_errors=True
        )
        return to01(classical.decode(data)), len(data)

    def jpeg_ai(self, pil, budget: int):
        from face1kb.baselines import jpeg_ai  # noqa: PLC0415

        def enc(bpp):
            data = jpeg_ai.encode_bpp(pil, bpp)
            return None if data is None else (len(data), data)

        fit = jpeg_ai.step_down_fit(enc, budget, pil.size[0] * pil.size[1])
        if fit is None:
            return None, 0
        dec = jpeg_ai.decode(fit.payload)
        return (to01(dec) if dec is not None else None), fit.size

    def face1kb(self, u8: np.ndarray, codec: str, budget: int):
        import torch  # noqa: PLC0415

        c = self.ours[codec]
        data, info = c.encode(u8, budget, paper_compat=True, return_info=True)
        x_hat, h = c.decode_tensor(data)
        if x_hat is None:
            x_hat = torch.zeros(1, 3, h.res, h.res, device=c.device)
        return x_hat[..., : h.res, : h.res][0].cpu(), int(info["bytes"])

    def run(self, pil, u8: np.ndarray, budget: int) -> dict:
        outs = {}
        for c in self.codecs:
            if c in classical.CODECS:
                outs[c] = self.classical(pil, c, budget)
            elif c == "jpeg_ai":
                try:
                    outs[c] = self.jpeg_ai(pil, budget)
                except Exception as e:  # noqa: BLE001 - JPEG-AI failures skip the row
                    log.debug("JPEG-AI failed: %s", e)
                    outs[c] = (None, 0)
            else:
                outs[c] = self.face1kb(u8, c, budget)
        return outs


# ------------------------------------------------------------------ metrics
class Metrics:
    """PSNR / SSIM / LPIPS and matcher identity cosine of a reconstruction."""

    def __init__(self, id_model: str, device: str):
        import lpips  # noqa: PLC0415
        from pytorch_msssim import ssim  # noqa: PLC0415

        from face1kb import fr  # noqa: PLC0415

        self.device = device
        self.ssim = ssim
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.embedder = fr.load(id_model, device=device)

    def _u8_112(self, x):
        from face1kb.codec.resize import resize112  # noqa: PLC0415

        y = resize112(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return np.ascontiguousarray((y * 255).round().astype(np.uint8))

    def __call__(self, clean01, dec01) -> dict[str, float]:
        import torch  # noqa: PLC0415

        from face1kb.eval.verification import paired_cosines  # noqa: PLC0415

        cn = clean01.unsqueeze(0).to(self.device).clamp(0, 1)
        dn = dec01.unsqueeze(0).to(self.device).clamp(0, 1)
        mse = torch.mean((cn - dn) ** 2).item()
        psnr = 99.0 if mse < 1e-9 else -10 * float(np.log10(mse))
        s = self.ssim(cn, dn, data_range=1.0).item()
        with torch.no_grad():
            lp = self.lpips(cn * 2 - 1, dn * 2 - 1).item()
        # one embedding call per image (batch of 1), as in the paper sweep
        ec = np.asarray(self.embedder.embed([self._u8_112(cn)]), np.float32)
        ed = np.asarray(self.embedder.embed([self._u8_112(dn)]), np.float32)
        cos = float(paired_cosines(ec, ed)[0])
        return {"id_cos": cos, "psnr": psnr, "ssim": s, "lpips": lp}


# ------------------------------------------------------------------ montage
CODEC_LABELS = {
    "jpeg": "JPEG",
    "jpeg2000": "JPEG 2000",
    "webp": "WebP",
    "avif": "AVIF",
    "heif": "HEIF",
    "jpeg_xl": "JPEG XL",
    "jpeg_ai": "JPEG-AI",
    "ours_fast": "Ours-FAST",
    "ours_accurate": "Ours-ACCURATE",
}


def montage(
    df,
    codecs: list[str],
    budget: int,
    res: int,
    qstep: int,
    device: str,
    out: Path,
    k: int = 5,
) -> None:
    """Easiest / hardest Color FERET crops per codec: clean above, decoded below.

    The ``k`` crops of highest and lowest ``id_cos`` at ``budget`` are re-encoded
    with the same codec settings as the sweep. The figure shows dataset faces; keep
    it local.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    d = df[(df.dataset == "colorferet") & (df.budget == budget)]
    enc = Codecs(codecs, res, qstep, device)
    root = config.aligned_dir("colorferet", res)
    ncols = 2 * k
    fig, axes = plt.subplots(
        len(codecs) * 2, ncols, figsize=(1.15 * ncols, 1.15 * len(codecs) * 2)
    )
    axes = np.asarray(axes).reshape(len(codecs) * 2, ncols)
    for ci, codec in enumerate(codecs):
        sub = d[d.codec == codec].copy()
        sub["idc"] = pd.to_numeric(sub["id_cos"], errors="coerce")
        sub = sub.dropna(subset=["idc"])
        picks = list(sub.nlargest(k, "idc").itertuples()) + list(
            sub.nsmallest(k, "idc").itertuples()
        )
        for j, row in enumerate(picks):
            rel = getattr(row, "rel_path", None) or f"{row.subject}/{row.image}.png"
            with Image.open(root / rel) as im:
                pil = im.convert("RGB")
            u8 = np.asarray(pil, dtype=np.uint8)
            if codec in classical.CODECS:
                dec01, _ = enc.classical(pil, codec, budget)
            elif codec in OURS:
                dec01, _ = enc.face1kb(u8, codec, budget)
            else:
                dec01, _ = enc.jpeg_ai(pil, budget)
            clean = np.asarray(pil.resize((112, 112)))
            rec = None
            if dec01 is not None:
                rec = (dec01.numpy().transpose(1, 2, 0).clip(0, 1) * 255).astype(
                    np.uint8
                )
            r0 = ci * 2
            for rr, img in ((r0, clean), (r0 + 1, rec)):
                ax = axes[rr, j]
                ax.axis("off")
                if img is not None:
                    ax.imshow(img)
            axes[r0, j].set_title(
                f"{row.idc:.2f}", fontsize=8, color=("#1a7f37" if j < k else "#b3261e")
            )
        axes[ci * 2, 0].set_ylabel(CODEC_LABELS.get(codec, codec), fontsize=9)
    fig.suptitle(
        f"Color FERET at {budget} B: {k} easiest (green) | {k} hardest (red). "
        "Top = clean, bottom = reconstruction, per codec.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    log.info("wrote %s (dataset faces: keep it local)", out)


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:  # noqa: C901 - sequential sweep
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument("--res", type=int, default=112)
    ap.add_argument("--budgets", default="1024,512")
    ap.add_argument("--n", type=int, default=300, help="crops per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--quality-step", type=int, default=4, help="classical quality scan step"
    )
    ap.add_argument("--codecs", default=",".join(CODECS))
    ap.add_argument("--no-jpegai", action="store_true")
    ap.add_argument(
        "--id-model",
        default=None,
        help="matcher of id_cos (default: face1kb.fr.IDCOS_DEFAULT); a registered "
        "user model works too (FACE1KB_FR_PLUGINS)",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="default: OUTPUT_ROOT/difficulty/...")
    ap.add_argument(
        "--montage", action="store_true", help="also render the difficulty montage"
    )
    ap.add_argument(
        "--montage-only",
        action="store_true",
        help="render the montage from an existing per-image CSV (--out)",
    )
    ap.add_argument("--montage-codecs", default="webp,ours_accurate")
    ap.add_argument("--montage-budget", type=int, default=512)
    ap.add_argument(
        "--montage-out", default=None, help="default: OUTPUT_ROOT/figures/..."
    )
    args = ap.parse_args(argv)
    setup_logging()
    from PIL import Image  # noqa: PLC0415

    from face1kb import fr  # noqa: PLC0415

    codecs = [
        c for c in parse_list(args.codecs) if not (args.no_jpegai and c == "jpeg_ai")
    ]
    bad = [c for c in codecs if c not in CODECS]
    if bad:
        raise SystemExit(f"unknown codec(s) {bad}; choose from {CODECS}")
    id_model = args.id_model or fr.IDCOS_DEFAULT
    budgets = parse_list(args.budgets, int)
    out = Path(args.out) if args.out else config.output_dir("difficulty") / OUT_NAME
    montage_out = (
        Path(args.montage_out)
        if args.montage_out
        else config.output_dir("figures") / "difficulty_montage.png"
    )

    def render_montage() -> None:
        import pandas as pd  # noqa: PLC0415

        montage(
            pd.read_csv(out),
            parse_list(args.montage_codecs),
            args.montage_budget,
            args.res,
            args.quality_step,
            args.device,
            montage_out,
        )

    if args.montage_only:
        render_montage()
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)

    metrics = Metrics(id_model, args.device)
    enc = Codecs(codecs, args.res, args.quality_step, args.device)
    attrs = load_attributes()
    log.info("id_cos matcher: %s; codecs: %s", id_model, codecs)
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(FIELDS), extrasaction="ignore")
        wr.writeheader()
        for ds in parse_list(args.datasets):
            crops = sample_crops(ds, args.res, args.n, args.seed)
            root = config.aligned_dir(ds, args.res)
            log.info("[%s] %d crops", ds, len(crops))
            amap = attrs.get(ds, {})
            for k, fp in enumerate(crops):
                subj, stem = fp.parent.name, fp.stem
                with Image.open(fp) as im:
                    pil = im.convert("RGB")
                u8 = np.asarray(pil, dtype=np.uint8)
                img01 = to01(u8)
                desc = descriptors(img01)
                at = amap.get((subj, stem), {})
                for budget in budgets:
                    for codec, (dec01, nbytes) in enc.run(pil, u8, budget).items():
                        if dec01 is None:
                            continue
                        wr.writerow(
                            {
                                "dataset": ds,
                                "subject": subj,
                                "image": stem,
                                "rel_path": fp.relative_to(root).as_posix(),
                                "budget": budget,
                                "codec": codec,
                                "bytes": nbytes,
                                **metrics(img01, dec01),
                                **desc,
                                **at,
                                "id_model": id_model,
                            }
                        )
                fh.flush()
                if (k + 1) % 25 == 0:
                    log.info("  [%s] %d/%d", ds, k + 1, len(crops))
    log.info("wrote %s (per-image data: keep it local)", out)
    if args.montage:
        del metrics, enc
        render_montage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
