# SPDX-License-Identifier: MIT
r"""Verification-grade comparison of the three JPEG-AI decoders on AI-Solutions-KK.

The JPEG-AI standard defines three operation points of increasing synthesis-transform
depth: SOP (simple), BOP (base) and HOP (high). This study (``tab:jpegai-kk``)
encodes a subject-stratified sample of AI-Solutions-KK 224 px crops with each of
them at a fixed target of 0.16 bpp (about 1.1 kB), then measures

* verification EER (%) over all pairs of the sample, per face matcher (ArcFace
  antelopev2 and EdgeFace-XS by default);
* identity cosine between each reconstruction and its original (mean);
* PSNR and SSIM of the reconstructions (medians) and the median stream size.

Phases:

``--phase encode``
    Encodes the sample with each profile (:func:`face1kb.baselines.jpeg_ai.encode_bpp`
    with ``return_recon=True``: the encoder-side reconstruction, which equals the
    decoder output) and stores the streams, reconstructions and a manifest in
    ``--run-dir`` (default ``WORK_ROOT/kk/jpegai_kk``). GPU; about 1 h for n=600
    on one RTX 2080 Ti.
``--phase score``
    Embeds the originals and reconstructions (resized to 112 px, bilinear),
    computes the metrics, and writes ``OUTPUT_ROOT/codec_comparison/jpegai_kk.csv``
    and the table. GPU, minutes.
``--phase table``
    Renders only the table, from ``--csv`` (e.g. the shipped
    ``results/codec_comparison/jpegai_kk.csv``, with ``--from-results``).

Sample: identities with at least two crops, shuffled with ``default_rng(seed)``; each
identity's crops are permuted with the same generator, then crops are taken
round-robin over the identities (at most ``--max-per-id`` each) until ``--n``.
EER: over all pairs of the sample (mated = same identity), the threshold minimising
``|FMR - FNMR|`` among the unique scores, EER = (FMR + FNMR) / 2.

Table: operation points x (synthesis depth, ArcFace EER, EdgeFace EER, id-cos, PSNR,
bytes), the uncompressed ``aligned`` reference row first. Shading on the ArcFace
EER, id-cos and PSNR columns (best / worst of the three operation points; on a tie
the first one listed is marked). ``--no-eer-shading`` leaves the EER column
unshaded, since the three EERs are statistically indistinguishable.

The per-crop files of the run folder are derived from AI-Solutions-KK and must not
be redistributed; only the aggregate CSV is shipped.

Examples
--------
    python experiments/compress/measure_jpegai_kk.py --phase all --gpu 0
    python experiments/compress/measure_jpegai_kk.py --phase table --from-results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

from face1kb import config

log = logging.getLogger("measure_jpegai_kk")

#: (JPEG-AI profile key, operation-point label).
PROFILES = (("sop", "SOP"), ("bop", "BOP"), ("hop", "HOP"))
#: Target bits per pixel x 100 (about 1.1 kB at 224 px).
TARGET_BPP = 16
RES = 224
EMBED_RES = 112
DEFAULT_MODELS = ("arcface_antelopev2", "edgeface_xs")
#: Matcher whose EER / id-cos / PSNR columns are shaded.
PRIMARY = "arcface_antelopev2"
STEM = "jpegai_kk"
CSV_COLUMNS = ("model", "op_point", "n", "eer", "id_cos", "psnr", "ssim", "bytes")


# ---------------------------------------------------------------- sampling
def sample_crops(root: Path, n: int, max_per_id: int, seed: int) -> list[tuple]:
    """Subject-stratified sample ``[(identity, stem, path), ...]`` of ``root``."""
    by_id: dict[str, list[Path]] = {}
    for p in sorted(root.rglob("*.png")):
        by_id.setdefault(p.parent.name, []).append(p)
    ids = sorted(k for k, v in by_id.items() if len(v) >= 2)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    per = {i: 0 for i in ids}
    pools = {i: list(rng.permutation(by_id[i])) for i in ids}
    picked = []
    while len(picked) < n:
        added = False
        for i in ids:
            if len(picked) >= n:
                break
            if per[i] < min(max_per_id, len(pools[i])):
                picked.append((i, pools[i][per[i]]))
                per[i] += 1
                added = True
        if not added:
            break
    return [(ident, p.stem, str(p)) for ident, p in picked]


# ---------------------------------------------------------------- metrics
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR (dB) of two uint8 images; 99 for identical images."""
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse < 1e-9 else float(10 * np.log10(255.0**2 / mse))


def all_pairs_eer(emb: np.ndarray, labels) -> float:
    """Verification EER (%) over all pairs of ``emb`` (cosine similarity).

    Mated pairs share a label. For every unique score ``t``: FNMR = share of mated
    scores ``< t``, FMR = share of non-mated scores ``>= t``; the EER is
    ``(FMR + FNMR) / 2`` at the ``t`` minimising ``|FMR - FNMR|``.
    """
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim = e @ e.T
    iu = np.triu_indices(sim.shape[0], k=1)
    scores = sim[iu]
    lab = np.array(labels)
    same = lab[iu[0]] == lab[iu[1]]
    gen = np.sort(scores[same])
    imp = np.sort(scores[~same])
    if len(gen) == 0 or len(imp) == 0:
        return float("nan")
    thr = np.unique(scores)
    frr = np.searchsorted(gen, thr, side="left") / len(gen)
    far = (len(imp) - np.searchsorted(imp, thr, side="left")) / len(imp)
    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2 * 100)


def _normalise(e: np.ndarray) -> np.ndarray:
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------- phases
def phase_encode(run_dir: Path, n: int, max_per_id: int, seed: int) -> None:
    """Encode the sample with each profile; write streams, recons and a manifest."""
    from PIL import Image  # noqa: PLC0415

    from face1kb.baselines import jpeg_ai  # noqa: PLC0415

    jpeg_ai.get_reference().require()
    samples = sample_crops(config.aligned_dir("kk", RES), n, max_per_id, seed)
    log.info(
        "sampled %d crops of %d identities", len(samples), len({s[0] for s in samples})
    )
    manifest = {
        "samples": [
            [s[0], s[1], str(Path(s[2]).relative_to(config.aligned_dir("kk", RES)))]
            for s in samples
        ],
        "target_bpp_x100": TARGET_BPP,
        "profiles": {},
    }
    for profile, op in PROFILES:
        pdir = run_dir / profile
        pdir.mkdir(parents=True, exist_ok=True)
        recs, sizes = [], []
        for k, (ident, stem, path) in enumerate(samples):
            tag = f"{ident}__{stem}"
            with Image.open(path) as im:
                img = im.convert("RGB")
            bits, recon = jpeg_ai.encode_bpp(
                img, TARGET_BPP, profile=profile, return_recon=True
            )
            if bits is None or recon is None:
                log.warning("%s %s: encode failed", op, tag)
                recs.append(None)
                sizes.append(-1)
                continue
            (pdir / f"{tag}.bits").write_bytes(bits)
            Image.fromarray(recon).save(pdir / f"{tag}_rec.png")
            recs.append(f"{profile}/{tag}_rec.png")
            sizes.append(len(bits))
            if (k + 1) % 50 == 0:
                good = [b for b in sizes if b > 0]
                log.info(
                    "[%s] %d/%d median %d B", op, k + 1, len(samples), np.median(good)
                )
        manifest["profiles"][op] = {"profile": profile, "rec": recs, "bytes": sizes}
        good = [b for b in sizes if b > 0]
        log.info(
            "%s: %d/%d ok, median %s B",
            op,
            len(good),
            len(samples),
            int(np.median(good)) if good else -1,
        )
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    log.info("wrote %s", run_dir / "manifest.json")


def phase_score(run_dir: Path, models, device: str | None):
    """Embed originals and reconstructions; return the metric rows."""
    import cv2  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from skimage.metrics import structural_similarity  # noqa: PLC0415

    from face1kb import fr  # noqa: PLC0415

    manifest = json.loads((run_dir / "manifest.json").read_text())
    root = config.aligned_dir("kk", RES)
    samples = manifest["samples"]
    labels = [s[0] for s in samples]

    def rgb(path):
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), np.uint8)

    def small(a):
        return cv2.resize(a, (EMBED_RES, EMBED_RES), interpolation=cv2.INTER_LINEAR)

    origs = [rgb(root / s[2]) for s in samples]
    recons_by_op = {}
    for _profile, op in PROFILES:
        info = manifest["profiles"][op]
        recs, idx, psnrs, ssims = [], [], [], []
        for i, rp in enumerate(info["rec"]):
            if rp is None or not (run_dir / rp).exists():
                continue
            r = rgb(run_dir / rp)
            if r.shape[:2] != (RES, RES):
                r = cv2.resize(r, (RES, RES), interpolation=cv2.INTER_LINEAR)
            recs.append(small(r))
            idx.append(i)
            psnrs.append(psnr(origs[i], r))
            ssims.append(structural_similarity(origs[i], r, channel_axis=2))
        recons_by_op[op] = (recs, idx, psnrs, ssims, info["bytes"])

    rows = []
    for name in models:
        model = fr.load(name, device=device)
        emb_o = model.embed([small(a) for a in origs]).astype(np.float32)
        rows.append(
            {
                "model": name,
                "op_point": "aligned",
                "n": len(origs),
                "eer": all_pairs_eer(emb_o, labels),
                "id_cos": 1.0,
                "psnr": 99.0,
                "ssim": 1.0,
                "bytes": 0,
            }
        )
        eo = _normalise(emb_o)
        for _profile, op in PROFILES:
            recs, idx, psnrs, ssims, sizes = recons_by_op[op]
            if not recs:
                continue
            er = _normalise(model.embed(recs).astype(np.float32))
            good = [sizes[i] for i in idx if sizes[i] > 0]
            row = {
                "model": name,
                "op_point": op,
                "n": len(recs),
                "eer": all_pairs_eer(er, [labels[i] for i in idx]),
                "id_cos": float(np.mean(np.sum(er * eo[idx], axis=1))),
                "psnr": float(np.median(psnrs)),
                "ssim": float(np.median(ssims)),
                "bytes": int(np.median(good)) if good else -1,
            }
            rows.append(row)
            log.info(
                "%s %s: n=%d EER=%.3f%% id-cos=%.4f PSNR=%.2f bytes=%d",
                name,
                op,
                row["n"],
                row["eer"],
                row["id_cos"],
                row["psnr"],
                row["bytes"],
            )
    return rows


# ---------------------------------------------------------------- table
def render_table(df, shade_eer: bool = True) -> str:
    """Colour-only LaTeX table of the CSV rows (see the module docstring)."""
    from face1kb.report import latex  # noqa: PLC0415

    d = df[df.model == PRIMARY].set_index("op_point")
    order = [o for o in ("aligned", "SOP", "BOP", "HOP") if o in d.index]
    ops = [o for o in order if o != "aligned"]

    def extremes(field, lower_better):
        vals = {o: float(d.loc[o, field]) for o in ops}
        if not vals:
            return {}
        pick_best = min if lower_better else max
        pick_worst = max if lower_better else min
        return {
            pick_best(vals, key=vals.get): "best",
            pick_worst(vals, key=vals.get): "worst",
        }

    marks = {
        "eer": extremes("eer", True) if shade_eer else {},
        "id_cos": extremes("id_cos", False),
        "psnr": extremes("psnr", False),
    }

    def cell(o, field, nd):
        txt = f"{float(d.loc[o, field]):.{nd}f}"
        mark = marks[field].get(o) if o != "aligned" else None
        if mark == "best":
            return latex.shade_best(txt, emphasis=False)
        if mark == "worst":
            return latex.shade_worst(txt, emphasis=False)
        return txt

    second = df[df.model == "edgeface_xs"].set_index("op_point")
    synth = {"aligned": "--", "SOP": "1", "BOP": "2", "HOP": "3"}
    lines = [
        "% generated by measure_jpegai_kk.py -- JPEG-AI 3 decoders, KK 224px/1024B",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Op.\\ point & Synth. & EER\\,\\% & EER\\,\\% & id-cos & PSNR & Bytes \\\\",
        " & transf. & (ArcFace) & (EdgeFace) & (recon) & (dB) & \\\\",
        "\\midrule",
    ]
    for o in order:
        edge = f"{float(second.loc[o, 'eer']):.3f}" if o in second.index else "--"
        ref = o == "aligned"
        lines.append(
            f"{o} & {synth[o]} & {cell(o, 'eer', 3)} & {edge} & "
            f"{'--' if ref else cell(o, 'id_cos', 4)} & "
            f"{'--' if ref else cell(o, 'psnr', 2)} & "
            f"{'--' if ref else int(d.loc[o, 'bytes'])} \\\\"
        )
        if ref:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def write_table(df, out_root: Path, shade_eer: bool) -> Path:
    """Write ``tables/jpegai_kk.tex`` with the paper's emphasis marker."""
    from face1kb.report import latex  # noqa: PLC0415

    out = out_root / "tables" / f"{STEM}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex.finalize_table(STEM, render_table(df, shade_eer)))
    log.info("wrote %s", out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--phase", choices=("encode", "score", "all", "table"), default="all"
    )
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--max-per-id", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES")
    ap.add_argument("--device", default=None, help="torch device of the matchers")
    ap.add_argument("--run-dir", default=None, help="per-crop working folder")
    ap.add_argument("--csv", default=None, help="CSV to render with --phase table")
    ap.add_argument("--from-results", action="store_true", help="table from results/")
    ap.add_argument("--output-root", default=None, help="override FACE1KB_OUTPUT_ROOT")
    ap.add_argument("--no-eer-shading", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import pandas as pd  # noqa: PLC0415

    out_root = Path(args.output_root) if args.output_root else config.OUTPUT_ROOT
    run_dir = (
        Path(args.run_dir) if args.run_dir else config.work_dir("kk") / "jpegai_kk"
    )
    csv_path = out_root / "codec_comparison" / f"{STEM}.csv"
    if args.phase == "table":
        src = Path(
            args.csv
            or (
                config.results_dir("codec_comparison", f"{STEM}.csv")
                if args.from_results
                else csv_path
            )
        )
        write_table(pd.read_csv(src), out_root, not args.no_eer_shading)
        return 0
    if args.phase in ("encode", "all"):
        run_dir.mkdir(parents=True, exist_ok=True)
        phase_encode(run_dir, args.n, args.max_per_id, args.seed)
    if args.phase in ("score", "all"):
        models = [m for m in args.models.split(",") if m]
        df = pd.DataFrame(
            phase_score(run_dir, models, args.device), columns=list(CSV_COLUMNS)
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        log.info("wrote %s", csv_path)
        if PRIMARY in set(df.model):
            write_table(df, out_root, not args.no_eer_shading)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
