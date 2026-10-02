# SPDX-License-Identifier: MIT
"""Train a face1kb codec variant (single GPU, step-based).

Recipe (the defaults are the training recipe of the released weights, except the
length: ``--steps`` defaults to 600k, while the released runs used 1M-step schedules;
pass ``--steps 1000000`` as in ``docs/training.md``):

* **Data**: every batch is drawn uniformly at random, with replacement, from the
  whole training source (see :mod:`face1kb.codec.data`); no epochs, no augmentation.
* **Variable rate**: each step samples a gain level ``s`` of the model's 8 training
  levels and weights the distortion terms by that level's ``lmbda[s]`` times a
  distortion scale ``DSCALE = 256`` (rate + lmbda * DSCALE * distortion), so one model
  spans all budgets.
* **Dynamic resolution**: each step samples one resolution bucket (weighted), resizes
  the 112 px batch to it and edge-pads it to a multiple of 128. FAST uses buckets
  ``64,128,192,224,256`` with weights ``1:4:3:3:5``; ACCURATE uses ``64,128,192,256``
  with uniform weights.
* **Losses** (:mod:`face1kb.codec.losses`): rate, MSE, LPIPS (AlexNet), MS-SSIM and an
  MS-SSIM floor hinge from step 0, plus an EdgeFace identity-cosine term (EdgeFace-XS
  for FAST, EdgeFace-S for ACCURATE) that is off for the first 2 % of the steps,
  ramped linearly until 20 % and at full weight afterwards.
* **Optimisation**: Adam (lr 1e-4) with a linear warm-up over 1 % of the steps
  (from 1 % of the lr) followed by cosine annealing to 1e-5; gradient clipping at
  1.0; fp16 autocast for the forward pass; TF32 enabled; a separate Adam (lr 1e-3)
  for the entropy-bottleneck quantiles (auxiliary loss).
* **Checkpoints**: ``<ckpt>/last.pt`` (network, optimisers, scheduler, step,
  variant) every ``--ckpt-every`` steps, ``--resume`` continues from it (an extended
  run, ``--rebuild-sched``, records its schedule and is resumed on the same curve).
  Logs go to ``<ckpt>/train_log.jsonl``. ``--export`` writes a net-only safetensors
  file; for the default FAST / ACCURATE configurations it loads with
  :func:`face1kb.load`.

The released weights come from warm-start chains (FAST: a 300k-step run continued
for 1M steps with the re-balanced buckets; ACCURATE: a 60k-step run, a warm start on
a 1M-step schedule stopped at 990k steps, then an extension to 3M steps on a rebuilt
cosine schedule), including runs on earlier code revisions. This trainer reproduces
the recipe, not the exact weights. See ``docs/training.md``.

Example::

    python -m face1kb.codec.train --variant fast --data /path/to/webface42m \
        --steps 1000000 --ckpt runs/fast
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from . import data as D
from .identity_loss import InLoopIdentityLoss
from .losses import CodecLoss, phase_weights
from .resize import pad_to_multiple, resize_to
from .variants import build_model

logger = logging.getLogger("face1kb.codec.train")

#: Resolution buckets and sampling weights of the released weights.
RECIPES: dict[str, dict] = {
    "fast": {"buckets": (64, 128, 192, 224, 256), "res_weights": (1, 4, 3, 3, 5)},
    "accurate": {"buckets": (64, 128, 192, 256), "res_weights": (1, 1, 1, 1)},
}
#: In-loop identity models per variant.
ID_MODELS = {"fast": ("edgeface_xs",), "accurate": ("edgeface_s",)}
#: Target loss weights (the distortion terms are scaled by lmbda[s] * DSCALE).
BASE_WEIGHTS = {"R": 1.0, "D": 2.0, "p": 1.0, "id": 0.5, "ms": 0.3, "floor": 1.0}
FAST_WEIGHTS = {"R": 1.0, "D": 0.3, "p": 1.0, "id": 0.4, "ms": 0.3, "floor": 1.0}
#: Distortion scale (restores the CompressAI ~255^2 distortion magnitude, so that
#: the distortion dominates the bpp term and the model spends bits up to the budget).
DSCALE = 256.0


def split_params(net):
    """Split trainable parameters into ``(main, aux)``; aux = bottleneck quantiles."""
    aux, main = [], []
    for n, p in net.named_parameters():
        if not p.requires_grad:  # the frozen side-stream anchor
            continue
        (aux if n.endswith(".quantiles") else main).append(p)
    return main, aux


def _csv(s: str, typ=float) -> tuple:
    return tuple(typ(v) for v in s.split(",") if v.strip())


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the trainer."""
    ap = argparse.ArgumentParser(
        prog="python -m face1kb.codec.train", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("--variant", choices=["fast", "accurate"], default="fast")
    ap.add_argument(
        "--data",
        required=True,
        help="training source: HuggingFace save_to_disk folder or image folder "
        "<root>/<subject>/<image>",
    )
    ap.add_argument(
        "--data-list",
        default=None,
        help="image folders only: cached file list (written on first use)",
    )
    ap.add_argument("--ckpt", required=True, help="checkpoint / log folder")
    ap.add_argument(
        "--buckets",
        default=None,
        help="comma-separated resolution buckets (default: the variant's recipe)",
    )
    ap.add_argument(
        "--res-weights",
        default=None,
        help="comma-separated bucket sampling weights (default: the recipe's)",
    )
    ap.add_argument("--dscale", type=float, default=DSCALE)
    ap.add_argument(
        "--no-side-stream",
        action="store_true",
        help="ACCURATE only: no identity side-stream and no refine head (ablation)",
    )
    ap.add_argument(
        "--anchor",
        default="edgeface_s",
        help="ACCURATE side-stream anchor (a registered torch FR model)",
    )
    ap.add_argument(
        "--anchor-train-mode",
        action="store_true",
        help="ACCURATE only: put the frozen anchor into training mode together with "
        "the network, as the paper runs did (matters only for anchors with "
        "batch-norm or dropout, e.g. the TopoFR ablation arm)",
    )
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    ap.add_argument(
        "--steps",
        type=int,
        default=600_000,
        help="schedule length: LR warm-up / cosine and the identity ramp are "
        "fractions of it (the released runs used 1000000)",
    )
    ap.add_argument(
        "--stop-at",
        type=int,
        default=None,
        help="stop after this absolute step but keep the schedule of --steps "
        "(default: --steps)",
    )
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--aux-lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="continue from last.pt")
    ap.add_argument(
        "--rebuild-sched",
        action="store_true",
        help="with --resume: discard the saved LR schedule and anneal a fresh cosine "
        "from the full --lr at --extend-from down to 1e-5 at --steps (to extend a "
        "finished run)",
    )
    ap.add_argument(
        "--extend-from",
        type=int,
        default=-1,
        help="absolute step where the rebuilt cosine starts (default: the "
        "checkpoint's step); pass it explicitly so repeated resumes give the same "
        "curve",
    )
    ap.add_argument(
        "--warmstart-from",
        default="",
        help="initialise the network from a checkpoint (last.pt or a .safetensors "
        "weight file)",
    )
    ap.add_argument(
        "--warmstart-reset-gain",
        action="store_true",
        help="keep the freshly initialised Gain (skip it in the warm-start load)",
    )
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--val-mod", type=int, default=64)
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="data / bucket / level seed")
    ap.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="seed torch before building the model (default: unseeded, as the "
        "released runs)",
    )
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 autocast")
    ap.add_argument(
        "--export",
        default="",
        help="also write a net-only safetensors file at every checkpoint",
    )
    return ap


def _load_checkpoint(path: Path, device) -> dict:
    """Load a trainer checkpoint, or a net-only ``.safetensors`` file as ``{"net"}``."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from .api import load_state_dict_file  # noqa: PLC0415

        return {"net": load_state_dict_file(path)[0]}
    # Trainer checkpoints hold only tensors and plain containers, so the restricted
    # unpickler suffices; it refuses checkpoints that would execute code on load.
    return torch.load(path, map_location=device, weights_only=True)


def save_safetensors(
    net,
    path: str | Path,
    variant: str,
    step: int,
    no_side: bool = False,
    anchor: str = "edgeface_s",
) -> None:
    """Write the network's state dict as a net-only fp32 safetensors file.

    For ACCURATE the metadata records ``face1kb.no_side`` and ``face1kb.anchor`` so
    that :func:`face1kb.load` can refuse ablation architectures it does not build.
    """
    from safetensors.torch import save_file  # noqa: PLC0415

    from .api import WEIGHTS_FORMAT_VERSION  # noqa: PLC0415

    sd = {k: v.detach().to("cpu").contiguous() for k, v in net.state_dict().items()}
    meta = {
        "format": "pt",
        "face1kb.format_version": WEIGHTS_FORMAT_VERSION,
        "face1kb.variant": variant,
        "face1kb.step": str(int(step)),
        "face1kb.architecture": type(net).__name__,
    }
    if variant == "accurate":
        meta["face1kb.no_side"] = "true" if no_side else "false"
        meta["face1kb.anchor"] = anchor
    save_file(sd, str(path), metadata=meta)


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - training entry point
    """Run the training loop."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    recipe = RECIPES[args.variant]
    buckets = _csv(args.buckets, int) if args.buckets else recipe["buckets"]
    rw = _csv(args.res_weights) if args.res_weights else recipe["res_weights"]
    if len(rw) != len(buckets):
        raise SystemExit(
            f"--res-weights has {len(rw)} entries for {len(buckets)} buckets"
        )
    res_weights = torch.tensor(rw, dtype=torch.float32)
    res_weights = res_weights / res_weights.sum()

    if not torch.cuda.is_available():
        raise SystemExit("training requires a CUDA GPU")
    dev = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ckpt_dir = Path(args.ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(
        json.dumps({**vars(args), "buckets": buckets, "res_weights": rw}, indent=2)
    )
    log_f = open(ckpt_dir / "train_log.jsonl", "a")  # noqa: SIM115

    if args.init_seed is not None:
        torch.manual_seed(args.init_seed)
    net = build_model(
        args.variant, args.no_side_stream, args.anchor, pretrained_anchor=True
    ).to(dev)
    if args.anchor_train_mode:
        side = getattr(net, "side", None)
        if side is None:
            raise SystemExit("--anchor-train-mode needs the ACCURATE side-stream")
        side.freeze_anchor_mode = False
    n_lmbda = len(net.lmbda)
    id_loss = InLoopIdentityLoss(ID_MODELS[args.variant], device=dev)
    criterion = CodecLoss(id_loss, use_lpips=not args.no_lpips).to(dev)
    base_w = FAST_WEIGHTS if args.variant == "fast" else BASE_WEIGHTS

    main_p, aux_p = split_params(net)
    opt = torch.optim.Adam(main_p, lr=args.lr)
    aux_opt = torch.optim.Adam(aux_p, lr=args.aux_lr)
    # Cosine LR with a short linear warm-up.
    warmup = max(1, int(0.01 * args.steps))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.01, total_iters=warmup
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, max(1, args.steps - warmup), eta_min=1e-5
            ),
        ],
        milestones=[warmup],
    )

    start_step = 0
    last = ckpt_dir / "last.pt"
    # Schedule bookkeeping stored in last.pt, so that a resumed extension run stays
    # on its rebuilt cosine without repeating --rebuild-sched --extend-from.
    sched_kind, sched_base = "warmup_cosine", 0
    if args.resume and last.exists():
        ck = _load_checkpoint(last, dev)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        aux_opt.load_state_dict(ck["aux_opt"])
        start_step = ck["step"]
        rebuild, extend_from = args.rebuild_sched, args.extend_from
        if not rebuild and ck.get("sched_kind") == "cosine_restart":
            rebuild = True  # identical to passing the recorded flags again
            if extend_from < 0:
                extend_from = int(ck["extend_from"])
        elif not rebuild and "_schedulers" not in ck["sched"]:
            raise SystemExit(
                f"{last} holds a rebuilt cosine schedule without its start step; "
                "resume it with --rebuild-sched --extend-from <step of the extension "
                "start>"
            )
        if rebuild:
            # Warm restart for an extension run: reset the LR to the full --lr and
            # anneal a fresh cosine over [extend_from, steps] down to 1e-5. The curve
            # is keyed to the absolute step, so repeated resumes with the same
            # --extend-from re-derive the same curve.
            base = extend_from if extend_from >= 0 else start_step
            sched_kind, sched_base = "cosine_restart", base
            for g in opt.param_groups:
                g["lr"] = args.lr
                g.pop("initial_lr", None)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, args.steps - base), eta_min=1e-5
            )
            for _ in range(max(0, start_step - base)):
                sched.step()
            logger.info(
                "rebuilt LR schedule: base=%.1e over [%d,%d], lr=%.3e at step %d",
                args.lr,
                base,
                args.steps,
                opt.param_groups[0]["lr"],
                start_step,
            )
        else:
            sched.load_state_dict(ck["sched"])
        logger.info("resumed from step %d", start_step)
    elif args.warmstart_from:
        ck = _load_checkpoint(Path(args.warmstart_from), dev)
        sd = ck["net"] if "net" in ck else ck
        if args.warmstart_reset_gain:
            sd = {k: v for k, v in sd.items() if k != "Gain"}
            net.load_state_dict(sd, strict=False)  # keep the freshly initialised Gain
            logger.info("warm-started (Gain reset) from %s", args.warmstart_from)
        else:
            net.load_state_dict(sd)
            logger.info("warm-started from %s", args.warmstart_from)

    loader = D.make_random_loader(
        args.data,
        "train",
        batch_size=args.batch,
        num_workers=args.workers,
        val_mod=args.val_mod,
        seed=args.seed,
        list_file=args.data_list,
    )
    gen = torch.Generator().manual_seed(args.seed)
    net.train()
    step = start_step
    t0 = time.time()
    data_iter = iter(loader)
    end = args.steps if args.stop_at is None else min(args.stop_at, args.steps)
    while step < end:
        try:
            imgs, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            imgs, _ = next(data_iter)
        # One resolution bucket per step (weighted); resize, then pad to /128.
        bucket = buckets[int(torch.multinomial(res_weights, 1, generator=gen).item())]
        x = resize_to(imgs.to(dev, non_blocking=True), bucket)
        x, _ = pad_to_multiple(x)
        s = int(torch.randint(n_lmbda, (1,), generator=gen).item())
        lmbda = float(net.lmbda[s])

        # fp16 autocast on the conv-heavy forward; rate and perceptual terms in fp32.
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = net(x, stage=2, s=s, res=bucket)
        out["x_hat"] = out["x_hat"].float()
        out["likelihoods"] = {k: v.float() for k, v in out["likelihoods"].items()}
        w = phase_weights(step, args.steps, base_w, fast=(args.variant == "fast"))
        # RD objective: bpp + lmbda * DSCALE * distortion.
        scaled = {"R": w.get("R", 1.0)}
        for k in ("D", "p", "id", "ms"):
            if k in w:
                scaled[k] = w[k] * lmbda * args.dscale
        scaled["floor"] = w.get("floor", 0.0) * args.dscale
        loss, terms = criterion(x, out, scaled)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(main_p, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        aux_opt.zero_grad(set_to_none=True)
        aux = net.aux_loss()  # fp32 (entropy-bottleneck quantiles)
        aux.backward()
        aux_opt.step()

        if step % args.log_every == 0:
            rate = (step - start_step) / max(time.time() - t0, 1e-6)
            lr = opt.param_groups[0]["lr"]
            rec = {
                "step": step,
                "bucket": bucket,
                "s": s,
                "lmbda": round(lmbda, 4),
                "lr": round(lr, 7),
                "aux": round(float(aux.detach()), 2),
                "it_s": round(rate, 2),
                **terms,
            }
            logger.info(json.dumps(rec))
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

        step += 1
        if step % args.ckpt_every == 0 or step == end:
            tmp = last.with_suffix(".pt.tmp")
            torch.save(
                {
                    "net": net.state_dict(),
                    "opt": opt.state_dict(),
                    "aux_opt": aux_opt.state_dict(),
                    "sched": sched.state_dict(),
                    "step": step,
                    "variant": args.variant,
                    "sched_kind": sched_kind,
                    "extend_from": sched_base,
                },
                tmp,
            )
            tmp.replace(last)
            if args.export:
                save_safetensors(
                    net,
                    args.export,
                    args.variant,
                    step,
                    no_side=args.no_side_stream,
                    anchor=args.anchor,
                )
    log_f.close()
    logger.info("training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
