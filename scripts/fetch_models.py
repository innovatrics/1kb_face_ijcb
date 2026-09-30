#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Download the face-recognition evaluators from their official sources.

Every file is written under ``FACE1KB_MODELS_ROOT`` (default ``<repo>/models``) and
verified against the SHA-256 pinned in ``face1kb/fr/sources.py``. Files that are
already present and valid are kept, so the script can be re-run at any time.

The weights are not part of face1kb and are not redistributed by it. They are
released by their authors under their own terms -- mostly for non-commercial
research only; the notice of each model is printed before its files are fetched
(full table: ``docs/models.md``).

Examples
--------
::

    python scripts/fetch_models.py                     # all 14 evaluators
    python scripts/fetch_models.py --only anchors      # the 4 anchor matchers
    python scripts/fetch_models.py --only training     # EdgeFace-XS/S (codec training)
    python scripts/fetch_models.py --only lvface_l,topofr_r100
    python scripts/fetch_models.py --list              # what would be fetched
    python scripts/fetch_models.py --check             # verify only, no download

TopoFR checkpoints are hosted on Google Drive and fetched with ``gdown``; when that
fails (quota, network), the script prints where to download each file by hand and
where to save it, and exits with status 1.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # allow running from a checkout without installing
    sys.path.insert(0, str(REPO))

from face1kb import fr  # noqa: E402
from face1kb.fr import download as fr_fetch  # noqa: E402

#: Named groups accepted by ``--only`` (besides the model names).
GROUPS: dict[str, tuple[str, ...]] = {
    "all": fr.ROSTER,
    "anchors": fr.ANCHORS,
    # The FR models the codec training uses (identity loss / ACCURATE anchor).
    "training": ("edgeface_xs", "edgeface_s"),
    "heldout": (fr.HELDOUT,),
    **{
        family: tuple(n for n in fr.ROSTER if fr.REGISTRY[n].family == family)
        for family in ("lvface", "arcface", "cvlface", "edgeface", "topofr")
    },
}


def select(spec: str) -> list[str]:
    """Resolve a comma list of model and group names, keeping roster order."""
    wanted: set[str] = set()
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        if token in GROUPS:
            wanted.update(GROUPS[token])
        elif token in fr.ROSTER:
            wanted.add(token)
        else:
            raise SystemExit(
                f"unknown model or group {token!r}; models: {', '.join(fr.ROSTER)}; "
                f"groups: {', '.join(GROUPS)}"
            )
    return [n for n in fr.ROSTER if n in wanted]


def _fmt_size(n: int) -> str:
    return f"{n / 2**20:,.0f} MiB" if n >= 2**20 else f"{n / 1024:,.0f} KiB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--only",
        default="all",
        help="comma list of model names and/or groups "
        f"({', '.join(GROUPS)}); default: all",
    )
    ap.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="destination (default: FACE1KB_MODELS_ROOT, else <repo>/models)",
    )
    ap.add_argument("--list", action="store_true", help="list the files and exit")
    ap.add_argument("--check", action="store_true", help="verify only; no download")
    ap.add_argument(
        "--force",
        action="store_true",
        help="delete files that fail verification and download them again",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="    %(message)s", stream=sys.stdout)

    names = select(args.only)
    root = fr_fetch.models_root(args.models_root)
    total = sum(fr.model_source(n).size for n in names)
    print(f"models root: {root}")
    print(f"{len(names)} evaluator(s), {_fmt_size(total)} in total\n")

    if args.list:
        for name in names:
            src = fr.model_source(name)
            print(f"{name}  ({_fmt_size(src.size)}; {src.homepage})")
            for f in src.files:
                via = ""
                if f.archive is not None:
                    archive = f.archive.urls[0].rsplit("/", 1)[-1]
                    via = f"  (from {archive}, {_fmt_size(f.archive.size)} download)"
                elif f.gdrive_id is not None:
                    via = "  (Google Drive)"
                print(f"    {f.path}  {f.sha256[:16]}...  {_fmt_size(f.size)}{via}")
        return 0

    manual: list[fr.ManualDownloadRequired] = []
    failed: list[str] = []
    for name in names:
        src = fr.model_source(name)
        print(f"[{name}] {src.notice}")
        if src.remarks:
            print(f"    note: {src.remarks}")
        if args.check:
            problems = fr_fetch.verify(name, models_root=args.models_root)
            for p in problems:
                print(f"    {p}")
            print(f"    -> {'OK' if not problems else 'FAILED'}\n")
            if problems:
                failed.append(name)
            continue
        try:
            fr_fetch.fetch(name, models_root=args.models_root, force=args.force)
        except fr.ManualDownloadRequired as e:
            manual.append(e)
            failed.append(name)
            print("    -> manual download needed (see below)\n")
            continue
        except (RuntimeError, OSError) as e:
            failed.append(name)
            print(f"    -> FAILED: {e}\n")
            continue
        print("    -> OK (sha256 verified)\n")

    for e in manual:
        print(f"MANUAL DOWNLOAD: {e}\n")
    if failed:
        print(f"{len(failed)} evaluator(s) not ready: {', '.join(failed)}")
        return 1
    print(f"all {len(names)} evaluator(s) ready under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
