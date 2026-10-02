# SPDX-License-Identifier: MIT
"""Small helpers shared by the data-preparation command-line scripts."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from face1kb import config

#: Third-party loggers that report every HTTP request at INFO level.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "huggingface_hub")


def setup_logging(verbose: bool = False) -> None:
    """Configure ``logging`` for a command-line run."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not verbose:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


def output_path(
    explicit: str | os.PathLike | None, default: Path, overwrite: bool
) -> Path:
    """Resolve an output file and refuse to clobber existing data by accident.

    In the ``legacy`` layout the default locations belong to an existing data store,
    so an explicit output path is required there.
    """
    if explicit is None and config.LAYOUT == "legacy":
        raise SystemExit(
            "FACE1KB_LAYOUT=legacy: refusing to write into the legacy store; "
            "pass an explicit output path"
        )
    path = Path(explicit) if explicit is not None else default
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    return path


def output_dir_guard(explicit: bool) -> None:
    """Refuse to write crop folders into a legacy store unless asked explicitly."""
    if config.LAYOUT == "legacy" and not explicit:
        raise SystemExit(
            "FACE1KB_LAYOUT=legacy: refusing to write into the legacy store; "
            "pass --out-root"
        )


def parse_list(value: str | None, cast=str) -> list:
    """Parse a comma-separated option value."""
    if not value:
        return []
    return [cast(v.strip()) for v in value.split(",") if v.strip()]
