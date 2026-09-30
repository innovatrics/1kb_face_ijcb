# SPDX-License-Identifier: MIT
"""Availability checks for optional resources of the adversarial tests."""

from __future__ import annotations


def released_proxy_available() -> bool:
    """Return True if ``weights/liae_proxy.safetensors`` holds the real weights.

    False when the file is missing or is still a git-lfs pointer file. The shared
    ``weights`` marker checks only the codec weights, so the tests that use the
    released proxy call this as well. Needs torch (it imports the proxy module).
    """
    from face1kb.adversarial import proxy  # noqa: PLC0415

    path = proxy.default_proxy_path()
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with open(path, "rb") as f:
        return not f.read(40).startswith(b"version https://git-lfs")
