# SPDX-License-Identifier: MIT
"""One-CVLface-snapshot-per-process guard (no model files needed)."""

from __future__ import annotations

import sys
import types

import pytest

from face1kb.fr import families


@pytest.fixture()
def loaded(monkeypatch):
    state: dict[str, str] = {}
    monkeypatch.setattr(families, "_CVLFACE_LOADED", state)
    monkeypatch.delitem(sys.modules, "models", raising=False)
    return state


def test_first_load_is_allowed(loaded):
    families._claim_cvlface("ir101", "/m/cvlface/ir101")


def test_same_variant_twice_is_allowed(loaded):
    loaded["ir101"] = "/m/cvlface/ir101"
    families._claim_cvlface("ir101", "/m/cvlface/ir101")


def test_second_variant_is_refused(loaded):
    loaded["ir101"] = "/m/cvlface/ir101"
    with pytest.raises(RuntimeError, match="own process"):
        families._claim_cvlface("vit_b", "/m/cvlface/vit_b")


def test_foreign_models_package_is_refused(loaded, monkeypatch, tmp_path):
    mod = types.ModuleType("models")
    mod.__file__ = str(tmp_path / "project" / "models" / "__init__.py")
    monkeypatch.setitem(sys.modules, "models", mod)
    with pytest.raises(RuntimeError, match="top-level `models`"):
        families._claim_cvlface("ir101", str(tmp_path / "snap"))


def test_template_is_the_arcface_112_template():
    t = families.ARCFACE_TEMPLATE_112
    assert len(t) == 5
    assert t[0] == (38.2946, 51.6963) and t[4] == (70.7299, 92.2041)
