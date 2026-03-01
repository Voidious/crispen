from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import _has_new_undefined_names


def test_has_new_undefined_names_no_new():
    """No new undefined names → returns False."""
    before = "x = 1\ny = x + 1\n"
    after = "x = 1\ny = x + 1\nz = y + 1\n"
    assert _has_new_undefined_names(before, after) is False


def test_has_new_undefined_names_introduced():
    """After introduces an undefined name that before didn't have → returns True."""
    before = "x = 1\n"
    after = "x = undefined_var\n"
    assert _has_new_undefined_names(before, after) is True


def test_has_new_undefined_names_non_undefined_warning():
    """Non-UndefinedName pyflakes warning (e.g. UnusedImport) → returns False."""
    # An unused import produces an UnusedImport warning, not UndefinedName.
    # This exercises the isinstance() False branch inside _Collector.flake.
    before = ""
    after = "import os\n"
    assert _has_new_undefined_names(before, after) is False


def test_has_new_undefined_names_exception():
    """If pyflakes raises an unexpected exception, returns False (safe default)."""
    with patch("pyflakes.api.check", side_effect=RuntimeError("boom")):
        assert _has_new_undefined_names("x = 1\n", "y = 1\n") is False
