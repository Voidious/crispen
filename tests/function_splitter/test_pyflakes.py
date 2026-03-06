from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import (
    _find_free_vars,
    _has_new_undefined_names,
    FunctionSplitter,
)
from .helpers import _make_long_func
from .mocks import _make_mock_response


def test_find_free_vars_del_context():
    """del statement adds name to stores (else branch for non-Load contexts)."""
    src = "del my_var\n"
    result = _find_free_vars(src)
    assert "my_var" not in result


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


@patch(
    "crispen.refactors.function_splitter._has_new_undefined_names", return_value=True
)
@patch("crispen.llm_client.anthropic")
def test_function_splitter_pyflakes_rejects_output(mock_anthropic, mock_has_undef):
    """If pyflakes detects new undefined names in output, the split is not applied."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["helper"])
    )
    src = _make_long_func(80, "foo")

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=50
        )

    # Pyflakes check returned True → split not applied
    assert splitter.get_rewritten_source() is None
