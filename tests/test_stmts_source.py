from __future__ import annotations
from crispen.refactors.function_splitter import _head_effective_lines, _stmts_source
from tests.parsers import _parse_func


def test_stmts_source_basic():
    src = "def foo():\n    x = 1\n    y = 2\n    z = 3\n"
    stmts, positions, lines = _parse_func(src)
    result = _stmts_source(stmts[:2], lines, positions)
    assert "x = 1" in result
    assert "y = 2" in result
    assert "z = 3" not in result


def test_stmts_source_empty():
    src = "def foo():\n    x = 1\n"
    _, positions, lines = _parse_func(src)
    assert _stmts_source([], lines, positions) == ""


def test_stmts_source_dedented():
    src = "def foo():\n    x = 1\n    y = 2\n"
    stmts, positions, lines = _parse_func(src)
    result = _stmts_source(stmts, lines, positions)
    # Should be dedented (no leading 4-space indent)
    assert result.startswith("x = 1") or result.startswith("x = 1\n")


def test_head_effective_lines_no_docstring():
    src = "def foo():\n    x = 1\n    y = 2\n    z = 3\n"
    stmts, positions, lines = _parse_func(src)
    # split_idx=2: head=[x,y], last=y at line 3, first=x at line 2 → 3-2+2=3
    result = _head_effective_lines(stmts, 2, positions, False)
    assert result == 3
