from __future__ import annotations
from crispen.refactors.function_splitter import _choose_best_split
from tests.fs_ast_parse_helpers_v2 import _parse_func


def test_choose_best_split_empty_splits_returns_none():
    # No valid split candidates → None returned
    src = "def foo():\n    x = 1\n    y = 2\n"
    stmts, positions, lines = _parse_func(src)
    result = _choose_best_split(stmts, [], lines, positions, [])
    assert result is None
