from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_all_local():
    src = "x = 1\ny = x + 1\n"
    assert _find_free_vars(src) == []


def test_find_free_vars_one_free():
    src = "y = external_var + 1\n"
    result = _find_free_vars(src)
    assert "external_var" in result
    assert "y" not in result


def test_find_free_vars_builtins_excluded():
    src = "print(len([1, 2, 3]))\n"
    result = _find_free_vars(src)
    assert "print" not in result
    assert "len" not in result
