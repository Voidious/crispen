from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_listcomp():
    # list comprehension: loop var is locally scoped
    src = "result = [x * 2 for x in data]\n"
    result = _find_free_vars(src)
    assert "x" not in result  # comprehension target, locally scoped
    assert "data" in result


def test_find_free_vars_listcomp_with_filter():
    # comprehension with 'if' guard: threshold must come from outside
    src = "result = [x for x in data if x > threshold]\n"
    result = _find_free_vars(src)
    assert "x" not in result
    assert "data" in result
    assert "threshold" in result


def test_find_free_vars_dictcomp():
    # dict comprehension: both key and value expressions are walked
    src = "result = {k: v for k, v in pairs}\n"
    result = _find_free_vars(src)
    assert "k" not in result  # tuple target of comprehension
    assert "v" not in result
    assert "pairs" in result
