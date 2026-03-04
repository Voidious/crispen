from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_listcomp():
    # list comprehension: loop var is locally scoped
    src = "result = [x * 2 for x in data]\n"
    result = _find_free_vars(src)
    assert "x" not in result  # comprehension target, locally scoped
    assert "data" in result
