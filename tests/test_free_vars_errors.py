from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_parse_error():
    assert _find_free_vars("def f(\n  !!") == []
