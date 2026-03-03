from __future__ import annotations
from crispen.refactors.function_splitter import (
    _find_free_vars,
    _has_new_undefined_names,
)


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
