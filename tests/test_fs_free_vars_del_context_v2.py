from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_del_context():
    """del statement adds name to stores (else branch for non-Load contexts)."""
    src = "del my_var\n"
    result = _find_free_vars(src)
    assert "my_var" not in result
