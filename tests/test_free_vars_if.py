from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_if_branch():
    # if-body assignments do not propagate to after the if block
    src = "if cond:\n    x = 1\nelse:\n    y = 2\nz = x + y\n"
    result = _find_free_vars(src)
    assert "cond" in result
    assert "x" in result  # only conditionally defined in if body
    assert "y" in result  # only conditionally defined in else body
