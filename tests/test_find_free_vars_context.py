from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_with_target():
    # with-statement target is locally scoped inside the body
    src = "with open(filename) as fp:\n    content = fp.read()\n"
    result = _find_free_vars(src)
    assert "fp" not in result  # with target, locally scoped
    assert "filename" in result  # context_expr is free


def test_find_free_vars_with_no_target():
    # with-statement without 'as' clause
    src = "with ctx_mgr():\n    do_work()\n"
    result = _find_free_vars(src)
    assert "ctx_mgr" in result
    assert "do_work" in result
