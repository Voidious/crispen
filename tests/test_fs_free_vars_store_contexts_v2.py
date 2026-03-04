from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_for_target_not_free():
    src = "for item in some_list:\n    pass\n"
    result = _find_free_vars(src)
    # item is a store, some_list is a load
    assert "item" not in result
    assert "some_list" in result


def test_find_free_vars_del_is_store():
    src = "del some_name\n"
    # some_name has Del context (not Load) — not treated as free
    result = _find_free_vars(src)
    assert "some_name" not in result
