from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_for_target_not_free():
    src = "for item in some_list:\n    pass\n"
    result = _find_free_vars(src)
    # item is a store, some_list is a load
    assert "item" not in result
    assert "some_list" in result


def test_find_free_vars_import_not_free():
    src = "import os\npath = os.getcwd()\n"
    result = _find_free_vars(src)
    # os is imported (stored), path is stored
    assert "os" not in result
    assert "path" not in result


def test_find_free_vars_import_from_not_free():
    src = "from os import path\nresult = path.join('a', 'b')\n"
    result = _find_free_vars(src)
    assert "path" not in result


def test_find_free_vars_del_is_store():
    src = "del some_name\n"
    # some_name has Del context (not Load) — not treated as free
    result = _find_free_vars(src)
    assert "some_name" not in result


def test_find_free_vars_augassign_free():
    # weight += 1 reads weight before writing — weight must come from outside
    src = "weight += 1\n"
    result = _find_free_vars(src)
    assert "weight" in result


def test_find_free_vars_augassign_already_defined():
    # weight is unconditionally assigned first, so AugAssign doesn't need it free
    src = "weight = 0\nweight += 1\n"
    result = _find_free_vars(src)
    assert "weight" not in result


def test_find_free_vars_augassign_subscript():
    # data[0] += 1: target is a subscript, data is loaded
    src = "data[0] += 1\n"
    result = _find_free_vars(src)
    assert "data" in result
