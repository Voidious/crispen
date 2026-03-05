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
