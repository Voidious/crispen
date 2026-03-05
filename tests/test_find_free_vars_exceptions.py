from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_except_handler_name():
    # except-handler name is locally bound for the handler body
    src = "try:\n    risky()\nexcept ValueError as exc:\n    handle(exc)\n"
    result = _find_free_vars(src)
    assert "exc" not in result  # locally bound by except clause
    assert "risky" in result
    assert "handle" in result


def test_find_free_vars_except_no_name():
    # bare except without 'as' binding
    src = "try:\n    risky()\nexcept ValueError:\n    pass\n"
    result = _find_free_vars(src)
    assert "risky" in result
