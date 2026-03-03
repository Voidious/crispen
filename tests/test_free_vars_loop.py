from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_for_body_sequential():
    # a variable assigned then used in the same for-body iteration is not free
    src = "for alias in names:\n    name = alias.asname\n    result.add(name)\n"
    result = _find_free_vars(src)
    assert "name" not in result  # assigned before used in same loop body
    assert "names" in result
    assert "result" in result


def test_find_free_vars_while_loop():
    # while condition is free; while-else is walked
    src = "while running:\n    do_work()\nelse:\n    finalize()\n"
    result = _find_free_vars(src)
    assert "running" in result
    assert "do_work" in result
    assert "finalize" in result
