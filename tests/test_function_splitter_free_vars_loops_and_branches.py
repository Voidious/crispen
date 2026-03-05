from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_for_orelse():
    # for-else: orelse runs when loop completes normally
    src = "for item in data:\n    pass\nelse:\n    fallback()\n"
    result = _find_free_vars(src)
    assert "item" not in result  # for target is locally scoped
    assert "data" in result
    assert "fallback" in result  # used in orelse, not locally defined


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


def test_find_free_vars_conditional_store_is_free():
    # variables only assigned inside a conditional block remain free
    src = "for i in xs:\n    result = f(i)\nprint(result)\n"
    result = _find_free_vars(src)
    assert "result" in result  # conditionally assigned → still free after loop


def test_find_free_vars_for_body_sequential():
    # a variable assigned then used in the same for-body iteration is not free
    src = "for alias in names:\n    name = alias.asname\n    result.add(name)\n"
    result = _find_free_vars(src)
    assert "name" not in result  # assigned before used in same loop body
    assert "names" in result
    assert "result" in result


def test_find_free_vars_if_branch():
    # if-body assignments do not propagate to after the if block
    src = "if cond:\n    x = 1\nelse:\n    y = 2\nz = x + y\n"
    result = _find_free_vars(src)
    assert "cond" in result
    assert "x" in result  # only conditionally defined in if body
    assert "y" in result  # only conditionally defined in else body


def test_find_free_vars_while_loop():
    # while condition is free; while-else is walked
    src = "while running:\n    do_work()\nelse:\n    finalize()\n"
    result = _find_free_vars(src)
    assert "running" in result
    assert "do_work" in result
    assert "finalize" in result
