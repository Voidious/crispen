from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_lambda_param_not_free():
    # lambda parameter must not appear as a free variable
    src = "result = sorted(tasks, key=lambda t: t.name)\n"
    result = _find_free_vars(src)
    assert "t" not in result
    assert "tasks" in result


def test_find_free_vars_lambda_vararg_not_free():
    # *args in lambda body — args is the vararg, not free
    src = "f = lambda *args: list(args)\n"
    result = _find_free_vars(src)
    assert "args" not in result


def test_find_free_vars_lambda_kwarg_not_free():
    # **kw in lambda body — kw is the kwarg, not free
    src = "f = lambda **kw: kw\n"
    result = _find_free_vars(src)
    assert "kw" not in result


def test_find_free_vars_lambda_default_outer_scope():
    # Default values are evaluated in the enclosing scope, not the lambda scope.
    src = "f = lambda x=outer_val: x\n"
    result = _find_free_vars(src)
    assert "outer_val" in result  # evaluated in outer scope → free
    assert "x" not in result  # lambda param → not free


def test_find_free_vars_lambda_kw_default_none_entry():
    # keyword-only param without a default: kw_defaults has a None entry
    # lambda *, x, y=outer_val: x+y → kw_defaults=[None, outer_val_node]
    src = "f = lambda *, x, y=outer_val: x + y\n"
    result = _find_free_vars(src)
    assert "x" not in result  # kwonly param → not free
    assert "y" not in result  # kwonly param → not free
    assert "outer_val" in result  # kw_default evaluated in outer scope → free
