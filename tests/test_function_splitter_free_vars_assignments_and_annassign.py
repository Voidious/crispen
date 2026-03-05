from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_subscript_assign_target():
    # subscript assignment target (e.g. data[0] = 1): _target_names returns {}
    # so nothing is added to definitely_defined, but data is loaded
    src = "data[0] = 1\n"
    result = _find_free_vars(src)
    assert "data" in result  # data is loaded as the subscript base


def test_find_free_vars_annassign_with_value():
    # annotated assignment with value: name is definitely defined afterwards
    src = "x: int = 5\ny = x + 1\n"
    result = _find_free_vars(src)
    assert "x" not in result
    assert "y" not in result


def test_find_free_vars_annassign_no_value():
    # annotation without assignment: x is NOT definitely defined
    src = "x: int\ny = x + 1\n"
    result = _find_free_vars(src)
    assert "x" in result  # not assigned, so it is free


def test_find_free_vars_annassign_non_name_target():
    # annotated assignment where target is not a plain Name
    src = "obj.attr: int = 5\n"
    result = _find_free_vars(src)
    assert "obj" in result  # obj is loaded to set the attribute


def test_find_free_vars_conditional_store_is_free():
    # variables only assigned inside a conditional block remain free
    src = "for i in xs:\n    result = f(i)\nprint(result)\n"
    result = _find_free_vars(src)
    assert "result" in result  # conditionally assigned → still free after loop
