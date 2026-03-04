from __future__ import annotations
from crispen.refactors.function_splitter import _module_global_names


def test_module_global_names_assignments():
    source = "_CONST = frozenset()\nVALUE: int = 42\n"
    result = _module_global_names(source)
    assert "_CONST" in result
    assert "VALUE" in result


def test_module_global_names_syntax_error():
    result = _module_global_names("def foo(")
    assert result == set()


def test_module_global_names_tuple_assign_target_not_collected():
    # Tuple-unpacking: Assign target is a Tuple node, not a Name → skipped
    source = "a, b = 1, 2\n"
    result = _module_global_names(source)
    assert "a" not in result
    assert "b" not in result


def test_module_global_names_ann_assign_non_name_target_skipped():
    # AnnAssign where target is an Attribute, not a Name → skipped
    source = "Foo.x: int\n"
    result = _module_global_names(source)
    assert "x" not in result
