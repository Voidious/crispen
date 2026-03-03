from __future__ import annotations
from crispen.refactors.function_splitter import _find_free_vars


def test_find_free_vars_tuple_for_target():
    # tuple-unpacking for target: both names locally scoped
    src = "for a, b in pairs:\n    use(a, b)\n"
    result = _find_free_vars(src)
    assert "a" not in result
    assert "b" not in result
    assert "pairs" in result


def test_find_free_vars_subscript_assign_target():
    # subscript assignment target (e.g. data[0] = 1): _target_names returns {}
    # so nothing is added to definitely_defined, but data is loaded
    src = "data[0] = 1\n"
    result = _find_free_vars(src)
    assert "data" in result  # data is loaded as the subscript base
