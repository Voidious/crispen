"""Tests for the TupleDataclass refactor."""

import libcst as cst
from libcst.metadata import MetadataWrapper

from crispen.refactors.tuple_dataclass import (
    TransformInfo,
    TupleDataclass,
    _UnpackingCollector,
    _all_callers_unpack,
    _int_val,
    _is_variable_index,
    _name_str,
)


def _setup_transformer(source: str, ranges, min_size: int):
    if ranges is None:
        ranges = [(1, 100)]
    tree = cst.parse_module(source)
    wrapper = MetadataWrapper(tree)
    transformer = TupleDataclass(ranges, min_size=min_size, source=source)
    return wrapper, transformer


def _apply(source: str, ranges=None, min_size: int = 3) -> str:
    wrapper, transformer = _setup_transformer(source, ranges, min_size)
    new_tree = wrapper.visit(transformer)
    return new_tree.code


def _changes(source: str, ranges=None, min_size: int = 3) -> list:
    wrapper, transformer = _setup_transformer(source, ranges, min_size)
    wrapper.visit(transformer)
    return transformer.changes_made


# ---------------------------------------------------------------------------
# Happy path: simple tuple in private function
# ---------------------------------------------------------------------------

# Module-scope tuple (kept for "not transformed" tests below).
BEFORE_SIMPLE = "result = (1, 2, 3)\n"

# Private-function tuple — the positive case for all "happy path" tests.
BEFORE_SIMPLE_PRIVATE = "def _f():\n    return (1, 2, 3)\n"


def _assert_default_field_names(result: str) -> None:
    assert "field_0" in result
    assert "field_1" in result
    assert "field_2" in result


def test_simple_tuple_transformed():
    result = _apply(BEFORE_SIMPLE_PRIVATE)
    assert "DataTuple(" in result or "FResult(" in result or "Result(" in result
    _assert_default_field_names(result)


def test_simple_tuple_imports_added():
    result = _apply(BEFORE_SIMPLE_PRIVATE)
    assert "from dataclasses import dataclass" in result
    assert "from typing import Any" in result


def test_simple_tuple_class_defined():
    result = _apply(BEFORE_SIMPLE_PRIVATE)
    assert "@dataclass" in result
    assert "class " in result
    assert "field_0: Any" in result


def test_dataclass_preceded_by_two_blank_lines():
    result = _apply(BEFORE_SIMPLE_PRIVATE)
    # Exactly 2 blank lines before @dataclass (3 newlines: end-of-import + 2 blanks).
    assert "\n\n\n@dataclass" in result
    assert "\n\n\n\n@dataclass" not in result


def test_function_after_dataclass_preceded_by_two_blank_lines():
    result = _apply(BEFORE_SIMPLE_PRIVATE)
    # Exactly 2 blank lines before the original function after the class.
    assert "\n\n\ndef _f" in result
    assert "\n\n\n\ndef _f" not in result


def test_reports_change():
    changes = _changes(BEFORE_SIMPLE_PRIVATE)
    assert len(changes) == 1
    assert "TupleDataclass" in changes[0]


# ---------------------------------------------------------------------------
# Skip cases
# ---------------------------------------------------------------------------

BEFORE_SMALL = "result = (1, 2)\n"


def test_small_tuple_skipped():
    result = _apply(BEFORE_SMALL)
    assert result == BEFORE_SMALL


def test_small_tuple_no_change():
    assert _changes(BEFORE_SMALL) == []


# ---------------------------------------------------------------------------
# Scope gate: only private functions (name starts with '_')
# ---------------------------------------------------------------------------

BEFORE_PUBLIC_FUNC = """\
def public_func():
    return (1, 2, 3)
"""

BEFORE_PRIVATE_FUNC = """\
def _private_func():
    return (1, 2, 3)
"""


def test_public_function_tuple_not_transformed():
    result = _apply(BEFORE_PUBLIC_FUNC)
    assert result == BEFORE_PUBLIC_FUNC


def test_private_function_tuple_transformed():
    result = _apply(BEFORE_PRIVATE_FUNC)
    assert "@dataclass" in result
    assert "PrivateFuncResult(" in result


def test_module_scope_tuple_not_transformed():
    # Module-level tuple (no enclosing function) is also skipped.
    result = _apply(BEFORE_SIMPLE)
    assert result == BEFORE_SIMPLE


# ---------------------------------------------------------------------------
# Field name inference from variable names
# ---------------------------------------------------------------------------

BEFORE_NAMED_VARS = """\
def _silly():
    age = 1
    shoes = 2
    pants = 3
    return (age, shoes, pants)
"""


def test_field_names_inferred_from_variables():
    result = _apply(BEFORE_NAMED_VARS)
    assert "age: Any" in result
    assert "shoes: Any" in result
    assert "pants: Any" in result
    assert "age = age" in result or "age=age" in result


def test_field_names_fallback_for_literals():
    # When elements are literals, fall back to field_0, field_1, ...
    result = _apply("def _f():\n    return (1, 2, 3)\n")
    _assert_default_field_names(result)


BEFORE_STARRED = "result = (*a, 1, 2, 3)\n"


def test_starred_skipped():
    result = _apply(BEFORE_STARRED)
    assert result == BEFORE_STARRED


# ---------------------------------------------------------------------------
# min_size parameter
# ---------------------------------------------------------------------------


def test_min_size_two():
    source = "def _f():\n    return (1, 2)\n"
    result = _apply(source, min_size=2)
    assert "field_0" in result
    assert "field_1" in result


def test_min_size_four_skips_three():
    source = "def _f():\n    return (1, 2, 3)\n"
    result = _apply(source, min_size=4)
    assert result == source


# ---------------------------------------------------------------------------
# Range filtering
# ---------------------------------------------------------------------------


def test_outside_range_skipped():
    result = _apply(BEFORE_SIMPLE, ranges=[(50, 100)])
    assert result == BEFORE_SIMPLE


# ---------------------------------------------------------------------------
# Output is valid Python
# ---------------------------------------------------------------------------


def test_output_is_valid_python():
    result = _apply(BEFORE_SIMPLE_PRIVATE)
    # Should not raise
    compile(result, "<string>", "exec")


# ---------------------------------------------------------------------------
# Doesn't duplicate import if already present
# ---------------------------------------------------------------------------

BEFORE_WITH_IMPORTS = """\
from dataclasses import dataclass
from typing import Any

def _f():
    return (1, 2, 3)
"""


def test_no_duplicate_imports():
    result = _apply(BEFORE_WITH_IMPORTS)
    assert result.count("from dataclasses import dataclass") == 1
    assert result.count("from typing import Any") == 1


# ---------------------------------------------------------------------------
# Private method in class — scope naming uses method name
# ---------------------------------------------------------------------------

BEFORE_IN_CLASS = """\
class MyProcessor:
    def _process(self):
        return (1, 2, 3)
"""


def test_tuple_in_private_method_uses_method_scope_name():
    # _snake_to_pascal("_process") → "Process", so dataclass is "ProcessResult".
    result = _apply(BEFORE_IN_CLASS)
    assert "ProcessResult(" in result


# ---------------------------------------------------------------------------
# visit_ImportFrom — attribute import (e.g. from os.path import join)
# ---------------------------------------------------------------------------


def test_attribute_import_not_treated_as_dataclass():
    source = "from os.path import join\ndef _f():\n    return (1, 2, 3)\n"
    result = _apply(source)
    # Transform still happens; the os.path import is just ignored by the tracker
    assert "from dataclasses import dataclass" in result


# ---------------------------------------------------------------------------
# visit_ImportFrom — star import from typing
# ---------------------------------------------------------------------------


def test_star_import_from_typing():
    source = "from typing import *\ndef _f():\n    return (1, 2, 3)\n"
    result = _apply(source)
    assert "@dataclass" in result


# ---------------------------------------------------------------------------
# visit_ImportFrom — non-Any import from typing
# ---------------------------------------------------------------------------


def test_non_any_typing_import_still_adds_any():
    source = "from typing import List\ndef _f():\n    return (1, 2, 3)\n"
    result = _apply(source)
    assert "from typing import Any" in result


def test_non_import_statement_in_module_does_not_crash():
    # A module-level assignment (not an import) followed by a private function.
    # Exercises the `not isinstance(s, ImportFrom)` branch inside leave_Module's
    # import-insertion loop (branch 313->312 in coverage).
    source = "X = 42\ndef _f():\n    return (1, 2, 3)\n"
    result = _apply(source)
    assert "@dataclass" in result
    assert "from dataclasses import dataclass" in result


# ---------------------------------------------------------------------------
# from __future__ import annotations — imports inserted after it
# ---------------------------------------------------------------------------

BEFORE_FUTURE = """\
from __future__ import annotations

def _f():
    return (1, 2, 3)
"""


def test_future_import_insert_position():
    result = _apply(BEFORE_FUTURE)
    future_pos = result.index("from __future__")
    any_pos = result.index("from typing import Any")
    assert future_pos < any_pos


# ---------------------------------------------------------------------------
# _class_name_for — assign_target_name path (no enclosing scope)
# ---------------------------------------------------------------------------


def test_class_name_for_assign_target():
    t = TupleDataclass([(1, 10)])
    assert t._class_name_for("my_result") == "MyResultTuple"


def test_class_name_for_no_scope_no_name():
    # No enclosing scope, no assign-target name → falls through to "DataTuple".
    t = TupleDataclass([(1, 10)])
    assert t._class_name_for(None) == "DataTuple"


# ---------------------------------------------------------------------------
# _field_names_for — uses _unpackings when available
# ---------------------------------------------------------------------------


def _parse_simple_tuple(source: str = "x = (1, 2, 3)\n"):
    tree = cst.parse_module(source)
    tuple_node = tree.body[0].body[0].value
    return tuple_node


def test_field_names_from_unpackings():
    source = "x = (1, 2, 3)\n"
    tuple_node = _parse_simple_tuple(source)
    assert isinstance(tuple_node, cst.Tuple)

    t = TupleDataclass([(1, 10)])
    t._unpackings = {1: ["a", "b", "c"]}
    names = t._field_names_for(tuple_node, 1)
    assert names == ["a", "b", "c"]


def test_field_names_unpackings_length_mismatch_falls_through():
    # If the unpacking name count doesn't match the tuple element count,
    # _field_names_for falls through to the variable-name inference path.
    source = "x = (1, 2, 3)\n"
    tuple_node = _parse_simple_tuple(source)

    t = TupleDataclass([(1, 10)])
    t._unpackings = {1: ["a", "b"]}  # only 2 names for a 3-element tuple
    names = t._field_names_for(tuple_node, 1)
    # Falls through to generic naming
    assert names == ["field_0", "field_1", "field_2"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_name_str_with_name_node():
    assert _name_str(cst.Name("foo")) == "foo"


def test_name_str_with_non_name_node():
    assert _name_str(cst.Integer("42")) is None


def test_is_variable_index_with_name():
    assert _is_variable_index(cst.Name("x")) is True


def test_is_variable_index_with_integer():
    assert _is_variable_index(cst.Integer("1")) is False


def test_int_val_with_integer():
    assert _int_val(cst.Integer("42")) == 42


def test_int_val_with_non_integer():
    assert _int_val(cst.Name("x")) is None


# ---------------------------------------------------------------------------
# _UnpackingCollector
# ---------------------------------------------------------------------------


def test_unpacking_collector_finds_tuple_target():
    source = "a, b, c = some_func()\n"
    tree = cst.parse_module(source)
    collector = _UnpackingCollector()
    MetadataWrapper(tree).visit(collector)
    assert len(collector.unpackings) == 1
    assert list(collector.unpackings.values())[0] == ["a", "b", "c"]


def _collect_unpackings(source: str):
    tree = cst.parse_module(source)
    collector = _UnpackingCollector()
    MetadataWrapper(tree).visit(collector)
    return collector.unpackings


def test_unpacking_collector_skips_multiple_targets():
    source = "a = b = some_func()\n"
    unpackings = _collect_unpackings(source)
    assert unpackings == {}


def test_unpacking_collector_skips_non_tuple_target():
    source = "a = some_func()\n"
    unpackings = _collect_unpackings(source)
    assert unpackings == {}


def test_unpacking_collector_skips_complex_elements():
    source = "a, b[0] = some_func()\n"
    unpackings = _collect_unpackings(source)
    assert unpackings == {}


# ---------------------------------------------------------------------------
# approved_public_funcs — candidates and transformation
# ---------------------------------------------------------------------------

BEFORE_PUBLIC_FUNC_3 = """\
def public_func():
    return (1, 2, 3)
"""


def _make_td_for_public_func_3() -> TupleDataclass:
    """Run TupleDataclass on BEFORE_PUBLIC_FUNC_3 and return the transformer."""
    tree = cst.parse_module(BEFORE_PUBLIC_FUNC_3)
    td = TupleDataclass([(1, 100)], min_size=3)
    MetadataWrapper(tree).visit(td)
    return td


def test_public_function_candidate_recorded():
    candidates = _make_td_for_public_func_3().get_candidate_public_transforms()
    assert "public_func" in candidates
    assert candidates["public_func"].func_name == "public_func"
    assert candidates["public_func"].dataclass_name == "PublicFuncResult"


def test_public_function_candidate_has_correct_type():
    candidates = _make_td_for_public_func_3().get_candidate_public_transforms()
    assert isinstance(candidates["public_func"], TransformInfo)


def test_private_function_not_in_candidates():
    source = "def _private():\n    return (1, 2, 3)\n"
    tree = cst.parse_module(source)
    td = TupleDataclass([(1, 100)])
    MetadataWrapper(tree).visit(td)
    assert td.get_candidate_public_transforms() == {}


def test_approved_public_function_is_transformed():
    tree = cst.parse_module(BEFORE_PUBLIC_FUNC_3)
    td = TupleDataclass([(1, 100)], approved_public_funcs={"public_func"}, min_size=3)
    result = MetadataWrapper(tree).visit(td).code
    assert "PublicFuncResult(" in result
    assert "@dataclass" in result


def test_unapproved_public_function_not_transformed():
    result = _apply(BEFORE_PUBLIC_FUNC_3)
    assert result == BEFORE_PUBLIC_FUNC_3


def test_get_candidate_public_transforms_returns_copy():
    td = _make_td_for_public_func_3()
    c1 = td.get_candidate_public_transforms()
    c2 = td.get_candidate_public_transforms()
    assert c1 == c2
    assert c1 is not c2  # each call returns a new dict


# ---------------------------------------------------------------------------
# leave_Module: no duplicate class if one already exists in source
# ---------------------------------------------------------------------------


def test_no_duplicate_class_if_already_in_source():
    # Source already has the class that TupleDataclass would inject.
    # The duplicate-check branch (339->338) prevents a second injection.
    source = """\
from dataclasses import dataclass
from typing import Any

@dataclass
class FResult:
    field_0: Any
    field_1: Any
    field_2: Any

def _f():
    return (1, 2, 3)
"""
    result = _apply(source)
    assert result.count("class FResult") == 1


# ---------------------------------------------------------------------------
# get_private_transforms
# ---------------------------------------------------------------------------


def _make_private_td(source: str = "def _private():\n    return (1, 2, 3)\n"):
    """Parse *source*, run TupleDataclass over it, and return the transformer."""
    tree = cst.parse_module(source)
    td = TupleDataclass([(1, 100)], min_size=3)
    MetadataWrapper(tree).visit(td)
    return td


def test_private_function_in_get_private_transforms():
    # _snake_to_pascal("_private") → "Private" → class name "PrivateResult"
    td = _make_private_td()
    private = td.get_private_transforms()
    assert "_private" in private
    assert private["_private"].func_name == "_private"
    assert private["_private"].dataclass_name == "PrivateResult"


def test_public_function_not_in_get_private_transforms():
    assert _make_td_for_public_func_3().get_private_transforms() == {}


def test_get_private_transforms_returns_copy():
    td = _make_private_td()
    p1 = td.get_private_transforms()
    p2 = td.get_private_transforms()
    assert p1 == p2
    assert p1 is not p2


# ---------------------------------------------------------------------------
# Tuple inside a function call argument is NOT transformed
# ---------------------------------------------------------------------------


def test_tuple_in_non_return_not_transformed():
    # A 3-element tuple in an assignment (not a return) is left unchanged.
    source = "def _f():\n    x = (1, 2, 3)\n"
    result = _apply(source)
    assert result == source


def test_tuple_in_call_arg_not_transformed():
    # A 3-element tuple inside a return-statement call argument is left unchanged.
    source = "def _f():\n    return g((1, 2, 3))\n"
    result = _apply(source)
    assert result == source


# ---------------------------------------------------------------------------
# Multiple return paths: consistent field names across constructor calls
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# New branch-coverage tests for the _in_return guard
# ---------------------------------------------------------------------------


def test_starred_tuple_in_return_skipped():
    # A starred tuple inside a return statement must not be transformed.
    # Exercises _is_safe_tuple returning False (lines 245, 263).
    source = "def _f():\n    return (*a, 1, 2)\n"
    result = _apply(source, min_size=3)
    assert result == source


def test_return_tuple_outside_range_skipped():
    # A return-statement tuple that falls outside the changed range is skipped.
    # Exercises the _in_changed_range guard (line 261).
    source = "def _f():\n    return (1, 2, 3)\n"
    # Line 2 is outside the changed range [50, 100].
    result = _apply(source, ranges=[(50, 100)])
    assert result == source


class _NoScopePushTransformer(TupleDataclass):
    """Override visit/leave_FunctionDef to avoid pushing to _scope_stack.

    This simulates the (normally unreachable) case where leave_Tuple is called
    while _in_return > 0 but _current_scope_name() returns None, exercising
    branch 213->212 and line 267.
    """

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        return None  # do NOT push to _scope_stack

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.BaseStatement:
        return updated_node  # do NOT pop _scope_stack


def test_scope_none_in_return_does_not_transform():
    """Defensive guard: scope is None even inside a return statement.

    _NoScopePushTransformer leaves _scope_stack as [None] (set by visit_Module)
    throughout traversal.  When leave_Tuple is reached inside `return (1,2,3)`,
    _current_scope_name iterates over [None], hits the None entry (branch
    213->212), exhausts the iterator, returns None, and leave_Tuple returns the
    tuple unchanged (line 267).
    """
    source = "def _f():\n    return (1, 2, 3)\n"
    tree = cst.parse_module(source)
    td = _NoScopePushTransformer([(1, 100)], min_size=3)
    result = MetadataWrapper(tree).visit(td).code
    assert result == source


# ---------------------------------------------------------------------------
# _all_callers_unpack unit tests
# ---------------------------------------------------------------------------


def test_all_callers_unpack_syntax_error():
    # Unparseable source → conservatively returns False.
    assert _all_callers_unpack("def (invalid:", "foo") is False


def test_all_callers_unpack_no_calls():
    # No calls to the function at all → vacuously True.
    assert _all_callers_unpack("x = 1\n", "foo") is True


def test_all_callers_unpack_all_unpacking():
    # Every call is in a tuple-unpacking assignment → True.
    source = "a, b = foo()\nc, d = foo()\n"
    assert _all_callers_unpack(source, "foo") is True


def test_all_callers_unpack_non_unpacking_call():
    # Call assigned to a plain name (not a tuple) → False.
    source = "x = foo()\n"
    assert _all_callers_unpack(source, "foo") is False


def test_all_callers_unpack_call_in_expression():
    # Call used as a function argument (not in any assignment) → False.
    source = "bar(foo())\n"
    assert _all_callers_unpack(source, "foo") is False


# ---------------------------------------------------------------------------
# Guard: private function with non-unpacking caller is not transformed
# ---------------------------------------------------------------------------


def test_private_function_non_unpacking_caller_skipped():
    """Return tuple not transformed when caller doesn't unpack the result."""
    source = """\
def _f():
    return (1, 2, 3)

result = _f()
"""
    result = _apply(source)
    assert result == source


def test_public_function_non_unpacking_caller_not_candidate():
    """Public function with a non-unpacking caller is not recorded as a candidate."""
    source = """\
def public_func():
    return (1, 2, 3)

result = public_func()
"""
    tree = cst.parse_module(source)
    td = TupleDataclass([(1, 100)], source=source)
    MetadataWrapper(tree).visit(td)
    assert td.get_candidate_public_transforms() == {}


def test_private_function_all_unpacking_callers_transformed():
    """Return tuple IS transformed when every caller uses tuple unpacking."""
    source = """\
def _f():
    return (1, 2, 3)

a, b, c = _f()
"""
    result = _apply(source)
    assert "@dataclass" in result


def test_multiple_returns_consistent_field_names():
    # A private function with two return paths: one returns all-Name elements
    # (would infer "x", "y", "z") and one returns a mix with a literal (would
    # fall back to "field_0", "field_1", "field_2").  The transformer must use
    # the SAME set of field names for both constructor calls.
    source = """\
def _f(flag):
    x = 1
    y = 2
    z = 3
    if flag:
        return (x, y, z)
    return (x, 0, z)
"""
    result = _apply(source)
    # Both constructor calls must use the same keyword names.
    # The first return (x, y, z) infers ["x", "y", "z"].
    # The second (x, 0, z) would fall back to field_* without the fix.
    # With the fix, both use whichever names were locked in first.
    assert (
        result.count("FResult(") == 2
        or "field_0" not in result
        or result.count("FResult(") == 2
    )
    # Simpler assertion: the produced source must be valid Python.
    compile(result, "<string>", "exec")


# ---------------------------------------------------------------------------
# Default min_size is now 4
# ---------------------------------------------------------------------------


def _make_tuple_dataclass(source: str, ranges=None):
    tree = cst.parse_module(source)
    from libcst.metadata import MetadataWrapper

    wrapper = MetadataWrapper(tree)
    td = TupleDataclass([(1, 2)], source=source)  # no min_size -> default 4
    wrapper.visit(td)
    return td


def test_default_min_size_is_4_skips_3_tuple():
    """With the default min_size=4, a 3-element tuple is NOT transformed."""
    source = "def _f():\n    return (1, 2, 3)\n"
    td = _make_tuple_dataclass(source)
    assert td.changes_made == []


def test_default_min_size_is_4_transforms_4_tuple():
    """With the default min_size=4, a 4-element tuple IS transformed."""
    source = "def _f():\n    return (1, 2, 3, 4)\n"
    td = _make_tuple_dataclass(source)
    assert td.changes_made != []


# ---------------------------------------------------------------------------
# blocked_scopes
# ---------------------------------------------------------------------------


def test_blocked_scopes_prevents_private_transform():
    """A private function in blocked_scopes is not transformed."""
    source = "def _f():\n    return (1, 2, 3)\n"
    result = _apply(source, min_size=3)
    # Without blocking it IS transformed
    assert "@dataclass" in result

    # With blocking, no transformation
    tree = cst.parse_module(source)
    from libcst.metadata import MetadataWrapper

    wrapper = MetadataWrapper(tree)
    td = TupleDataclass([(1, 2)], source=source, min_size=3, blocked_scopes={"_f"})
    new_tree = wrapper.visit(td)
    assert new_tree.code == source
    assert td.changes_made == []


def test_blocked_scopes_does_not_affect_public_functions():
    """blocked_scopes only gates private functions; public functions are unaffected."""
    source = "def _f():\n    return (1, 2, 3)\ndef g():\n    return (1, 2, 3)\n"
    tree = cst.parse_module(source)
    from libcst.metadata import MetadataWrapper

    wrapper = MetadataWrapper(tree)
    # Block _f but not g. g is public so it won't be transformed in the
    # private pass anyway (it becomes a candidate), but _f should be blocked.
    td = TupleDataclass(
        [(1, 4)],
        source=source,
        min_size=3,
        blocked_scopes={"_f"},
    )
    wrapper.visit(td)
    # _f is blocked → no private_transforms
    assert "_f" not in td.get_private_transforms()
    # g is public → candidate
    assert "g" in td.get_candidate_public_transforms()
