"""Tests for the CallerUpdater refactor."""

import libcst as cst
from libcst.metadata import MetadataWrapper

from crispen.refactors.caller_updater import (
    CallerUpdater,
    _module_to_str,
    _pascal_to_snake,
    _resolve_relative_module,
    _tmp_name,
)
from crispen.refactors.tuple_dataclass import TransformInfo

# ---------------------------------------------------------------------------
# Shared transform fixture
# ---------------------------------------------------------------------------

_TRANSFORMS = {
    "mypkg.service.get_user": TransformInfo(
        func_name="get_user",
        dataclass_name="GetUserResult",
        field_names=["name", "age", "score"],
    )
}


def _setup_caller_updater(source, file_module, transforms=None, ranges=None):
    if transforms is None:
        transforms = _TRANSFORMS
    if ranges is None:
        ranges = [(1, 100)]
    tree = cst.parse_module(source)
    wrapper = MetadataWrapper(tree)
    cu = CallerUpdater(ranges, transforms, file_module=file_module, source=source)
    return wrapper, cu


def _apply(source, transforms=None, ranges=None, file_module="mypkg.api"):
    wrapper, cu = _setup_caller_updater(source, file_module, transforms, ranges)
    new_tree = wrapper.visit(cu)
    return new_tree.code


def _changes(source, transforms=None, ranges=None, file_module="mypkg.api"):
    wrapper, cu = _setup_caller_updater(source, file_module, transforms, ranges)
    wrapper.visit(cu)
    return cu.changes_made


# ---------------------------------------------------------------------------
# _module_to_str helper
# ---------------------------------------------------------------------------


def test_module_to_str_name():
    assert _module_to_str(cst.Name("foo")) == "foo"


def test_module_to_str_attribute():
    node = cst.Attribute(value=cst.Name("pkg"), attr=cst.Name("service"))
    assert _module_to_str(node) == "pkg.service"


def test_module_to_str_none():
    assert _module_to_str(None) == ""


# ---------------------------------------------------------------------------
# _resolve_relative_module helper
# ---------------------------------------------------------------------------


def test_resolve_relative_one_dot_with_suffix():
    assert _resolve_relative_module("mypkg.api", 1, "service") == "mypkg.service"


def test_resolve_relative_two_dots():
    assert _resolve_relative_module("pkg.sub.api", 2, "utils") == "pkg.utils"


def test_resolve_relative_no_suffix():
    # "from . import something" has no module suffix
    assert _resolve_relative_module("mypkg.api", 1, "") == "mypkg"


# ---------------------------------------------------------------------------
# Happy path: absolute import
# ---------------------------------------------------------------------------

BEFORE_ABSOLUTE = "from mypkg.service import get_user\n" "a, b, c = get_user()\n"


def test_simple_unpack_expanded():
    result = _apply(BEFORE_ABSOLUTE)
    assert "_ = get_user()" in result
    assert "a = _.name" in result
    assert "b = _.age" in result
    assert "c = _.score" in result


def test_original_unpack_removed():
    result = _apply(BEFORE_ABSOLUTE)
    assert "a, b, c = get_user()" not in result


def test_reports_change():
    changes = _changes(BEFORE_ABSOLUTE)
    assert len(changes) == 1
    assert "CallerUpdater" in changes[0]
    assert "GetUserResult" in changes[0]


def test_output_is_valid_python():
    result = _apply(BEFORE_ABSOLUTE)
    compile(result, "<string>", "exec")


# ---------------------------------------------------------------------------
# Relative import
# ---------------------------------------------------------------------------


def test_relative_import_expanded():
    source = "from .service import get_user\n" "a, b, c = get_user()\n"
    result = _apply(source, file_module="mypkg.api")
    assert "_ = get_user()" in result
    assert "a = _.name" in result


# ---------------------------------------------------------------------------
# Aliased import (from X import Y as Z)
# ---------------------------------------------------------------------------


def test_aliased_import_expanded():
    source = "from mypkg.service import get_user as gu\n" "a, b, c = gu()\n"
    result = _apply(source)
    assert "_ = gu()" in result
    assert "a = _.name" in result


# ---------------------------------------------------------------------------
# Star import — must not register any local transform
# ---------------------------------------------------------------------------


def test_star_import_not_registered():
    source = "from mypkg.service import *\n" "a, b, c = get_user()\n"
    result = _apply(source)
    assert result == source  # star import is silently skipped


# ---------------------------------------------------------------------------
# No matching import — _local_transforms stays empty
# ---------------------------------------------------------------------------


def test_unimported_func_not_expanded():
    source = "a, b, c = get_user()\n"
    result = _apply(source)
    assert result == source


# ---------------------------------------------------------------------------
# Non-Name call target (attribute access)
# ---------------------------------------------------------------------------


def test_attribute_call_not_expanded():
    source = "from mypkg.service import get_user\n" "a, b, c = obj.get_user()\n"
    result = _apply(source)
    # call.func is an Attribute, not a Name → skipped
    assert "_crispen_r" not in result


# ---------------------------------------------------------------------------
# Complex tuple target (subscript / attribute in unpacking)
# ---------------------------------------------------------------------------


def test_complex_target_not_expanded():
    source = "from mypkg.service import get_user\n" "a, b[0] = get_user()\n"
    result = _apply(source)
    assert "_crispen_r" not in result


# ---------------------------------------------------------------------------
# Arity mismatch (tuple has different length than field_names)
# ---------------------------------------------------------------------------


def test_wrong_arity_not_expanded():
    source = "from mypkg.service import get_user\n" "a, b = get_user()\n"
    result = _apply(source)
    assert "_crispen_r" not in result


# ---------------------------------------------------------------------------
# Non-tuple target (single variable)
# ---------------------------------------------------------------------------


def test_non_tuple_target_not_expanded():
    source = "from mypkg.service import get_user\n" "result = get_user()\n"
    result = _apply(source)
    assert "_crispen_r" not in result


# ---------------------------------------------------------------------------
# Non-call RHS (simple name, not a call)
# ---------------------------------------------------------------------------


def test_non_call_rhs_not_expanded():
    source = "from mypkg.service import get_user\n" "a, b, c = some_tuple\n"
    result = _apply(source)
    assert "_crispen_r" not in result


# ---------------------------------------------------------------------------
# Multiple targets in assignment (a = b = func())
# ---------------------------------------------------------------------------


def test_multiple_targets_not_expanded():
    source = "from mypkg.service import get_user\n" "x = y = get_user()\n"
    result = _apply(source)
    assert "_crispen_r" not in result


# ---------------------------------------------------------------------------
# Multiple statements on one line (body length > 1)
# ---------------------------------------------------------------------------


def test_multi_stmt_line_not_expanded():
    source = "from mypkg.service import get_user\n" "x = 1; a, b, c = get_user()\n"
    result = _apply(source)
    assert "_crispen_r" not in result


# ---------------------------------------------------------------------------
# Outside the changed range — callers in diffed files are always updated
# ---------------------------------------------------------------------------


def test_outside_range_still_expanded():
    # CallerUpdater updates all callers in the file, not just those in diff ranges.
    source = "from mypkg.service import get_user\n" "a, b, c = get_user()\n"
    result = _apply(source, ranges=[(50, 100)])
    assert "_ = get_user()" in result
    assert "a = _.name" in result


# ---------------------------------------------------------------------------
# Leading whitespace preserved on the first replacement statement
# ---------------------------------------------------------------------------


def test_leading_lines_preserved():
    source = "from mypkg.service import get_user\n" "\n" "a, b, c = get_user()\n"
    result = _apply(source)
    # The blank line before the original statement should still be there
    assert "\n\n" in result


# ---------------------------------------------------------------------------
# CallerUpdater.name() classmethod (line 65)
# ---------------------------------------------------------------------------


def test_caller_updater_name():
    assert CallerUpdater.name() == "CallerUpdater"


# ---------------------------------------------------------------------------
# leave_SimpleStatementLine: transform is None for imported-but-wrong func (line 140)
# ---------------------------------------------------------------------------


def test_imported_func_in_transforms_but_different_call_not_expanded():
    # _local_transforms is non-empty (get_user imported), but we call other_func —
    # so transform is None → hits the line-140 branch.
    source = "from mypkg.service import get_user\n" "a, b, c = other_func()\n"
    result = _apply(source)
    assert "_crispen_r" not in result


# ---------------------------------------------------------------------------
# _pascal_to_snake helper
# ---------------------------------------------------------------------------


def test_pascal_to_snake_multi_word():
    assert _pascal_to_snake("GetUserResult") == "get_user_result"


def test_pascal_to_snake_two_word():
    assert _pascal_to_snake("DataTuple") == "data_tuple"


# ---------------------------------------------------------------------------
# _tmp_name helper
# ---------------------------------------------------------------------------


def test_tmp_name_chooses_underscore():
    assert _tmp_name("GetUserResult", "x = 1\n") == "_"


def test_tmp_name_chooses_result():
    # "_" appears → skip to "_result"
    assert _tmp_name("GetUserResult", "_ = None\n") == "_result"


def test_tmp_name_chooses_snake_case():
    # "_" and "_result" both appear → skip to snake_case
    assert _tmp_name("GetUserResult", "_ = None\n_result = None\n") == "get_user_result"


def test_tmp_name_chooses_crispen_result():
    # First three all appear → fall back to _crispen_result
    source = "_ = None\n_result = None\nget_user_result = None\n"
    assert _tmp_name("GetUserResult", source) == "_crispen_result"


# ---------------------------------------------------------------------------
# Fallback integration: _crispen_result used when first three are taken
# ---------------------------------------------------------------------------


def test_tmp_name_crispen_result_used_in_expansion():
    source = (
        "from mypkg.service import get_user\n"
        "_ = None\n"
        "_result = None\n"
        "get_user_result = None\n"
        "a, b, c = get_user()\n"
    )
    result = _apply(source)
    assert "_crispen_result = get_user()" in result
    assert "a = _crispen_result.name" in result


# ---------------------------------------------------------------------------
# local_transforms: expand same-file private function calls (no import needed)
# ---------------------------------------------------------------------------


def test_local_transforms_expands_without_import():
    """local_transforms pre-populates _local_transforms for same-file private funcs."""
    source = "a, b, c = _make_result()\n"
    local = {
        "_make_result": TransformInfo(
            func_name="_make_result",
            dataclass_name="MakeResultResult",
            field_names=["field_0", "field_1", "field_2"],
        )
    }
    tree = cst.parse_module(source)
    wrapper = MetadataWrapper(tree)
    cu = CallerUpdater(
        [(1, 100)],
        transforms={},
        local_transforms=local,
        source=source,
    )
    result = wrapper.visit(cu).code
    assert "_ = _make_result()" in result
    assert "a = _.field_0" in result
    assert "b = _.field_1" in result
    assert "c = _.field_2" in result


def test_local_transforms_none_leaves_local_transforms_empty():
    """Passing local_transforms=None leaves _local_transforms empty."""
    source = "a, b, c = _make_result()\n"
    tree = cst.parse_module(source)
    wrapper = MetadataWrapper(tree)
    cu = CallerUpdater([(1, 100)], transforms={}, source=source)
    wrapper.visit(cu)
    assert cu._local_transforms == {}
