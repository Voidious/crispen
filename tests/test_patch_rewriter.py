"""Tests for patch_rewriter — 100% branch coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch as mock_patch

import libcst as cst

from crispen.config import CrispenConfig
from crispen.llm_client import LLMCallResult
from crispen.patch_rewriter import (
    _ConstRef,
    _FLContext,
    RewriteAccumulator,
    _CgIndex,
    _CG_CANDIDATES_LLM_THRESHOLD,
    _CG_MAX_DEPTH,
    _CG_MAX_MODULES,
    _apply_cross_file_const_updates,
    _expand_module_terminals,
    _build_attr_const_map,
    _build_classify_prompt,
    _build_const_map,
    _build_context_message,
    _build_func_verify_prompt,
    _build_local_const_map,
    _build_no_change_verify_prompt,
    _build_rewrite_func_prompt,
    _build_rewrite_verify_prompt,
    _callgraph_update_file,
    _candidates_check,
    _patch_strings_in_text,
    _rewrite_candidates_check,
    _cg_build_index,
    _cg_collect_called_names,
    _cg_collect_defined_names,
    _cg_collect_func_body_calls,
    _cg_file_to_module_and_package,
    _cg_parse_imports,
    _cg_resolve_call_to_import,
    _compiles,
    _extract_migration_reminder,
    _extract_patch_lookup,
    _build_rename_guard_sets,
    _extract_still_imported_names,
    _is_bad_rename,
    _find_test_functions_to_update,
    _find_with_patch_paths_in_body,
    _import_header,
    _is_patch_call,
    _get_external_import_names,
    _name_reference_map,
    _matches_any,
    _process_file_source,
    _resolve_forking_path_candidates,
    _resolve_forking_path_via_callgraph,
    _resolve_import_to_file,
    _splice_function,
    _restore_const_refs,
    _get_const_votes_from_rewrite,
    _substitute_consts_in_func_text,
    apply_patch_callgraph,
    apply_patch_rewrite,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(tool_input=None) -> LLMCallResult:
    return LLMCallResult(
        tool_input=tool_input, elapsed=0.0, input_tokens=0, output_tokens=0
    )


def _truncated_ok() -> LLMCallResult:
    """Simulate a truncated verify response (tool_input=None, truncated=True)."""
    return LLMCallResult(
        tool_input=None, elapsed=0.0, input_tokens=0, output_tokens=0, truncated=True
    )


def _make_fl_ctx(**kwargs) -> _FLContext:
    defaults = dict(
        filepath="/proj/pkg/big.py",
        old_module="pkg.big",
        original_source="class A: pass\nclass B: pass\n",
        modified_source="from .sub_a import A\nfrom .sub_b import B\n",
        new_files={"sub_a.py": "class A: pass\n", "sub_b.py": "class B: pass\n"},
        new_module_paths={"sub_a.py": "pkg.sub_a", "sub_b.py": "pkg.sub_b"},
        entity_to_target={"A": "sub_a.py", "B": "sub_b.py"},
        forking_old_paths={"pkg.big.A", "pkg.big.B"},
    )
    defaults.update(kwargs)
    return _FLContext(**defaults)


_CFG = CrispenConfig(patch_update_retries=1)
_CFG_NO_LLM_VERIFY = CrispenConfig(patch_update_retries=1, llm_verify_retries=0)
_FORKING_PATHS = {"crispen.before.X"}
_SRC_WITH_PATCH = '@patch("crispen.before.X")\ndef test_f(mock_x):\n    pass\n'

_PATCH_GET_KEY = "crispen.patch_rewriter.get_api_key"
_PATCH_MAKE_CLIENT = "crispen.patch_rewriter.make_client"
_PATCH_CALL_TOOL = "crispen.patch_rewriter.call_with_tool"

# Shorthand classify tool_inputs.
_CLASSIFY_RENAME = {
    "needs_rewrite": False,
    "patch_renames": {"crispen.before.X": "crispen.after.X"},
}
_CLASSIFY_NO_CHANGE = {"needs_rewrite": False, "patch_renames": {}}
_CLASSIFY_REWRITE = {"needs_rewrite": True}
_VERIFY_OK = {"correct": True, "issue": ""}
_VERIFY_REJECT = {"correct": False, "issue": "wrong path"}
_VERIFY_REJECT_WITH_CORRECTIONS = {
    "correct": False,
    "issue": "wrong path",
    "corrections": {"crispen.before.X": "crispen.after.X"},
}
_REWRITE_VERIFY_OK = {"correct": True, "issue": ""}
_REWRITE_VERIFY_REJECT = {"correct": False, "issue": "wrong mock setup"}


# ---------------------------------------------------------------------------
# _is_patch_call
# ---------------------------------------------------------------------------


def test_is_patch_call_name_match():
    call_node = cst.parse_expression('patch("foo")')
    assert _is_patch_call(call_node) is True


def test_is_patch_call_attribute_match():
    call_node = cst.parse_expression('mock.patch("foo")')
    assert _is_patch_call(call_node) is True


def test_is_patch_call_other_name():
    call_node = cst.parse_expression('other("foo")')
    assert _is_patch_call(call_node) is False


# ---------------------------------------------------------------------------
# _matches_any
# ---------------------------------------------------------------------------


def test_matches_any_exact():
    assert _matches_any("a.b.C", {"a.b.C"}) is True


def test_matches_any_prefix():
    assert _matches_any("a.b.C.method", {"a.b.C"}) is True


def test_matches_any_near_miss():
    # "a.b.CExtra" should NOT match "a.b.C"
    assert _matches_any("a.b.CExtra", {"a.b.C"}) is False


def test_matches_any_no_match():
    assert _matches_any("x.y.Z", {"a.b.C"}) is False


# ---------------------------------------------------------------------------
# _compiles
# ---------------------------------------------------------------------------


def test_compiles_valid():
    assert _compiles("x = 1\n") is True


def test_compiles_invalid():
    assert _compiles("def f(:\n    pass\n") is False


# ---------------------------------------------------------------------------
# _find_test_functions_to_update
# ---------------------------------------------------------------------------


def test_find_empty_old_paths():
    src = '@patch("crispen.before.X")\ndef test_f(): pass\n'
    assert _find_test_functions_to_update(src, set()) == []


def test_find_parse_error():
    assert _find_test_functions_to_update("def f(:\n", {"crispen.before.X"}) == []


def test_find_no_match():
    src = '@patch("other.mod.Y")\ndef test_f(): pass\n'
    assert _find_test_functions_to_update(src, {"crispen.before.X"}) == []


def test_find_match_exact():
    src = '@patch("crispen.before.X")\ndef test_f(): pass\n'
    result = _find_test_functions_to_update(src, {"crispen.before.X"})
    assert len(result) == 1
    assert result[0].function_name == "test_f"
    assert "crispen.before.X" in result[0].old_patch_paths


def test_find_match_prefix():
    src = '@patch("crispen.before.X.method")\ndef test_f(): pass\n'
    result = _find_test_functions_to_update(src, {"crispen.before.X"})
    assert len(result) == 1
    assert "crispen.before.X.method" in result[0].old_patch_paths


def test_find_not_a_call_decorator():
    # @patch used as a bare name (no parentheses), not a Call node.
    src = "@patch\ndef test_f(): pass\n"
    assert _find_test_functions_to_update(src, {"crispen.before.X"}) == []


def test_find_no_args():
    src = "@patch()\ndef test_f(): pass\n"
    assert _find_test_functions_to_update(src, {"crispen.before.X"}) == []


def test_find_arg_not_simple_string():
    # @patch(some_variable) — first arg is a Name, not a SimpleString.
    src = "@patch(some_var)\ndef test_f(): pass\n"
    assert _find_test_functions_to_update(src, {"crispen.before.X"}) == []


def test_find_prefixed_string():
    # b"..." — raw[0] is 'b', not a quote character.
    src = '@patch(b"crispen.before.X")\ndef test_f(): pass\n'
    assert _find_test_functions_to_update(src, {"crispen.before.X"}) == []


def test_find_triple_quoted():
    src = '@patch("""crispen.before.X""")\ndef test_f(): pass\n'
    assert _find_test_functions_to_update(src, {"crispen.before.X"}) == []


def test_find_not_patch_name():
    # @decorate("crispen.before.X") — attribute name is not "patch".
    src = '@decorate("crispen.before.X")\ndef test_f(): pass\n'
    assert _find_test_functions_to_update(src, {"crispen.before.X"}) == []


def test_find_attribute_patch():
    # @mock.patch("crispen.before.X") — Attribute form.
    src = '@mock.patch("crispen.before.X")\ndef test_f(): pass\n'
    result = _find_test_functions_to_update(src, {"crispen.before.X"})
    assert len(result) == 1
    assert result[0].function_name == "test_f"


def test_find_multiple_functions():
    src = (
        '@patch("crispen.before.X")\ndef test_a(): pass\n\n'
        '@patch("crispen.before.Y")\ndef test_b(): pass\n'
    )
    result = _find_test_functions_to_update(
        src, {"crispen.before.X", "crispen.before.Y"}
    )
    assert {f.function_name for f in result} == {"test_a", "test_b"}


def test_find_full_text_includes_decorator():
    src = '@patch("crispen.before.X")\ndef test_f():\n    pass\n'
    result = _find_test_functions_to_update(src, {"crispen.before.X"})
    assert '@patch("crispen.before.X")' in result[0].full_text
    assert "def test_f" in result[0].full_text


def test_find_start_end_lines():
    # line 1: # header, line 2: @patch..., line 3: def test_f, line 4: pass
    src = "# header\n" '@patch("crispen.before.X")\n' "def test_f():\n" "    pass\n"
    result = _find_test_functions_to_update(src, {"crispen.before.X"})
    assert result[0].start_line == 2  # @patch line (first decorator)
    assert result[0].end_line == 4  # last line of body


def test_find_body_with_patch_no_decorator():
    # Function has no @patch decorator but uses ``with patch(...)`` in the body.
    src = (
        "def test_f():\n" '    with patch("crispen.before.X") as m:\n' "        pass\n"
    )
    result = _find_test_functions_to_update(src, {"crispen.before.X"})
    assert len(result) == 1
    assert result[0].function_name == "test_f"
    assert "crispen.before.X" in result[0].old_patch_paths
    # start_line should be the ``def`` line (no decorators).
    assert result[0].start_line == 1


def test_find_body_with_patch_combined_with_decorator():
    # Function has both an @patch decorator and a body-level with patch(...).
    src = (
        '@patch("crispen.before.Y")\n'
        "def test_f(mock_y):\n"
        '    with patch("crispen.before.X") as m:\n'
        "        pass\n"
    )
    result = _find_test_functions_to_update(
        src, {"crispen.before.X", "crispen.before.Y"}
    )
    assert len(result) == 1
    paths = result[0].old_patch_paths
    assert "crispen.before.X" in paths
    assert "crispen.before.Y" in paths


# ---------------------------------------------------------------------------
# _find_with_patch_paths_in_body
# ---------------------------------------------------------------------------


def test_body_scan_syntax_error():
    assert _find_with_patch_paths_in_body("def f(:\n", {"old.X"}, {}, {}) == []


def test_body_scan_no_funcdef():
    # Parsed text has no FunctionDef at the top level.
    assert _find_with_patch_paths_in_body("x = 1\n", {"old.X"}, {}, {}) == []


def test_body_scan_simple_match():
    src = 'def test_f():\n    with patch("old.X") as m:\n        pass\n'
    result = _find_with_patch_paths_in_body(src, {"old.X"}, {}, {})
    assert result == ["old.X"]


def test_body_scan_no_match():
    src = 'def test_f():\n    with patch("other.Y") as m:\n        pass\n'
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_attribute_patch():
    # ``with mock.patch(...)`` form.
    src = 'def test_f():\n    with mock.patch("old.X") as m:\n        pass\n'
    result = _find_with_patch_paths_in_body(src, {"old.X"}, {}, {})
    assert result == ["old.X"]


def test_body_scan_not_patch_call():
    src = 'def test_f():\n    with other("old.X") as m:\n        pass\n'
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_no_args():
    src = "def test_f():\n    with patch() as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_non_call_context_manager():
    # Context manager is a plain Name, not a Call.
    src = "def test_f():\n    with ctx as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_non_string_arg():
    # First arg is a Call expression (not string/Name/Attribute).
    src = "def test_f():\n    with patch(get_target()) as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_name_const_match():
    const_map = {"MY_TARGET": ("old.X", "/file.py")}
    src = "def test_f():\n    with patch(MY_TARGET) as m:\n        pass\n"
    result = _find_with_patch_paths_in_body(src, {"old.X"}, const_map, {})
    assert result == ["old.X"]


def test_body_scan_name_const_no_match():
    # Constant value doesn't match old_paths.
    const_map = {"MY_TARGET": ("other.Y", "/file.py")}
    src = "def test_f():\n    with patch(MY_TARGET) as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, const_map, {}) == []


def test_body_scan_name_not_in_const_map():
    src = "def test_f():\n    with patch(unknown_var) as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_attr_const_match():
    attr_const_map = {"consts": {"TARGET": ("old.X", "/consts.py")}}
    src = "def test_f():\n    with patch(consts.TARGET) as m:\n        pass\n"
    result = _find_with_patch_paths_in_body(src, {"old.X"}, {}, attr_const_map)
    assert result == ["old.X"]


def test_body_scan_attr_const_module_not_in_map():
    src = "def test_f():\n    with patch(unknown_mod.X) as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_attr_const_attr_not_in_map():
    attr_const_map = {"consts": {"OTHER": ("old.X", "/consts.py")}}
    src = "def test_f():\n    with patch(consts.MISSING) as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, attr_const_map) == []


def test_body_scan_attr_const_no_match():
    # Attribute constant value doesn't match old_paths.
    attr_const_map = {"consts": {"TARGET": ("other.Y", "/consts.py")}}
    src = "def test_f():\n    with patch(consts.TARGET) as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, attr_const_map) == []


def test_body_scan_nested_funcdef_excluded():
    # ``with patch(...)`` inside a nested function should NOT trigger inclusion of
    # the outer function — the nested function is its own unit.
    src = (
        "def test_outer():\n"
        "    def inner():\n"
        '        with patch("old.X") as m:\n'
        "            pass\n"
    )
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_multiple_with_items():
    # ``with patch("a") as m, patch("b") as n:`` — both items should be found.
    src = (
        "def test_f():\n"
        '    with patch("old.X") as m, patch("old.Y") as n:\n'
        "        pass\n"
    )
    result = _find_with_patch_paths_in_body(src, {"old.X", "old.Y"}, {}, {})
    assert set(result) == {"old.X", "old.Y"}


def test_body_scan_async_with():
    src = 'async def test_f():\n    async with patch("old.X") as m:\n        pass\n'
    result = _find_with_patch_paths_in_body(src, {"old.X"}, {}, {})
    assert result == ["old.X"]


def test_body_scan_nested_in_if():
    # ``with patch(...)`` inside an ``if`` block should still be found.
    src = (
        "def test_f():\n"
        "    if True:\n"
        '        with patch("old.X") as m:\n'
        "            pass\n"
    )
    result = _find_with_patch_paths_in_body(src, {"old.X"}, {}, {})
    assert result == ["old.X"]


# ---------------------------------------------------------------------------
# _build_context_message
# ---------------------------------------------------------------------------


def test_build_context_no_diff():
    # Diff is no longer included — only imports header and entity migration.
    ctx = _make_fl_ctx()
    msg = _build_context_message([ctx])
    assert "```diff" not in msg


def test_build_context_new_file_imports_and_refs():
    # New-file section shows imports header and name-reference map; no bodies.
    src = "import os\n\ndef my_func():\n    os.path.join('a', 'b')\n"
    ctx = _make_fl_ctx(new_files={"sub_a.py": src, "sub_b.py": "class B: pass\n"})
    msg = _build_context_message([ctx])
    assert "**Imports:**" in msg
    assert "import os" in msg
    assert "**Name references**" in msg
    assert "`os`: `my_func`" in msg
    assert "def my_func" not in msg  # body not included


def test_build_context_entity_migration_present():
    ctx = _make_fl_ctx()
    msg = _build_context_message([ctx])
    assert "sub_a.py" in msg
    assert "pkg.sub_a" in msg


def test_build_context_empty_new_files_and_entities():
    # Covers the zero-iteration branches of the two for-loops.
    ctx = _make_fl_ctx(new_files={}, new_module_paths={}, entity_to_target={})
    msg = _build_context_message([ctx])
    assert "Split module" in msg
    assert "Entity migration" in msg


def test_build_context_multiple_contexts():
    ctx1 = _make_fl_ctx(old_module="pkg.big", filepath="/p/pkg/big.py")
    ctx2 = _make_fl_ctx(old_module="pkg.large", filepath="/p/pkg/large.py")
    msg = _build_context_message([ctx1, ctx2])
    assert "pkg.big" in msg
    assert "pkg.large" in msg


# ---------------------------------------------------------------------------
# _import_header
# ---------------------------------------------------------------------------


def test_import_header_stops_before_def():
    src = "import os\nfrom x import y\n\ndef foo():\n    pass\n"
    assert _import_header(src) == "import os\nfrom x import y\n"


def test_import_header_stops_before_class():
    src = "import os\n\nclass Foo:\n    pass\n"
    assert _import_header(src) == "import os\n"


def test_import_header_stops_before_async_def():
    src = "import os\nasync def foo(): pass\n"
    assert _import_header(src) == "import os\n"


def test_import_header_no_defs_returns_all():
    src = "import os\nfrom x import y\n"
    assert _import_header(src) == "import os\nfrom x import y\n"


def test_import_header_empty_source():
    assert _import_header("") == ""


def test_import_header_strips_trailing_blanks():
    src = "import os\n\n\ndef foo(): pass\n"
    assert _import_header(src) == "import os\n"


# ---------------------------------------------------------------------------
# _name_reference_map
# ---------------------------------------------------------------------------


def test_name_reference_map_basic():
    src = (
        "import os\n"
        "from x import Foo\n"
        "\n"
        "def alpha():\n"
        "    os.getcwd()\n"
        "    Foo()\n"
        "\n"
        "def beta():\n"
        "    os.path.join('a', 'b')\n"
    )
    refs = _name_reference_map(src)
    assert refs["os"] == ["alpha", "beta"]
    assert refs["Foo"] == ["alpha"]


def test_name_reference_map_alias():
    src = "import libcst as cst\n\ndef run():\n    cst.parse_module('x')\n"
    refs = _name_reference_map(src)
    assert refs["cst"] == ["run"]


def test_name_reference_map_unused_import():
    # Imported but never referenced in a function body → absent from map.
    src = "import os\n\ndef alpha():\n    pass\n"
    refs = _name_reference_map(src)
    assert "os" not in refs


def test_name_reference_map_no_imports():
    src = "def alpha():\n    x = 1\n"
    assert _name_reference_map(src) == {}


def test_name_reference_map_star_import_ignored():
    # ``from x import *`` should not add anything (alias.name == "*" branch).
    src = "from x import *\n\ndef alpha():\n    foo()\n"
    refs = _name_reference_map(src)
    assert refs == {}


def test_name_reference_map_syntax_error():
    assert _name_reference_map("def (broken:") == {}


def test_name_reference_map_class():
    src = (
        "from x import Dep\n"
        "\n"
        "class MyClass:\n"
        "    def method(self):\n"
        "        return Dep()\n"
    )
    refs = _name_reference_map(src)
    assert refs["Dep"] == ["MyClass"]


# ---------------------------------------------------------------------------
# _splice_function
# ---------------------------------------------------------------------------


def test_splice_function_basic():
    source = "line1\nline2\nline3\nline4\n"
    result = _splice_function(source, 2, 3, "new2\nnew3\n")
    assert result == "line1\nnew2\nnew3\nline4\n"


def test_splice_function_single_line():
    source = "line1\nline2\nline3\n"
    result = _splice_function(source, 2, 2, "replacement\n")
    assert result == "line1\nreplacement\nline3\n"


def test_splice_function_size_change():
    # Replace 1 line with 3 lines.
    source = "a\nb\nc\n"
    result = _splice_function(source, 2, 2, "x\ny\nz\n")
    assert result == "a\nx\ny\nz\nc\n"


def test_splice_function_no_trailing_newline():
    # new_func_text without trailing newline gets one added.
    source = "a\nb\nc\n"
    result = _splice_function(source, 2, 2, "replacement")
    assert result == "a\nreplacement\nc\n"


def test_splice_function_empty_new_text():
    # Empty string: no trailing newline added (falsy check), splitlines gives [].
    source = "a\nb\nc\n"
    result = _splice_function(source, 2, 2, "")
    assert result == "a\nc\n"


# ---------------------------------------------------------------------------
# _extract_migration_reminder
# ---------------------------------------------------------------------------


def test_extract_migration_reminder_basic():
    ctx_msg = _build_context_message([_make_fl_ctx()])
    reminder = _extract_migration_reminder(ctx_msg)
    assert "Entity migration (quick reference)" in reminder
    assert "pkg.sub_a" in reminder
    assert "pkg.sub_b" in reminder


def test_extract_migration_reminder_empty_context():
    reminder = _extract_migration_reminder("no migration here")
    assert reminder == ""


def test_extract_migration_reminder_no_entities():
    ctx = _make_fl_ctx(entity_to_target={}, new_module_paths={})
    ctx_msg = _build_context_message([ctx])
    # Empty entity_to_target → no bullets → reminder is empty string
    reminder = _extract_migration_reminder(ctx_msg)
    assert reminder == ""


def test_extract_migration_reminder_heading_stops_capture():
    # When a second fl_context follows the first, a new ## heading appears after
    # the entity migration section — the extractor must stop capturing there.
    ctx1 = _make_fl_ctx(old_module="pkg.big", filepath="/p/pkg/big.py")
    ctx2 = _make_fl_ctx(old_module="pkg.large", filepath="/p/pkg/large.py")
    ctx_msg = _build_context_message([ctx1, ctx2])
    reminder = _extract_migration_reminder(ctx_msg)
    # The reminder should contain migration bullets from both contexts but
    # not any heading markers.
    assert "### Entity migration:" not in reminder
    assert "## Split module:" not in reminder
    assert "pkg.sub_a" in reminder


# ---------------------------------------------------------------------------
# _get_external_import_names
# ---------------------------------------------------------------------------


def test_get_external_import_names_absolute():
    src = "from pkg import Foo\nimport os\n"
    names = _get_external_import_names(src)
    assert "Foo" in names
    assert "os" in names


def test_get_external_import_names_level1_skipped():
    src = "from .sub import Bar\nfrom . import Baz\n"
    names = _get_external_import_names(src)
    assert names == set()


def test_get_external_import_names_level2_included():
    src = "from ..pkg import Foo\nfrom ...llm_client import call_with_tool\n"
    names = _get_external_import_names(src)
    assert "Foo" in names
    assert "call_with_tool" in names


def test_get_external_import_names_star_import_skipped():
    src = "from pkg import *\n"
    names = _get_external_import_names(src)
    assert names == set()


def test_get_external_import_names_asname():
    src = "import libcst as cst\nfrom pkg import Foo as F\n"
    names = _get_external_import_names(src)
    assert "cst" in names
    assert "F" in names
    assert "libcst" not in names
    assert "Foo" not in names


def test_get_external_import_names_syntax_error():
    assert _get_external_import_names("def (broken:") == set()


# ---------------------------------------------------------------------------
# _extract_patch_lookup
# ---------------------------------------------------------------------------


def _make_ctx_with_ext_imports() -> _FLContext:
    """Context where original_source has real external imports that moved."""
    orig = "from ...llm_client import call_with_tool\ndef foo(): pass\n"
    mod = "from .llm_planning import call_with_tool\n"
    new_files = {
        "llm_planning.py": (
            "from ...llm_client import call_with_tool\ndef advise(): call_with_tool()\n"
        )
    }
    return _make_fl_ctx(
        original_source=orig,
        modified_source=mod,
        new_files=new_files,
        new_module_paths={"llm_planning.py": "pkg.llm_planning"},
        entity_to_target={"advise": "llm_planning.py"},
    )


def test_extract_patch_lookup_basic():
    ctx_msg = _build_context_message([_make_ctx_with_ext_imports()])
    lookup = _extract_patch_lookup(ctx_msg)
    assert "Patch target lookup" in lookup
    assert "call_with_tool" in lookup
    assert "pkg.llm_planning" in lookup


def test_extract_patch_lookup_no_section():
    # Default fixture has no external imports → no lookup section generated.
    ctx_msg = _build_context_message([_make_fl_ctx()])
    assert _extract_patch_lookup(ctx_msg) == ""


def test_extract_patch_lookup_multiple_contexts():
    ctx1 = _make_ctx_with_ext_imports()
    orig2 = "from ...config import CrispenConfig\ndef bar(): pass\n"
    mod2 = "from .cfg import CrispenConfig\n"
    new2 = {"cfg.py": "from ...config import CrispenConfig\ndef run(): pass\n"}
    ctx2 = _make_fl_ctx(
        old_module="pkg.other",
        filepath="/proj/pkg/other.py",
        original_source=orig2,
        modified_source=mod2,
        new_files=new2,
        new_module_paths={"cfg.py": "pkg.cfg"},
        entity_to_target={"run": "cfg.py"},
    )
    ctx_msg = _build_context_message([ctx1, ctx2])
    lookup = _extract_patch_lookup(ctx_msg)
    assert "call_with_tool" in lookup
    assert "CrispenConfig" in lookup


def test_extract_patch_lookup_still_in_section():
    # Name in both original and modified → appears under "still imported".
    orig = "from ...llm_client import call_with_tool, make_client\ndef foo(): pass\n"
    mod = (
        "from ...llm_client import make_client\n"
        "from .llm_planning import call_with_tool\n"
    )
    new_files = {
        "llm_planning.py": (
            "from ...llm_client import call_with_tool\ndef advise(): pass\n"
        )
    }
    ctx = _make_fl_ctx(
        original_source=orig,
        modified_source=mod,
        new_files=new_files,
        new_module_paths={"llm_planning.py": "pkg.llm_planning"},
        entity_to_target={"advise": "llm_planning.py"},
    )
    ctx_msg = _build_context_message([ctx])
    lookup = _extract_patch_lookup(ctx_msg)
    assert "call_with_tool" in lookup
    assert "make_client" in lookup
    assert "still" in lookup


def test_extract_patch_lookup_name_not_in_new_files():
    # Name moved out but not found in any new file → "(not found in new files)".
    orig = "from ...llm_client import call_with_tool\ndef foo(): pass\n"
    mod = ""  # name removed
    new_files = {"sub.py": "class X: pass\n"}  # no imports
    ctx = _make_fl_ctx(
        original_source=orig,
        modified_source=mod,
        new_files=new_files,
        new_module_paths={"sub.py": "pkg.sub"},
        entity_to_target={},
    )
    ctx_msg = _build_context_message([ctx])
    lookup = _extract_patch_lookup(ctx_msg)
    assert "not found in new files" in lookup


# ---------------------------------------------------------------------------
# _extract_still_imported_names
# ---------------------------------------------------------------------------


def test_extract_still_imported_names_basic():
    ctx_msg = _build_context_message([_make_ctx_with_ext_imports()])
    # _make_ctx_with_ext_imports has call_with_tool moved out — not still imported.
    names = _extract_still_imported_names(ctx_msg)
    assert "call_with_tool" not in names


def test_extract_still_imported_names_finds_retained():
    # Build a context where a name is retained in the modified original.
    orig = "from ...llm_client import call_with_tool, make_client\ndef foo(): pass\n"
    mod = (
        "from ...llm_client import make_client\n"
        "from .llm_planning import call_with_tool\n"
    )
    new_files = {
        "llm_planning.py": (
            "from ...llm_client import call_with_tool\ndef advise(): pass\n"
        )
    }
    ctx = _make_fl_ctx(
        original_source=orig,
        modified_source=mod,
        new_files=new_files,
        new_module_paths={"llm_planning.py": "pkg.llm_planning"},
        entity_to_target={"advise": "llm_planning.py"},
    )
    ctx_msg = _build_context_message([ctx])
    names = _extract_still_imported_names(ctx_msg)
    assert "make_client" in names
    assert "call_with_tool" not in names


def test_extract_still_imported_names_no_section():
    # No lookup section in context → empty set.
    names = _extract_still_imported_names("no relevant section here")
    assert names == set()


def test_extract_still_imported_names_section_ends_at_non_bullet():
    # Section capture stops when a non-bullet line is encountered.
    ctx_msg = (
        "Names still externally imported in the modified original (check):\n"
        "- `alpha`\n"
        "- `beta`\n"
        "\n"  # blank line — not a bullet, stops capture
        "- `gamma`\n"  # not captured
    )
    names = _extract_still_imported_names(ctx_msg)
    assert "alpha" in names
    assert "beta" in names
    assert "gamma" not in names


def test_extract_still_imported_names_malformed_bullet_ignored():
    # A bullet that starts with "- `" but has no closing backtick is silently skipped.
    ctx_msg = (
        "Names still externally imported in the modified original (check):\n"
        "- `valid`\n"
        "- `\n"  # malformed — no closing backtick → end <= 3 branch
    )
    names = _extract_still_imported_names(ctx_msg)
    assert "valid" in names
    assert len(names) == 1


# ---------------------------------------------------------------------------
# _build_rename_guard_sets
# ---------------------------------------------------------------------------


def test_build_rename_guard_sets_moved_out():
    # call_with_tool is in original_source but removed from modified_source.
    ctx = _make_fl_ctx(
        original_source="from ...llm_client import call_with_tool\ndef f(): pass\n",
        modified_source="from .sub import call_with_tool\n",
        new_files={"sub.py": "from ...llm_client import call_with_tool\n"},
    )
    moved_out, still_in, orig_users, new_mod_imports = _build_rename_guard_sets([ctx])
    assert "call_with_tool" in moved_out
    assert "call_with_tool" not in still_in


def test_build_rename_guard_sets_still_imported():
    # make_client stays in modified_source as an external import.
    ctx = _make_fl_ctx(
        original_source=(
            "from ...llm_client import make_client, call_with_tool\n"
            "def advise(): make_client()\n"
        ),
        modified_source=(
            "from ...llm_client import make_client\ndef advise(): make_client()\n"
        ),
        new_files={"sub.py": "from ...llm_client import call_with_tool\n"},
    )
    moved_out, still_in, orig_users, new_mod_imports = _build_rename_guard_sets([ctx])
    assert "make_client" in still_in
    assert "call_with_tool" in moved_out
    assert "make_client" not in moved_out


def test_build_rename_guard_sets_orig_users_map():
    # make_client is still imported and used by advise in modified_source.
    ctx = _make_fl_ctx(
        original_source="from ...llm_client import make_client\ndef advise(): pass\n",
        modified_source=(
            "from ...llm_client import make_client\ndef advise(): make_client()\n"
        ),
        new_files={},
    )
    _, _, orig_users, *_ = _build_rename_guard_sets([ctx])
    assert orig_users.get("make_client") == ["advise"]


def test_build_rename_guard_sets_no_users_not_in_map():
    # make_client is still imported but not referenced by any top-level def.
    ctx = _make_fl_ctx(
        original_source="from ...llm_client import make_client\ndef advise(): pass\n",
        modified_source="from ...llm_client import make_client\ndef advise(): pass\n",
        new_files={},
    )
    _, _, orig_users, *_ = _build_rename_guard_sets([ctx])
    assert "make_client" not in orig_users


def test_build_rename_guard_sets_empty_contexts():
    moved_out, still_in, orig_users, new_mod_imports = _build_rename_guard_sets([])
    assert moved_out == set()
    assert still_in == set()
    assert orig_users == {}
    assert new_mod_imports == {}


def test_build_rename_guard_sets_merges_multiple_contexts():
    # Two contexts each contributing one still-in name with users.
    ctx1 = _make_fl_ctx(
        original_source="from ...a import foo\ndef f1(): foo()\n",
        modified_source="from ...a import foo\ndef f1(): foo()\n",
        new_files={},
    )
    ctx2 = _make_fl_ctx(
        original_source="from ...b import bar\ndef f2(): bar()\n",
        modified_source="from ...b import bar\ndef f2(): bar()\n",
        new_files={},
    )
    _, still_in, orig_users, *_ = _build_rename_guard_sets([ctx1, ctx2])
    assert "foo" in still_in
    assert "bar" in still_in
    assert orig_users["foo"] == ["f1"]
    assert orig_users["bar"] == ["f2"]


def test_build_rename_guard_sets_deduplicates_merged_users():
    # Same name+user in two contexts → appears once in orig_users_map.
    ctx1 = _make_fl_ctx(
        original_source="from ...a import foo\ndef f1(): foo()\n",
        modified_source="from ...a import foo\ndef f1(): foo()\n",
        new_files={},
    )
    ctx2 = _make_fl_ctx(
        original_source="from ...a import foo\ndef f1(): foo()\n",
        modified_source="from ...a import foo\ndef f1(): foo()\n",
        new_files={},
    )
    _, _, orig_users, *_ = _build_rename_guard_sets([ctx1, ctx2])
    assert orig_users["foo"].count("f1") == 1


# ---------------------------------------------------------------------------
# _is_bad_rename
# ---------------------------------------------------------------------------


def test_is_bad_rename_pattern_a_shallowing_moved_out():
    # advisor.placement.call_with_tool → advisor.call_with_tool
    # call_with_tool is moved out; new_depth < old_depth → bad
    assert _is_bad_rename(
        "crispen.advisor.placement.call_with_tool",
        "crispen.advisor.call_with_tool",
        moved_out_names={"call_with_tool"},
        still_imported=set(),
        orig_users_map={},
        test_text="",
    )


def test_is_bad_rename_pattern_a_deepening_moved_out_ok():
    # Deepening a moved-out name is fine (not shallowing).
    assert not _is_bad_rename(
        "crispen.advisor.call_with_tool",
        "crispen.advisor.placement.call_with_tool",
        moved_out_names={"call_with_tool"},
        still_imported=set(),
        orig_users_map={},
        test_text="",
    )


def test_is_bad_rename_pattern_b_deepening_still_in_with_orig_user_in_test():
    # advisor.make_client → advisor.placement.make_client
    # make_client is still_imported, orig user advise_file_limiter is in test body → bad
    assert _is_bad_rename(
        "crispen.advisor.make_client",
        "crispen.advisor.placement.make_client",
        moved_out_names=set(),
        still_imported={"make_client"},
        orig_users_map={"make_client": ["advise_file_limiter"]},
        test_text="def test_foo():\n    advise_file_limiter(src)\n",
    )


def test_is_bad_rename_pattern_b_deepening_still_in_no_orig_user_in_test():
    # Same deepening but test body doesn't contain advise_file_limiter → ok
    assert not _is_bad_rename(
        "crispen.advisor.make_client",
        "crispen.advisor.placement.make_client",
        moved_out_names=set(),
        still_imported={"make_client"},
        orig_users_map={"make_client": ["advise_file_limiter"]},
        test_text="def test_foo():\n    _propose_files_step(src)\n",
    )


def test_is_bad_rename_pattern_b_deepening_no_orig_users_map():
    # Name is still_imported but not in orig_users_map → not blocked
    assert not _is_bad_rename(
        "crispen.advisor.make_client",
        "crispen.advisor.placement.make_client",
        moved_out_names=set(),
        still_imported={"make_client"},
        orig_users_map={},
        test_text="def test_foo():\n    advise_file_limiter(src)\n",
    )


def test_is_bad_rename_not_bad_when_no_relevant_sets():
    assert not _is_bad_rename(
        "a.b.foo",
        "a.b.c.foo",
        moved_out_names=set(),
        still_imported=set(),
        orig_users_map={},
        test_text="",
    )


def test_is_bad_rename_pattern_c_target_module_missing_name():
    # Target module "pkg.advisor.placement" exists in new_module_imports
    # but doesn't import call_with_tool; name is in moved_out_names → bad rename.
    assert _is_bad_rename(
        "pkg.advisor.call_with_tool",
        "pkg.advisor.placement.call_with_tool",
        moved_out_names={"call_with_tool"},
        still_imported=set(),
        orig_users_map={},
        test_text="",
        new_module_imports={"pkg.advisor.placement": {"make_client"}},
    )


def test_is_bad_rename_pattern_c_target_module_has_name():
    # Target module imports the name → not blocked by Pattern C.
    assert not _is_bad_rename(
        "pkg.advisor.call_with_tool",
        "pkg.advisor.placement.call_with_tool",
        moved_out_names={"call_with_tool"},
        still_imported=set(),
        orig_users_map={},
        test_text="",
        new_module_imports={"pkg.advisor.placement": {"call_with_tool"}},
    )


def test_is_bad_rename_pattern_c_target_module_unknown():
    # Target module not in new_module_imports (unknown module) → not blocked.
    assert not _is_bad_rename(
        "pkg.advisor.call_with_tool",
        "pkg.advisor.placement.call_with_tool",
        moved_out_names={"call_with_tool"},
        still_imported=set(),
        orig_users_map={},
        test_text="",
        new_module_imports={"pkg.advisor.schemas": {"call_with_tool"}},
    )


def test_is_bad_rename_pattern_c_name_not_tracked():
    # Name is not in moved_out_names or still_imported → Pattern C skipped
    # even if the target module doesn't import it (locally-defined symbols).
    assert not _is_bad_rename(
        "pkg.big.A",
        "pkg.sub_a.A",
        moved_out_names=set(),
        still_imported=set(),
        orig_users_map={},
        test_text="",
        new_module_imports={"pkg.sub_a": set()},
    )


def test_is_bad_rename_pattern_c_none_new_module_imports():
    # new_module_imports=None (not passed) → Pattern C skipped entirely.
    assert not _is_bad_rename(
        "pkg.advisor.call_with_tool",
        "pkg.advisor.placement.call_with_tool",
        moved_out_names={"call_with_tool"},
        still_imported=set(),
        orig_users_map={},
        test_text="",
        new_module_imports=None,
    )


def test_build_rename_guard_sets_new_module_imports():
    # new_files with known module paths populate new_module_imports correctly.
    ctx = _make_fl_ctx(
        original_source="from ...llm_client import call_with_tool, make_client\n",
        modified_source="from .placement import call_with_tool\n",
        new_files={
            "placement.py": "from ...llm_client import call_with_tool\n",
            "schemas.py": "from ...llm_client import make_client\n",
        },
        new_module_paths={
            "placement.py": "pkg.advisor.placement",
            "schemas.py": "pkg.advisor.schemas",
        },
    )
    _, _, _, new_mod_imports = _build_rename_guard_sets([ctx])
    assert new_mod_imports["pkg.advisor.placement"] == {"call_with_tool"}
    assert new_mod_imports["pkg.advisor.schemas"] == {"make_client"}


# ---------------------------------------------------------------------------
# _build_context_message: patch target lookup section
# ---------------------------------------------------------------------------


def test_build_context_lookup_present_when_names_moved():
    ctx_msg = _build_context_message([_make_ctx_with_ext_imports()])
    assert "Patch target lookup" in ctx_msg
    assert "call_with_tool" in ctx_msg


def test_build_context_lookup_annotates_using_entities():
    # When a moved-out name is used by a top-level entity in a new file, the
    # lookup entry should include "used by: <entity>" so the LLM can pick the
    # right sub-module when the name appears in multiple new files.
    orig = "from ...llm_client import call_with_tool\ndef foo(): pass\n"
    mod = "from .sub import call_with_tool\n"
    new_files = {
        "sub.py": (
            "from ...llm_client import call_with_tool\n"
            "def _do_work(): call_with_tool()\n"
        )
    }
    ctx = _make_fl_ctx(
        original_source=orig,
        modified_source=mod,
        new_files=new_files,
        new_module_paths={"sub.py": "pkg.sub"},
        entity_to_target={"_do_work": "sub.py"},
    )
    ctx_msg = _build_context_message([ctx])
    assert "used by" in ctx_msg
    assert "_do_work" in ctx_msg


def test_build_context_lookup_no_using_entities_when_name_unused():
    # If a moved-out name is imported but not referenced by any top-level entity,
    # the entry should not include a "used by" annotation.
    orig = "from ...llm_client import call_with_tool\ndef foo(): pass\n"
    mod = "from .sub import call_with_tool\n"
    new_files = {"sub.py": "from ...llm_client import call_with_tool\n"}
    ctx = _make_fl_ctx(
        original_source=orig,
        modified_source=mod,
        new_files=new_files,
        new_module_paths={"sub.py": "pkg.sub"},
        entity_to_target={},
    )
    ctx_msg = _build_context_message([ctx])
    assert "used by" not in ctx_msg


def test_build_context_lookup_absent_when_no_ext_imports():
    # Default fixture has class defs only — no external imports.
    ctx_msg = _build_context_message([_make_fl_ctx()])
    assert "Patch target lookup" not in ctx_msg


def test_build_context_lookup_only_still_in():
    # All external imports preserved in modified original → only "still imported"
    # section, no "moved" section.  Covers the if moved_out: False branch.
    # sub.py does NOT import make_client → "NOT imported in any new submodule".
    orig = "from ...llm_client import make_client\ndef foo(): pass\n"
    mod = "from ...llm_client import make_client\nfrom .sub import helper\n"
    ctx = _make_fl_ctx(
        original_source=orig,
        modified_source=mod,
        new_files={"sub.py": "def helper(): pass\n"},
        new_module_paths={"sub.py": "pkg.sub"},
        entity_to_target={"helper": "sub.py"},
    )
    ctx_msg = _build_context_message([ctx])
    assert "Patch target lookup" in ctx_msg
    assert "still" in ctx_msg
    assert "moved" not in ctx_msg
    assert "NOT imported in any new submodule" in ctx_msg


def test_build_context_lookup_still_in_also_in_new_submodule_with_users():
    # A still-in name imported by a new submodule whose entity USES it →
    # annotation shows "used by" and the migration-based guidance.
    orig = "from ...llm_client import make_client\ndef foo(): pass\n"
    mod = "from ...llm_client import make_client\nfrom .sub import helper\n"
    ctx = _make_fl_ctx(
        original_source=orig,
        modified_source=mod,
        new_files={
            "sub.py": (
                "from ...llm_client import make_client\n"
                "def helper(): make_client()\n"
            )
        },
        new_module_paths={"sub.py": "pkg.sub"},
        entity_to_target={"helper": "sub.py"},
    )
    ctx_msg = _build_context_message([ctx])
    assert "also externally imported in" in ctx_msg
    assert "pkg.sub" in ctx_msg
    assert "used by" in ctx_msg
    assert "helper" in ctx_msg
    assert "migrated to that submodule" in ctx_msg
    assert "Name references" in ctx_msg


def test_build_context_lookup_still_in_also_in_new_submodule_no_users():
    # A still-in name imported by a new submodule but NOT referenced by any
    # top-level entity → annotation shows the submodule without "used by".
    orig = "from ...llm_client import make_client\ndef foo(): pass\n"
    mod = "from ...llm_client import make_client\nfrom .sub import helper\n"
    ctx = _make_fl_ctx(
        original_source=orig,
        modified_source=mod,
        new_files={
            "sub.py": "from ...llm_client import make_client\ndef helper(): pass\n"
        },
        new_module_paths={"sub.py": "pkg.sub"},
        entity_to_target={"helper": "sub.py"},
    )
    ctx_msg = _build_context_message([ctx])
    assert "also externally imported in" in ctx_msg
    assert "pkg.sub" in ctx_msg
    # No entity in sub.py uses make_client → no "(used by: ...)" parenthetical.
    assert "(used by:" not in ctx_msg


# ---------------------------------------------------------------------------
# _build_classify_prompt
# ---------------------------------------------------------------------------


def _ctx_msg() -> str:
    return _build_context_message([_make_fl_ctx()])


def test_build_classify_prompt_no_prev():
    prompt = _build_classify_prompt(
        _ctx_msg(), "def test_f(): pass", ["crispen.before.X"]
    )
    assert "crispen.before.X" in prompt
    assert "Previous attempt was rejected" not in prompt
    assert "patch_renames" in prompt
    assert "Entity migration (quick reference)" in prompt


def test_build_classify_prompt_with_prev():
    prompt = _build_classify_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        ["crispen.before.X"],
        prev_issue="wrong module",
        prev_proposed="{'crispen.before.X': 'bad.mod.X'}",
    )
    assert "Previous attempt was rejected" in prompt
    assert "wrong module" in prompt
    assert "bad.mod.X" in prompt


def test_build_classify_prompt_multiple_paths():
    prompt = _build_classify_prompt(
        _ctx_msg(), "def test_f(): pass", ["crispen.before.X", "crispen.before.Y"]
    )
    assert "crispen.before.X" in prompt
    assert "crispen.before.Y" in prompt


def test_build_classify_prompt_with_lookup():
    # When the context has a patch target lookup, it appears in the classify prompt
    # and the simplified lookup-based algorithm is used.
    ctx_msg = _build_context_message([_make_ctx_with_ext_imports()])
    prompt = _build_classify_prompt(
        ctx_msg, "def test_f(): pass", ["pkg.big.call_with_tool"]
    )
    assert "Patch target lookup" in prompt
    assert "call_with_tool" in prompt
    assert "pkg.llm_planning" in prompt
    assert "patch_renames" in prompt
    assert "Entity migration (quick reference)" in prompt


def test_build_classify_prompt_with_stable_paths():
    # stable_patch_paths appear in a separate "already correct" section and
    # the forking path remains in the "needs updating" section.
    prompt = _build_classify_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        ["crispen.before.X"],
        stable_patch_paths=["crispen.after.Y"],
    )
    assert "crispen.before.X" in prompt
    assert "crispen.after.Y" in prompt
    assert "already correct" in prompt
    assert "do not modify" in prompt


# ---------------------------------------------------------------------------
# _build_func_verify_prompt
# ---------------------------------------------------------------------------


def test_build_func_verify_prompt_basic():
    prompt = _build_func_verify_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        {"crispen.before.X": "crispen.after.X"},
    )
    assert "crispen.before.X" in prompt
    assert "crispen.after.X" in prompt
    assert "correct" in prompt


def test_build_func_verify_prompt_multiple_renames():
    prompt = _build_func_verify_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        {"crispen.before.X": "crispen.after.X", "crispen.before.Y": "crispen.after.Y"},
    )
    assert "crispen.before.X" in prompt
    assert "crispen.before.Y" in prompt
    assert "crispen.after.X" in prompt
    assert "crispen.after.Y" in prompt


def test_build_func_verify_prompt_includes_patch_lookup():
    # When the context has a patch lookup section, it should be repeated near
    # the verify instructions.
    ctx_msg = _build_context_message([_make_ctx_with_ext_imports()])
    prompt = _build_func_verify_prompt(
        ctx_msg,
        "def test_f(): pass",
        {"pkg.old.call_with_tool": "pkg.llm_planning.call_with_tool"},
    )
    assert "Patch target lookup" in prompt


# ---------------------------------------------------------------------------
# _build_no_change_verify_prompt
# ---------------------------------------------------------------------------


def test_build_no_change_verify_prompt_includes_migration_reminder():
    # Prompt built with a context that has migration entries should include
    # the migration quick-reference block near the instructions.
    prompt = _build_no_change_verify_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        ["crispen.before.X"],
    )
    assert "crispen.before.X" in prompt
    assert "Entity migration" in prompt


def test_build_no_change_verify_prompt_includes_patch_lookup():
    # When the context has a patch lookup section, it should be repeated near
    # the verify instructions so the model doesn't have to scan the full context.
    ctx_msg = _build_context_message([_make_ctx_with_ext_imports()])
    prompt = _build_no_change_verify_prompt(
        ctx_msg,
        "def test_f(): pass",
        ["pkg.old.call_with_tool"],
    )
    assert "Patch target lookup" in prompt


def test_build_no_change_verify_prompt_with_stable_paths():
    # stable_patch_paths appear in a separate "already correct" section and
    # the instruction tells the verifier not to include them in corrections.
    prompt = _build_no_change_verify_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        ["crispen.before.X"],
        stable_patch_paths=["crispen.after.Y"],
    )
    assert "crispen.before.X" in prompt
    assert "crispen.after.Y" in prompt
    assert "already correct" in prompt
    assert "do not include" in prompt


# ---------------------------------------------------------------------------
# _build_rewrite_func_prompt
# ---------------------------------------------------------------------------


def test_build_rewrite_func_prompt_no_error():
    prompt = _build_rewrite_func_prompt(
        _ctx_msg(), "def test_f(): pass", ["crispen.before.X"]
    )
    assert "crispen.before.X" in prompt
    assert "Previous rewrite" not in prompt
    assert "Rewrite the complete function" in prompt


def test_build_rewrite_func_prompt_with_error():
    prompt = _build_rewrite_func_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        ["crispen.before.X"],
        prev_error="SyntaxError on line 3",
    )
    assert "Previous rewrite was rejected" in prompt
    assert "SyntaxError on line 3" in prompt


def test_build_rewrite_func_prompt_with_stable_paths():
    # stable_patch_paths appear in a separate "already correct" section and
    # the instruction tells the LLM not to modify them.
    prompt = _build_rewrite_func_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        ["crispen.before.X"],
        stable_patch_paths=["crispen.after.Y"],
    )
    assert "crispen.before.X" in prompt
    assert "crispen.after.Y" in prompt
    assert "already correct" in prompt
    assert "do not modify" in prompt.lower()


# ---------------------------------------------------------------------------
# _build_rewrite_verify_prompt
# ---------------------------------------------------------------------------


def test_build_rewrite_verify_prompt_basic():
    prompt = _build_rewrite_verify_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        '@patch("crispen.after.X")\ndef test_f(mock_x):\n    pass\n',
    )
    assert "Original test function" in prompt
    assert "Rewritten test function" in prompt
    assert "crispen.after.X" in prompt
    assert "correct" in prompt


# ---------------------------------------------------------------------------
# _process_file_source — basic flow
# ---------------------------------------------------------------------------


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_functions(mock_call):
    src = "def test_f(): pass\n"
    result, changed, cross = _process_file_source(
        src, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert result == src
    assert changed is False
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL, return_value=_ok(None))
def test_process_classify_tool_none(mock_call):
    # Classify returns tool_input=None with one attempt → retries exhausted, no update.
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_classify_none_retries(mock_call):
    # Classify returns tool_input=None → retry; second attempt succeeds with no change.
    mock_call.side_effect = [_ok(None), _ok(_CLASSIFY_NO_CHANGE), _ok(_VERIFY_OK)]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False
    assert mock_call.call_count == 3  # failed classify + classify + verify


@mock_patch(_PATCH_CALL_TOOL)
def test_process_classify_truncated_retries(mock_call):
    # Classify response truncated → retry; second attempt succeeds with no change.
    mock_call.side_effect = [_truncated_ok(), _ok(_CLASSIFY_NO_CHANGE), _ok(_VERIFY_OK)]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False
    assert mock_call.call_count == 3  # truncated classify + classify + verify


@mock_patch(_PATCH_CALL_TOOL)
def test_process_classify_truncated_exhausted(mock_call):
    # All attempts truncated → retries exhausted, no update.
    mock_call.side_effect = [_truncated_ok(), _truncated_ok()]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False
    assert mock_call.call_count == 2


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_needed(mock_call):
    # Classify returns empty renames → verify confirms no-change → no update.
    mock_call.side_effect = [_ok(_CLASSIFY_NO_CHANGE), _ok(_VERIFY_OK)]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False
    assert mock_call.call_count == 2  # classify + verify


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_verify_none_accept(mock_call):
    # Classify says no change; verify returns None → accept no-change.
    mock_call.side_effect = [_ok(_CLASSIFY_NO_CHANGE), _ok(None)]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False
    assert mock_call.call_count == 2


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_verify_truncated_reject(mock_call):
    # No-change verify truncated → treated as rejection, not accepted as no-change.
    mock_call.side_effect = [_ok(_CLASSIFY_NO_CHANGE), _truncated_ok()]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG_NO_LLM_VERIFY, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False
    assert mock_call.call_count == 2


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_verify_rejects_then_accepts(mock_call):
    # No-change verify rejects with corrections; corrections-verify accepts.
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),
        _ok(_VERIFY_OK),  # corrections-verify accepts
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is True
    assert "crispen.after.X" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_verify_retries_exhausted(mock_call):
    # llm_verify_retries=0: no escalation, accept no-change immediately.
    from crispen.config import CrispenConfig

    cfg = CrispenConfig(patch_update_retries=1, llm_verify_retries=0)
    mock_call.side_effect = [_ok(_CLASSIFY_NO_CHANGE), _ok(_VERIFY_REJECT)]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), cfg, 1
    )
    assert changed is False
    assert mock_call.call_count == 2


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_exhausted_escalates_to_rewrite(mock_call):
    # When llm_verify_retries>0 and no-change retries are exhausted, escalate
    # to the full rewrite path seeded with the verifier's explanation.
    from crispen.config import CrispenConfig

    cfg = CrispenConfig(patch_update_retries=3, llm_verify_retries=1)
    # Corrections that rename the function itself (X→Y) are filtered out by the
    # name-invariant guard, so corrections_renames ends up empty, causing the
    # retry to exhaust and escalate to rewrite (covers lines 2897-2901).
    name_change_correction = {
        "correct": False,
        "issue": "wrong path",
        "corrections": {"crispen.before.X": "crispen.before.Y"},
    }
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),  # classify → no change
        _ok(name_change_correction),  # verify → reject (filtered corrections)
        _ok(_CLASSIFY_NO_CHANGE),  # classify (retry) → no change again
        _ok(name_change_correction),  # verify → reject (retries exhausted → escalate)
        _ok({"rewritten_function": _VALID_REWRITE}),  # rewrite (escalated)
        _ok(_REWRITE_VERIFY_OK),  # verify rewrite → accept
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), cfg, 3
    )
    assert changed is True
    assert mock_call.call_count == 6


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_exhausted_escalate_verbose(mock_call, capsys):
    # verbose=True prints 'escalating to rewrite' when escalation is triggered.
    from crispen.config import CrispenConfig

    cfg = CrispenConfig(patch_update_retries=3, llm_verify_retries=1)
    name_change_correction = {
        "correct": False,
        "issue": "wrong path",
        "corrections": {"crispen.before.X": "crispen.before.Y"},
    }
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(name_change_correction),
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(name_change_correction),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        cfg,
        3,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "escalating to rewrite" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_applied(mock_call):
    # No-change verify returns corrections → corrections-verify accepts → apply.
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),
        _ok(_VERIFY_OK),  # corrections-verify accepts
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is True
    assert "crispen.after.X" in result
    assert mock_call.call_count == 3


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_verify_none_accept(mock_call):
    # Corrections-verify returns tool_input=None → accept.
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),
        _ok(None),  # corrections-verify returns None → accept
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is True
    assert "crispen.after.X" in result
    assert mock_call.call_count == 3


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_verify_truncated_reject(mock_call):
    # Corrections-verify truncated → treated as rejection, corrections not applied.
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),
        _truncated_ok(),  # corrections-verify truncated → reject
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG_NO_LLM_VERIFY, 2
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False
    assert mock_call.call_count == 3


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_verify_fails_retry(mock_call):
    # Corrections-verify rejects → retries left → retry classify which succeeds.
    from crispen.config import CrispenConfig

    cfg = CrispenConfig(patch_update_retries=3, llm_verify_retries=1)
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),  # classify → no change
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),  # verify → reject + corrections
        _ok(_VERIFY_REJECT),  # corrections-verify → rejected
        _ok(_CLASSIFY_RENAME),  # classify (retry) → rename
        _ok(_VERIFY_OK),  # rename verify → accept
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), cfg, 3
    )
    assert changed is True
    assert "crispen.after.X" in result
    assert mock_call.call_count == 5


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_verbose(mock_call, capsys):
    # verbose=True prints 'verifying corrections for' and 'ACCEPTED'.
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),
        _ok(_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        2,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "verifying corrections for" in err
    assert "ACCEPTED" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_verbose_reject(mock_call, capsys):
    # verbose=True prints 'REJECTED' and issue when corrections-verify rejects.
    from crispen.config import CrispenConfig

    cfg = CrispenConfig(patch_update_retries=3, llm_verify_retries=1)
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),
        _ok({"correct": False, "issue": "correction still wrong", "corrections": {}}),
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        cfg,
        3,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "corrections verify REJECTED" in err
    assert "correction still wrong" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_timing_detailed(mock_call, capsys):
    # timing='detailed' prints elapsed/token info after corrections-verify call.
    from crispen.config import CrispenConfig

    cfg = CrispenConfig(patch_update_retries=2, timing="detailed")
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_NO_CHANGE,
            elapsed=0.5,
            input_tokens=100,
            output_tokens=10,
        ),
        LLMCallResult(
            tool_input=_VERIFY_REJECT_WITH_CORRECTIONS,
            elapsed=0.4,
            input_tokens=90,
            output_tokens=20,
        ),
        LLMCallResult(
            tool_input=_VERIFY_OK,
            elapsed=0.3,
            input_tokens=80,
            output_tokens=5,
        ),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        cfg,
        2,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "0.30s" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_acc(mock_call):
    # _acc accumulates calls from classify, no-change verify, and corrections-verify.
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_NO_CHANGE,
            elapsed=0.5,
            input_tokens=100,
            output_tokens=10,
        ),
        LLMCallResult(
            tool_input=_VERIFY_REJECT_WITH_CORRECTIONS,
            elapsed=0.4,
            input_tokens=90,
            output_tokens=20,
        ),
        LLMCallResult(
            tool_input=_VERIFY_OK,
            elapsed=0.3,
            input_tokens=80,
            output_tokens=5,
        ),
    ]
    acc = RewriteAccumulator()
    _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2, _acc=acc
    )
    assert acc.calls == 3
    assert abs(acc.elapsed - 1.2) < 1e-9
    assert acc.input_tokens == 270
    assert acc.output_tokens == 35


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_no_splice(mock_call, tmp_path):
    # Corrections-verify accepts; function uses const ref → no splice; const updated.
    src = (
        'TARGET = "crispen.before.X"\n\n@patch(TARGET)\ndef test_f(mock_x):\n    pass\n'
    )
    scan = str(tmp_path / "test_foo.py")
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src, {"crispen.before.X"}, "ctx", MagicMock(), _CFG, 2, scan_file=scan
    )
    assert changed is True
    assert "crispen.after.X" in result
    assert mock_call.call_count == 3


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_name_invariant_filtered(mock_call):
    # Verifier proposes corrections that rename the patched name itself
    # (e.g. X → Y).  These must be filtered out; with an empty corrections set
    # the no-change result falls through to retry logic — here retries=1 so
    # the second classify call is made and returns no-change confirmed by verify.
    verify_name_change_correction = {
        "correct": False,
        "issue": "module moved",
        "corrections": {"crispen.before.X": "crispen.before.Y"},  # name changed!
    }
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(verify_name_change_correction),
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_OK),
    ]
    from crispen.config import CrispenConfig

    cfg = CrispenConfig(patch_update_retries=3, llm_verify_retries=1)
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), cfg, 3
    )
    # Correction was filtered (name changed X→Y) — no change applied.
    assert "crispen.before.Y" not in result
    assert mock_call.call_count == 4


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_still_imported_guard(mock_call):
    # Verifier proposes corrections that move a name listed as still-imported in
    # the context message to a non-submodule path.  The second still-imported
    # filter drops the correction; with empty corrections the retry loop resumes
    # and accepts no-change on verify.
    still_imported_ctx = (
        "Names still externally imported in the modified original (check):\n" "- `X`\n"
    )
    verify_still_imported_correction = {
        "correct": False,
        "issue": "hallucinated move",
        "corrections": {"crispen.before.X": "crispen.sub.X"},  # X is still in orig
    }
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(verify_still_imported_correction),
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_OK),
    ]
    from crispen.config import CrispenConfig

    cfg = CrispenConfig(patch_update_retries=3, llm_verify_retries=1)
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        still_imported_ctx,
        MagicMock(),
        cfg,
        3,
        still_imported={"X"},
    )
    # Correction was filtered (X still imported, non-submodule target) —
    # no change applied; retry loop accepted no-change on subsequent verify.
    assert "crispen.sub.X" not in result
    assert mock_call.call_count == 4


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_verify_verbose(mock_call, capsys):
    # verbose=True prints 'verifying no-change' and 'ACCEPTED'.
    mock_call.side_effect = [_ok(_CLASSIFY_NO_CHANGE), _ok(_VERIFY_OK)]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "verifying no-change" in err
    assert "ACCEPTED" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_verify_verbose_reject(mock_call, capsys):
    # verbose=True prints 'REJECTED' and the issue when no-change verify rejects.
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok({"correct": False, "issue": "patch still points to old module"}),
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        2,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "REJECTED" in err
    assert "patch still points to old module" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_verify_timing_detailed(mock_call, capsys):
    # timing='detailed' appends elapsed/token info after the no-change verify call.
    from crispen.config import CrispenConfig

    cfg = CrispenConfig(patch_update_retries=1, timing="detailed")
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_NO_CHANGE,
            elapsed=0.5,
            input_tokens=100,
            output_tokens=10,
        ),
        LLMCallResult(
            tool_input=_VERIFY_OK,
            elapsed=0.3,
            input_tokens=80,
            output_tokens=5,
        ),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        cfg,
        1,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "→ done" in err
    assert "0.30s" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_acc_accumulates(mock_call):
    # _acc accumulates calls from both classify and no-change verify; no_change counted.
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_NO_CHANGE,
            elapsed=0.5,
            input_tokens=100,
            output_tokens=10,
        ),
        LLMCallResult(
            tool_input=_VERIFY_OK,
            elapsed=0.3,
            input_tokens=80,
            output_tokens=5,
        ),
    ]
    acc = RewriteAccumulator()
    _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1, _acc=acc
    )
    assert acc.calls == 2
    assert abs(acc.elapsed - 0.8) < 1e-9
    assert acc.input_tokens == 180
    assert acc.output_tokens == 15
    assert acc.no_change == 1
    assert acc.rename == 0
    assert acc.rewrite == 0
    assert acc.edit_failures == 0


@mock_patch(_PATCH_CALL_TOOL, return_value=_ok(None))
def test_process_acc_edit_failure_on_classify_none(mock_call):
    # Classify returns tool_input=None → edit_failures incremented.
    acc = RewriteAccumulator()
    _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1, _acc=acc
    )
    assert acc.edit_failures == 1
    assert acc.no_change == 0
    assert acc.rename == 0
    assert acc.rewrite == 0


@mock_patch(
    _PATCH_CALL_TOOL,
    return_value=_ok(
        {
            "needs_rewrite": False,
            "patch_renames": {"crispen.before.X": "crispen.before.X"},
        }
    ),
)
def test_process_same_path_filtered_out(mock_call):
    # Rename where old == new → filtered to empty → triggers no-change verify.
    # return_value repeats for both calls; verify gets wrong type → rejects; retries
    # exhaust → accept no-change.
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False


@mock_patch(
    _PATCH_CALL_TOOL,
    return_value=_ok({"needs_rewrite": False, "patch_renames": "not-a-dict"}),
)
def test_process_patch_renames_not_dict(mock_call):
    # patch_renames is not a dict → treated as empty, no change.
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False


@mock_patch(
    _PATCH_CALL_TOOL,
    return_value=_ok(
        {"needs_rewrite": False, "patch_renames": {42: "crispen.after.X"}}
    ),
)
def test_process_patch_renames_non_string_key(mock_call):
    # Non-string key in patch_renames → filtered out.
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_patch_renames_name_invariant_filtered(mock_call):
    # LLM proposes renaming crispen.before.X → crispen.before.Y (name changed from
    # X to Y).  A file split never renames an entity — only its module path changes.
    # The rename must be filtered out, leaving no renames → triggers no-change verify.
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.before.Y"},
            }
        ),
        _ok(_VERIFY_OK),  # no-change verify confirms
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False
    assert mock_call.call_count == 2  # classify + no-change verify


@mock_patch(_PATCH_CALL_TOOL)
def test_process_string_swap_verify_accepts(mock_call):
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is True
    assert "crispen.after.X" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verify_none_accept(mock_call):
    # Verify call returns tool_input=None → accept proposed renames.
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(None),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is True
    assert "crispen.after.X" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verify_truncated_reject(mock_call):
    # Verify call truncated → treated as rejection, renames not applied.
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _truncated_ok(),  # verify truncated → reject
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG_NO_LLM_VERIFY, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False
    assert mock_call.call_count == 2


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verify_none_accept_no_splice(mock_call, tmp_path):
    # Verify returns None; function uses const ref → new_text == orig_text → no splice.
    src = (
        'TARGET = "crispen.before.X"\n\n@patch(TARGET)\ndef test_f(mock_x):\n    pass\n'
    )
    scan = str(tmp_path / "test_foo.py")
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(None),
    ]
    result, changed, cross = _process_file_source(
        src, {"crispen.before.X"}, "ctx", MagicMock(), _CFG, 1, scan_file=scan
    )
    # No splice but const should be updated via same_file_const_map.
    assert changed is True
    assert "crispen.after.X" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verify_rejected_then_accept(mock_call):
    # First verify rejects; second classify+verify is accepted.
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is True


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verify_rejected_exhausted(mock_call):
    # Verify rejects with llm_verify_retries=0 → function skipped.
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_REJECT),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG_NO_LLM_VERIFY, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verify_rejected_exhausted_escalates_to_rewrite(mock_call):
    # When llm_verify_retries>0 and rename verify retries are exhausted,
    # escalate to the full rewrite path seeded with the verifier's explanation.
    from crispen.config import CrispenConfig

    cfg = CrispenConfig(patch_update_retries=3, llm_verify_retries=1)
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),  # classify → rename
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),  # verify → reject (retries left)
        _ok(_CLASSIFY_RENAME),  # classify (retry) → rename again
        _ok(
            _VERIFY_REJECT_WITH_CORRECTIONS
        ),  # verify → reject (retries exhausted → escalate)
        _ok({"rewritten_function": _VALID_REWRITE}),  # rewrite (escalated)
        _ok(_REWRITE_VERIFY_OK),  # verify rewrite → accept
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), cfg, 3
    )
    assert changed is True
    assert mock_call.call_count == 6


# ---------------------------------------------------------------------------
# _process_file_source — full rewrite path
# ---------------------------------------------------------------------------

_VALID_REWRITE = (
    '@patch("crispen.after.X")\n'
    '@patch("crispen.after.Y")\n'
    "def test_f(mock_x, mock_y):\n"
    "    pass\n"
)


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_success(mock_call):
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is True
    assert "crispen.after.X" in result
    assert "crispen.after.Y" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_tool_none(mock_call):
    # Rewrite call returns tool_input=None → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok(None),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_empty_text(mock_call):
    # Rewrite returns empty string → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": ""}),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_non_string(mock_call):
    # Rewrite returns non-string value → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": 42}),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_compile_error_retry(mock_call):
    # First rewrite has syntax error; second is valid.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": "def f(:\n    pass\n"}),  # invalid
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is True
    assert "crispen.after.X" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_compile_error_exhausted(mock_call):
    # Both rewrite attempts fail to compile → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": "def f(:\n    pass\n"}),  # invalid
        _ok({"rewritten_function": "def f(:\n    pass\n"}),  # still invalid
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_verify_none_accept(mock_call):
    # Verify returns tool_input=None → accept the rewrite.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(None),  # verify returns None → accept
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is True
    assert "crispen.after.X" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_verify_truncated_reject(mock_call):
    # Rewrite verify truncated → treated as rejection, rewrite not accepted.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _truncated_ok(),  # verify truncated → reject
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG_NO_LLM_VERIFY, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False
    assert mock_call.call_count == 3


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_verify_rejected_then_accept(mock_call):
    # Verify rejects first rewrite; second rewrite+verify is accepted.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_REJECT),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is True


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_verify_rejected_exhausted(mock_call):
    # Verify rejects with llm_verify_retries=0 → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_REJECT),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG_NO_LLM_VERIFY, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False


# ---------------------------------------------------------------------------
# _process_file_source — per-function processing (forking case)
# ---------------------------------------------------------------------------


@mock_patch(_PATCH_CALL_TOOL)
def test_process_per_function_different_renames(mock_call):
    """Two functions with the same @patch string can receive different renames.

    This is the forking case: test_a tests an entity that moved to mod1,
    test_b tests an entity that moved to mod2.  Each gets classified and
    renamed independently.
    """
    src = (
        '@patch("crispen.before.X")\ndef test_a(m):\n    call_a()\n\n'
        '@patch("crispen.before.X")\ndef test_b(m):\n    call_b()\n'
    )
    mock_call.side_effect = [
        # test_a: classify → rename to mod1
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.mod1.X"},
            }
        ),
        _ok(_VERIFY_OK),
        # test_b: classify → rename to mod2
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.mod2.X"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src, {"crispen.before.X"}, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is True
    assert "crispen.mod1.X" in result
    assert "crispen.mod2.X" in result
    assert mock_call.call_count == 4


@mock_patch(_PATCH_CALL_TOOL)
def test_process_per_function_both_updated(mock_call):
    """Two functions with the same @patch string both get the same rename."""
    src = (
        '@patch("crispen.before.X")\ndef test_a(m):\n    pass\n\n'
        '@patch("crispen.before.X")\ndef test_b(m):\n    pass\n'
    )
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is True
    assert result.count("crispen.after.X") == 2


# ---------------------------------------------------------------------------
# _process_file_source — accumulator and verbose
# ---------------------------------------------------------------------------


@mock_patch(_PATCH_CALL_TOOL)
def test_process_acc_accumulates(mock_call):
    """_process_file_source accumulates calls, elapsed, and tokens into _acc."""
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_RENAME,
            elapsed=1.2,
            input_tokens=200,
            output_tokens=40,
        ),
        LLMCallResult(
            tool_input=_VERIFY_OK,
            elapsed=0.3,
            input_tokens=150,
            output_tokens=5,
        ),
    ]
    acc = RewriteAccumulator()
    _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1, _acc=acc
    )
    assert acc.calls == 2
    assert abs(acc.elapsed - 1.5) < 1e-9
    assert acc.input_tokens == 350
    assert acc.output_tokens == 45


@mock_patch(_PATCH_CALL_TOOL)
def test_process_acc_rewrite_accumulates(mock_call):
    """Full rewrite path accumulates classify, rewrite, and verify calls."""
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_REWRITE,
            elapsed=0.5,
            input_tokens=100,
            output_tokens=10,
        ),
        LLMCallResult(
            tool_input={"rewritten_function": _VALID_REWRITE},
            elapsed=1.5,
            input_tokens=300,
            output_tokens=60,
        ),
        LLMCallResult(
            tool_input=_REWRITE_VERIFY_OK,
            elapsed=0.2,
            input_tokens=80,
            output_tokens=5,
        ),
    ]
    acc = RewriteAccumulator()
    _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1, _acc=acc
    )
    assert acc.calls == 3
    assert abs(acc.elapsed - 2.2) < 1e-9


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_prints_to_stderr(mock_call, capsys):
    """verbose=True emits per-call messages to stderr."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "patch_rewriter" in err
    assert "classifying" in err
    assert "verifying renames" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_detailed_timing(mock_call, capsys):
    """timing='detailed' appends elapsed/token info after each call."""
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_RENAME,
            elapsed=1.23,
            input_tokens=100,
            output_tokens=20,
        ),
        LLMCallResult(
            tool_input=_VERIFY_OK,
            elapsed=0.45,
            input_tokens=80,
            output_tokens=5,
        ),
    ]
    from crispen.config import CrispenConfig

    cfg = CrispenConfig(patch_update_retries=1, timing="detailed")
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        cfg,
        1,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "→ done" in err
    assert "1.23s" in err
    assert "0.45s" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_retry_label(mock_call, capsys):
    """Retry attempts include '(retry)' in the verbose message."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_REJECT),
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        2,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "(retry)" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_verify_accepted(mock_call, capsys):
    """verbose=True prints 'ACCEPTED' when verify succeeds."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "ACCEPTED" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_verify_rejected_prints_issue(mock_call, capsys):
    """verbose=True prints 'REJECTED' and the issue when verify rejects."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(
            {
                "correct": False,
                "issue": "wrong module path",
                "corrections": {"crispen.before.X": "crispen.after.X"},
            }
        ),
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        2,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "REJECTED" in err
    assert "wrong module path" in err
    assert "ACCEPTED" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_rewrite_path(mock_call, capsys):
    """verbose=True prints 'rewriting', 'verifying rewrite', and 'rewrote'."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "rewriting" in err
    assert "verifying rewrite" in err
    assert "rewrote" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_rewrite_verify_rejected(mock_call, capsys):
    """verbose=True prints 'REJECTED' and issue when rewrite verify fails."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_REJECT),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        2,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "REJECTED" in err
    assert "wrong mock setup" in err
    assert "ACCEPTED" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_rewrite_compile_retry(mock_call, capsys):
    """verbose=True prints '(retry)' when rewrite compile fails."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": "def f(:\n    pass\n"}),  # invalid
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    cfg = CrispenConfig(patch_update_retries=1, timing="detailed")
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        cfg,
        2,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "rewriting" in err
    assert "(retry)" in err


# ---------------------------------------------------------------------------
# _process_file_source — const ref restoration after full rewrite
# ---------------------------------------------------------------------------


@mock_patch(_PATCH_CALL_TOOL)
def test_process_rewrite_restores_unchanged_const_ref(mock_call):
    """After full rewrite, @patch("value") left unchanged → reverted to @patch(NAME)."""
    src = (
        'STABLE = "pkg.stable.X"\n'
        'TARGET = "pkg.big.A"\n\n'
        "@patch(STABLE)\n"
        "@patch(TARGET)\n"
        "def test_f(mock_stable, mock_target):\n"
        "    pass\n"
    )
    # LLM updates TARGET but leaves STABLE's substituted literal unchanged.
    rewritten = (
        '@patch("pkg.stable.X")\n'
        '@patch("pkg.sub_a.A")\n'
        "def test_f(mock_stable, mock_target):\n"
        "    pass\n"
    )
    mock_call.side_effect = [
        _ok({"needs_rewrite": True}),
        _ok({"rewritten_function": rewritten}),
        _ok({"correct": True, "issue": ""}),
    ]
    result, changed, _ = _process_file_source(
        src,
        {"pkg.big.A"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file="tests/test_foo.py",
    )
    assert changed is True
    # STABLE decorator reverted to named constant form.
    assert "@patch(STABLE)" in result
    assert '@patch("pkg.stable.X")' not in result
    # TARGET decorator keeps the LLM's updated literal value.
    assert '@patch("pkg.sub_a.A")' in result


# ---------------------------------------------------------------------------
# apply_patch_rewrite
# ---------------------------------------------------------------------------


def test_rewrite_empty_contexts():
    msgs = list(apply_patch_rewrite([], {}, "/repo", _CFG))
    assert msgs == []


def test_rewrite_no_forking_paths():
    ctx = _make_fl_ctx(forking_old_paths=set())
    msgs = list(apply_patch_rewrite([ctx], {}, "/repo", _CFG))
    assert msgs == []


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(_PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_per_file_update(mock_key, mock_client, mock_call):
    mock_call.side_effect = [
        _ok({"needs_rewrite": False, "patch_renames": {"pkg.big.A": "pkg.sub_a.A"}}),
        _ok(_VERIFY_OK),
    ]
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    per_file = {"/repo/tests/test_big.py": {"source": src, "msgs": []}}
    ctx = _make_fl_ctx()
    msgs = list(apply_patch_rewrite([ctx], per_file, None, _CFG))
    updated = per_file["/repo/tests/test_big.py"]["source"]
    assert "pkg.sub_a.A" in updated
    assert any("patch_update" in m for m in per_file["/repo/tests/test_big.py"]["msgs"])
    assert msgs == []  # no disk messages since repo_root=None


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(_PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_no_repo_root_no_disk_scan(mock_key, mock_client, mock_call):
    # repo_root=None → exits after per_file; empty per_file → no LLM calls.
    msgs = list(apply_patch_rewrite([_make_fl_ctx()], {}, None, _CFG))
    assert msgs == []
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(_PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_disk_file_update(mock_key, mock_client, mock_call, tmp_path):
    test_file = tmp_path / "test_big.py"
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    test_file.write_text(src, encoding="utf-8")
    mock_call.side_effect = [
        _ok({"needs_rewrite": False, "patch_renames": {"pkg.big.A": "pkg.sub_a.A"}}),
        _ok(_VERIFY_OK),
    ]
    msgs = list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG))
    assert "pkg.sub_a.A" in test_file.read_text(encoding="utf-8")
    assert len(msgs) == 1
    assert "patch_update" in msgs[0]


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(_PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_skip_excluded_dir(mock_key, mock_client, mock_call, tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()
    f = venv / "test_big.py"
    f.write_text('@patch("pkg.big.A")\ndef test_f(): pass\n', encoding="utf-8")
    msgs = list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG))
    assert msgs == []
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(_PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_skip_per_file_abs(mock_key, mock_client, mock_call, tmp_path):
    # A file already in per_file should NOT be re-processed from disk.
    test_file = tmp_path / "test_big.py"
    test_file.write_text('@patch("pkg.big.A")\ndef test_f(): pass\n', encoding="utf-8")
    original_disk = test_file.read_text(encoding="utf-8")
    # per_file entry uses a source without matching patches (no LLM call needed).
    per_file = {str(test_file): {"source": "# no patches\n", "msgs": []}}
    list(apply_patch_rewrite([_make_fl_ctx()], per_file, str(tmp_path), _CFG))
    # Disk file untouched since it was in per_file_abs.
    assert test_file.read_text(encoding="utf-8") == original_disk
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(_PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_oserror_skipped(mock_key, mock_client, mock_call, tmp_path):
    test_file = tmp_path / "test_big.py"
    test_file.write_text('@patch("pkg.big.A")\ndef test_f(): pass\n', encoding="utf-8")
    test_file.chmod(0o000)
    try:
        msgs = list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG))
        assert msgs == []
        mock_call.assert_not_called()
    finally:
        test_file.chmod(0o644)


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(_PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_disk_file_no_match_not_updated(
    mock_key, mock_client, mock_call, tmp_path
):
    # Disk file exists but has no matching @patch decorators → changed=False,
    # file is not written, no yield message (covers the `if changed: False` branch).
    test_file = tmp_path / "no_patches.py"
    test_file.write_text("def test_unrelated(): pass\n", encoding="utf-8")
    msgs = list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG))
    assert msgs == []
    assert test_file.read_text(encoding="utf-8") == "def test_unrelated(): pass\n"
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(_PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_no_py_files_in_repo(mock_key, mock_client, mock_call, tmp_path):
    # tmp_path has no .py files → disk scan loop body never executes.
    msgs = list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG))
    assert msgs == []
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(_PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_acc_tracks_calls_and_files(mock_key, mock_client, mock_call, tmp_path):
    """RewriteAccumulator is populated with call counts and files_updated."""
    test_file = tmp_path / "test_big.py"
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    test_file.write_text(src, encoding="utf-8")
    mock_call.side_effect = [
        LLMCallResult(
            tool_input={
                "needs_rewrite": False,
                "patch_renames": {"pkg.big.A": "pkg.sub_a.A"},
            },
            elapsed=1.5,
            input_tokens=100,
            output_tokens=50,
        ),
        LLMCallResult(
            tool_input=_VERIFY_OK,
            elapsed=0.5,
            input_tokens=80,
            output_tokens=10,
        ),
    ]
    acc = RewriteAccumulator()
    list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG, _acc=acc))
    assert acc.calls == 2
    assert acc.elapsed == 2.0
    assert acc.input_tokens == 180
    assert acc.output_tokens == 60
    assert acc.files_updated == 1


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(_PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_acc_per_file_files_updated(mock_key, mock_client, mock_call):
    """files_updated is incremented for in-memory per_file changes."""
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    per_file = {"/repo/tests/test_big.py": {"source": src, "msgs": []}}
    mock_call.side_effect = [
        _ok({"needs_rewrite": False, "patch_renames": {"pkg.big.A": "pkg.sub_a.A"}}),
        _ok(_VERIFY_OK),
    ]
    acc = RewriteAccumulator()
    list(apply_patch_rewrite([_make_fl_ctx()], per_file, None, _CFG, _acc=acc))
    assert acc.files_updated == 1


# ---------------------------------------------------------------------------
# _build_local_const_map
# ---------------------------------------------------------------------------


def test_local_const_map_string_assignment():
    src = 'TARGET = "myapp.service.MyClass"\n'
    result = _build_local_const_map(src)
    assert result == {"TARGET": "myapp.service.MyClass"}


def test_local_const_map_non_string_excluded():
    src = "TARGET = 42\n"
    assert _build_local_const_map(src) == {}


def test_local_const_map_multi_target_excluded():
    # a = b = "value" has two targets → not included.
    src = 'a = b = "value"\n'
    assert _build_local_const_map(src) == {}


def test_local_const_map_syntax_error():
    assert _build_local_const_map("def f(:\n") == {}


def test_local_const_map_empty_source():
    assert _build_local_const_map("") == {}


def test_local_const_map_last_wins():
    src = 'X = "first"\nX = "second"\n'
    assert _build_local_const_map(src)["X"] == "second"


def test_local_const_map_annotated_assignment():
    src = 'TARGET: str = "myapp.service.MyClass"\n'
    assert _build_local_const_map(src) == {"TARGET": "myapp.service.MyClass"}


def test_local_const_map_annotated_non_string_excluded():
    src = "TARGET: int = 42\n"
    assert _build_local_const_map(src) == {}


def test_local_const_map_annotated_no_value_excluded():
    # Bare annotation with no value: ``TARGET: str`` — ast.AnnAssign with value=None
    src = "TARGET: str\n"
    assert _build_local_const_map(src) == {}


# ---------------------------------------------------------------------------
# _resolve_import_to_file
# ---------------------------------------------------------------------------


def test_resolve_relative_level1_py(tmp_path):
    # from .sub import NAME — sub.py exists
    (tmp_path / "sub.py").write_text("X = 1\n", encoding="utf-8")
    scan = str(tmp_path / "test_foo.py")
    result = _resolve_import_to_file("sub", 1, scan, None)
    assert result == str(tmp_path / "sub.py")


def test_resolve_relative_level1_init(tmp_path):
    # from .pkg import NAME — pkg/__init__.py exists
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    scan = str(tmp_path / "test_foo.py")
    result = _resolve_import_to_file("pkg", 1, scan, None)
    assert result == str(pkg / "__init__.py")


def test_resolve_relative_level1_no_module(tmp_path):
    # from . import NAME — finds __init__.py in same dir
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    scan = str(tmp_path / "test_foo.py")
    result = _resolve_import_to_file(None, 1, scan, None)
    assert result == str(tmp_path / "__init__.py")


def test_resolve_relative_level2(tmp_path):
    # from ..sub import NAME — goes up one level
    parent = tmp_path / "parent"
    parent.mkdir()
    child = parent / "child"
    child.mkdir()
    (parent / "sub.py").write_text("X = 1\n", encoding="utf-8")
    scan = str(child / "test_foo.py")
    result = _resolve_import_to_file("sub", 2, scan, None)
    assert result == str(parent / "sub.py")


def test_resolve_relative_not_found(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file("missing", 1, scan, None) is None


def test_resolve_relative_no_module_no_init(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file(None, 1, scan, None) is None


def test_resolve_absolute_found(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "helpers.py").write_text("X = 1\n", encoding="utf-8")
    scan = str(tmp_path / "tests" / "test_foo.py")
    result = _resolve_import_to_file("mypkg.helpers", 0, scan, str(tmp_path))
    assert result == str(pkg / "helpers.py")


def test_resolve_absolute_no_repo_root(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file("mypkg.helpers", 0, scan, None) is None


def test_resolve_absolute_no_module(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file(None, 0, scan, str(tmp_path)) is None


def test_resolve_absolute_not_found(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file("no.such.module", 0, scan, str(tmp_path)) is None


# ---------------------------------------------------------------------------
# _build_const_map
# ---------------------------------------------------------------------------


def test_build_const_map_same_file(tmp_path):
    src = 'TARGET = "myapp.service.MyClass"\n'
    scan = str(tmp_path / "test_foo.py")
    result = _build_const_map(src, scan, None)
    val, def_file = result["TARGET"]
    assert val == "myapp.service.MyClass"
    assert def_file == str((tmp_path / "test_foo.py").resolve())


def test_build_const_map_cross_file(tmp_path):
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "myapp.service.MyClass"\n', encoding="utf-8")
    src = "from .helpers import TARGET\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_const_map(src, scan, None)
    val, def_file = result["TARGET"]
    assert val == "myapp.service.MyClass"
    assert def_file == str(helpers.resolve())


def test_build_const_map_alias(tmp_path):
    helpers = tmp_path / "helpers.py"
    helpers.write_text('X = "myapp.service.MyClass"\n', encoding="utf-8")
    src = "from .helpers import X as MY_TARGET\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_const_map(src, scan, None)
    assert "MY_TARGET" in result
    assert result["MY_TARGET"][0] == "myapp.service.MyClass"


def test_build_const_map_local_priority(tmp_path):
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "imported.value"\n', encoding="utf-8")
    src = 'TARGET = "local.value"\nfrom .helpers import TARGET\n'
    scan = str(tmp_path / "test_foo.py")
    result = _build_const_map(src, scan, None)
    assert result["TARGET"][0] == "local.value"


def test_build_const_map_star_import_skipped(tmp_path):
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "myapp.service.MyClass"\n', encoding="utf-8")
    src = "from .helpers import *\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_const_map(src, scan, None)
    assert result == {}


def test_build_const_map_import_file_not_found(tmp_path):
    src = "from .missing import TARGET\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_const_map(src, scan, None)
    assert result == {}


def test_build_const_map_import_oserror(tmp_path):
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "val"\n', encoding="utf-8")
    helpers.chmod(0o000)
    try:
        src = "from .helpers import TARGET\n"
        scan = str(tmp_path / "test_foo.py")
        result = _build_const_map(src, scan, None)
        assert result == {}
    finally:
        helpers.chmod(0o644)


def test_build_const_map_syntax_error():
    result = _build_const_map("def f(:\n", "/some/file.py", None)
    assert result == {}


def test_build_const_map_no_const_in_import(tmp_path):
    helpers = tmp_path / "helpers.py"
    helpers.write_text("def some_func(): pass\n", encoding="utf-8")
    src = "from .helpers import some_func\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_const_map(src, scan, None)
    assert result == {}


# ---------------------------------------------------------------------------
# _build_attr_const_map
# ---------------------------------------------------------------------------


def test_build_attr_const_map_basic(tmp_path):
    """``import constants`` resolves string constants from the module file."""
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('TARGET = "myapp.service.MyClass"\n', encoding="utf-8")
    src = "import constants\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_attr_const_map(src, scan, str(tmp_path))
    assert "constants" in result
    val, def_file = result["constants"]["TARGET"]
    assert val == "myapp.service.MyClass"
    assert def_file == str(constants_file.resolve())


def test_build_attr_const_map_with_alias(tmp_path):
    """``import pkg.constants as C`` maps alias ``C`` to module constants."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    constants_file = pkg / "constants.py"
    constants_file.write_text('TARGET = "myapp.svc.MyClass"\n', encoding="utf-8")
    src = "import pkg.constants as C\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_attr_const_map(src, scan, str(tmp_path))
    assert "C" in result
    assert result["C"]["TARGET"][0] == "myapp.svc.MyClass"


def test_build_attr_const_map_no_file(tmp_path):
    """Import that doesn't resolve to a file → skipped, empty result."""
    src = "import missing_module\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_attr_const_map(src, scan, str(tmp_path))
    assert result == {}


def test_build_attr_const_map_oserror(tmp_path):
    """Module file exists but is unreadable → skipped."""
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('TARGET = "val"\n', encoding="utf-8")
    constants_file.chmod(0o000)
    try:
        src = "import constants\n"
        scan = str(tmp_path / "test_foo.py")
        result = _build_attr_const_map(src, scan, str(tmp_path))
        assert result == {}
    finally:
        constants_file.chmod(0o644)


def test_build_attr_const_map_syntax_error():
    """SyntaxError in source → empty result."""
    assert _build_attr_const_map("def f(:\n", "/some/file.py", None) == {}


def test_build_attr_const_map_non_import_skipped(tmp_path):
    """Non-``import`` statements (from-imports, assignments) are skipped."""
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('TARGET = "val"\n', encoding="utf-8")
    # Only a from-import and an assignment; no plain ``import`` → empty.
    src = 'from .constants import TARGET\nX = "y"\n'
    scan = str(tmp_path / "test_foo.py")
    result = _build_attr_const_map(src, scan, str(tmp_path))
    assert result == {}


# ---------------------------------------------------------------------------
# _substitute_consts_in_func_text
# ---------------------------------------------------------------------------


def test_substitute_replaces_const():
    code = "@patch(TARGET)\ndef test_f(mock): pass\n"
    result = _substitute_consts_in_func_text(code, {"TARGET": "myapp.svc.MyClass"})
    assert '@patch("myapp.svc.MyClass")' in result
    assert "TARGET" not in result


def test_substitute_no_subs_unchanged():
    code = "@patch(TARGET)\ndef test_f(mock): pass\n"
    assert _substitute_consts_in_func_text(code, {}) == code


def test_substitute_parse_error_returns_original():
    code = "def f(:\n"
    assert _substitute_consts_in_func_text(code, {"X": "val"}) == code


def test_substitute_non_patch_call_unchanged():
    # other_func(TARGET) inside the body is not a patch call → left as-is.
    code = "@patch(TARGET)\ndef test_f(mock):\n    other_func(TARGET)\n"
    result = _substitute_consts_in_func_text(code, {"TARGET": "myapp.svc.MyClass"})
    assert '@patch("myapp.svc.MyClass")' in result
    assert "other_func(TARGET)" in result


def test_substitute_name_not_in_subs_unchanged():
    # @patch(OTHER) where OTHER is not in substitutions → left as-is (line 311).
    code = "@patch(TARGET)\n@patch(OTHER)\ndef test_f(m1, m2):\n    pass\n"
    result = _substitute_consts_in_func_text(code, {"TARGET": "myapp.svc.MyClass"})
    assert '@patch("myapp.svc.MyClass")' in result
    assert "@patch(OTHER)" in result


def test_substitute_attr_in_subs():
    """@patch(module.CONSTANT) with dotted key in subs → substituted."""
    code = "@patch(constants.TARGET)\ndef test_f(mock):\n    pass\n"
    result = _substitute_consts_in_func_text(
        code, {"constants.TARGET": "myapp.svc.MyClass"}
    )
    assert '@patch("myapp.svc.MyClass")' in result
    assert "constants.TARGET" not in result


def test_substitute_attr_not_in_subs():
    """@patch(constants.OTHER) where dotted key not in subs → unchanged."""
    code = (
        "@patch(constants.TARGET)\n"
        "@patch(constants.OTHER)\n"
        "def test_f(m1, m2):\n    pass\n"
    )
    result = _substitute_consts_in_func_text(
        code, {"constants.TARGET": "myapp.svc.MyClass"}
    )
    assert '@patch("myapp.svc.MyClass")' in result
    assert "@patch(constants.OTHER)" in result


def test_substitute_attr_non_name_base():
    """@patch(a.b.c) where base is Attribute (not Name) → else branch, unchanged."""
    code = "@patch(a.b.c)\ndef test_f(mock):\n    pass\n"
    result = _substitute_consts_in_func_text(code, {"a.b.c": "should.not.replace"})
    assert "@patch(a.b.c)" in result


# ---------------------------------------------------------------------------
# _restore_const_refs
# ---------------------------------------------------------------------------


def _make_ref(const_name: str, resolved_value: str) -> _ConstRef:
    return _ConstRef(
        const_name=const_name,
        source_file="/proj/tests/helpers.py",
        resolved_value=resolved_value,
        patch_dec_idx=0,
    )


def test_restore_reverts_unchanged_plain_name():
    """@patch("value") whose value matches a const_ref → reverted to @patch(NAME)."""
    code = '@patch("myapp.svc.MyClass")\ndef test_f(mock): pass\n'
    refs = [_make_ref("TARGET", "myapp.svc.MyClass")]
    result = _restore_const_refs(code, refs)
    assert "@patch(TARGET)" in result
    assert '"myapp.svc.MyClass"' not in result


def test_restore_reverts_unchanged_attr_form():
    """@patch("value") matching module.CONST ref → reverted to @patch(module.CONST)."""
    code = '@patch("myapp.svc.MyClass")\ndef test_f(mock): pass\n'
    refs = [_make_ref("constants.TARGET", "myapp.svc.MyClass")]
    result = _restore_const_refs(code, refs)
    assert "@patch(constants.TARGET)" in result
    assert '"myapp.svc.MyClass"' not in result


def test_restore_leaves_changed_value_as_literal():
    """@patch("new.value") where new.value is not in const_refs → kept as literal."""
    code = '@patch("myapp.new.MyClass")\ndef test_f(mock): pass\n'
    refs = [_make_ref("TARGET", "myapp.old.MyClass")]
    result = _restore_const_refs(code, refs)
    assert '@patch("myapp.new.MyClass")' in result


def test_restore_empty_refs_unchanged():
    """No const_refs → text returned as-is."""
    code = '@patch("myapp.svc.MyClass")\ndef test_f(mock): pass\n'
    assert _restore_const_refs(code, []) == code


def test_restore_parse_error_returns_original():
    """Unparseable text → original returned unchanged."""
    code = "def f(:\n"
    refs = [_make_ref("TARGET", "myapp.svc.X")]
    assert _restore_const_refs(code, refs) == code


def test_restore_empty_args_patch_unchanged():
    """@patch() with no args → left as-is."""
    code = "@patch()\ndef test_f(): pass\n"
    refs = [_make_ref("TARGET", "myapp.svc.MyClass")]
    assert _restore_const_refs(code, refs) == code


def test_restore_non_string_arg_unchanged():
    """@patch(NAME) where arg is a Name node (not SimpleString) → left as-is."""
    code = "@patch(OTHER_NAME)\ndef test_f(mock): pass\n"
    refs = [_make_ref("TARGET", "myapp.svc.MyClass")]
    result = _restore_const_refs(code, refs)
    assert "@patch(OTHER_NAME)" in result


def test_restore_non_patch_call_untouched():
    """other_func("value") is not a patch call → left as-is."""
    code = (
        '@patch("myapp.svc.MyClass")\n'
        "def test_f(mock):\n"
        '    other_func("myapp.svc.OtherClass")\n'
    )
    refs = [
        _make_ref("TARGET", "myapp.svc.MyClass"),
        _make_ref("OTHER", "myapp.svc.OtherClass"),
    ]
    result = _restore_const_refs(code, refs)
    assert "@patch(TARGET)" in result
    assert 'other_func("myapp.svc.OtherClass")' in result


def test_restore_single_quote_string():
    """SimpleString with single quotes → still reverted."""
    code = "@patch('myapp.svc.MyClass')\ndef test_f(mock): pass\n"
    refs = [_make_ref("TARGET", "myapp.svc.MyClass")]
    result = _restore_const_refs(code, refs)
    assert "@patch(TARGET)" in result


def test_restore_partial_revert_mixed():
    """One decorator changed, one unchanged → only unchanged one is reverted."""
    code = (
        '@patch("myapp.svc.MyClass")\n'
        '@patch("myapp.new.Y")\n'
        "def test_f(m1, m2): pass\n"
    )
    # MyClass unchanged (should revert), Y was updated by LLM (keep literal)
    refs = [
        _make_ref("TARGET", "myapp.svc.MyClass"),
        _make_ref("Y_CONST", "myapp.old.Y"),  # old value; new value won't match
    ]
    result = _restore_const_refs(code, refs)
    assert "@patch(TARGET)" in result
    assert '@patch("myapp.new.Y")' in result


# ---------------------------------------------------------------------------
# _find_test_functions_to_update — constant reference handling
# ---------------------------------------------------------------------------


def test_find_const_ref_same_file(tmp_path):
    """@patch(CONST) where CONST is in the same file → collected, substituted."""
    src = (
        'TARGET = "crispen.before.X"\n\n@patch(TARGET)\ndef test_f(mock_x):\n    pass\n'
    )
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(src, {"crispen.before.X"}, scan_file=scan)
    assert len(result) == 1
    assert result[0].function_name == "test_f"
    # full_text sent to LLM has the value inlined
    assert '"crispen.before.X"' in result[0].full_text
    assert "TARGET" not in result[0].full_text
    # const_ref recorded
    assert len(result[0].const_refs) == 1
    assert result[0].const_refs[0].const_name == "TARGET"
    assert result[0].const_refs[0].resolved_value == "crispen.before.X"
    assert result[0].const_refs[0].patch_dec_idx == 0


def test_find_const_ref_not_in_map_not_collected(tmp_path):
    """@patch(UNRESOLVED) where name not in const_map → not collected."""
    src = "@patch(UNRESOLVED)\ndef test_f(mock): pass\n"
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(src, {"crispen.before.X"}, scan_file=scan)
    assert result == []


def test_find_const_ref_value_no_match(tmp_path):
    """@patch(CONST) where const value doesn't match old_paths → not collected."""
    src = 'TARGET = "other.mod.Y"\n\n@patch(TARGET)\ndef test_f(mock): pass\n'
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(src, {"crispen.before.X"}, scan_file=scan)
    assert result == []


def test_find_mix_literal_and_const(tmp_path):
    """Function with both a literal @patch and a const @patch → both collected."""
    src = (
        'TARGET = "crispen.before.X"\n\n'
        '@patch("crispen.before.X")\n'
        "@patch(TARGET)\n"
        "def test_f(m1, m2):\n    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(src, {"crispen.before.X"}, scan_file=scan)
    assert len(result) == 1
    assert len(result[0].old_patch_paths) == 2
    assert len(result[0].const_refs) == 1
    # patch_dec_idx of the const ref is 1 (second @patch decorator)
    assert result[0].const_refs[0].patch_dec_idx == 1


def test_find_non_matching_decorator_split_into_stable(tmp_path):
    """Non-matching decorators go to stable_patch_paths, not old_patch_paths.

    A test that patches get_api_key (already correct) and call_with_tool
    (forking, needs rewrite) should have only call_with_tool in old_patch_paths
    and get_api_key in stable_patch_paths so the LLM is not asked to evaluate
    the already-correct path.
    """
    src = (
        'KEY = "crispen.mod.get_api_key"\n'
        'CALL = "crispen.mod.call_with_tool"\n\n'
        "@patch(KEY)\n"
        "@patch(CALL)\n"
        "def test_f(mock_call, mock_key):\n    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # Only CALL's value is in old_paths; KEY's value is already correct.
    result = _find_test_functions_to_update(
        src, {"crispen.mod.call_with_tool"}, scan_file=scan
    )
    assert len(result) == 1
    # Forking path goes to old_patch_paths only.
    assert result[0].old_patch_paths == ["crispen.mod.call_with_tool"]
    # Already-correct path goes to stable_patch_paths.
    assert result[0].stable_patch_paths == ["crispen.mod.get_api_key"]
    # Both const refs must be recorded so their definitions can be updated.
    assert len(result[0].const_refs) == 2


def test_find_patch_no_args_increments_idx(tmp_path):
    """@patch() with no args increments patch_dec_idx before the const @patch."""
    src = (
        'TARGET = "crispen.before.X"\n\n'
        "@patch()\n"
        "@patch(TARGET)\n"
        "def test_f(m):\n    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(src, {"crispen.before.X"}, scan_file=scan)
    assert len(result) == 1
    assert result[0].const_refs[0].patch_dec_idx == 1


def test_find_cross_file_const(tmp_path):
    """@patch(CONST) where CONST comes from a relative import → collected."""
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "crispen.before.X"\n', encoding="utf-8")
    src = "from .helpers import TARGET\n\n@patch(TARGET)\ndef test_f(mock):\n    pass\n"
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(
        src, {"crispen.before.X"}, scan_file=scan, repo_root=str(tmp_path)
    )
    assert len(result) == 1
    assert result[0].const_refs[0].source_file == str(helpers.resolve())


def test_find_attr_const_ref_collected(tmp_path):
    """@patch(constants.TARGET) where ``import constants`` resolves → collected."""
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('TARGET = "crispen.before.X"\n', encoding="utf-8")
    src = "import constants\n\n@patch(constants.TARGET)\ndef test_f(mock):\n    pass\n"
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(
        src, {"crispen.before.X"}, scan_file=scan, repo_root=str(tmp_path)
    )
    assert len(result) == 1
    assert result[0].function_name == "test_f"
    assert result[0].const_refs[0].const_name == "constants.TARGET"
    assert result[0].const_refs[0].resolved_value == "crispen.before.X"
    assert result[0].const_refs[0].patch_dec_idx == 0
    assert result[0].const_refs[0].source_file == str(constants_file.resolve())
    # LLM sees inlined value, not the attribute access form.
    assert '"crispen.before.X"' in result[0].full_text
    assert "constants.TARGET" not in result[0].full_text


def test_find_attr_const_module_not_in_map(tmp_path):
    """@patch(unknown.TARGET) where module not in attr_const_map → not collected."""
    src = "@patch(unknown.TARGET)\ndef test_f(mock):\n    pass\n"
    scan = str(tmp_path / "test_foo.py")
    # No ``import unknown`` in source → attr_const_map empty → no match.
    result = _find_test_functions_to_update(
        src, {"crispen.before.X"}, scan_file=scan, repo_root=str(tmp_path)
    )
    assert result == []


def test_find_attr_const_attr_not_in_module(tmp_path):
    """@patch(constants.UNKNOWN) where attr not in module constants → not collected."""
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('TARGET = "crispen.before.X"\n', encoding="utf-8")
    src = "import constants\n\n@patch(constants.UNKNOWN)\ndef test_f(mock):\n    pass\n"
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(
        src, {"crispen.before.X"}, scan_file=scan, repo_root=str(tmp_path)
    )
    assert result == []


def test_find_attr_const_value_no_match(tmp_path):
    """@patch(constants.OTHER) where value doesn't match old_paths → not collected."""
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('OTHER = "unrelated.path.Class"\n', encoding="utf-8")
    src = "import constants\n\n@patch(constants.OTHER)\ndef test_f(mock):\n    pass\n"
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(
        src, {"crispen.before.X"}, scan_file=scan, repo_root=str(tmp_path)
    )
    assert result == []


def test_find_attr_multi_level_not_handled(tmp_path):
    """@patch(a.b.c) multi-level attribute (base not Name) → not collected."""
    src = "@patch(a.b.c)\ndef test_f(mock):\n    pass\n"
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(
        src, {"a.b.c"}, scan_file=scan, repo_root=str(tmp_path)
    )
    assert result == []


# ---------------------------------------------------------------------------
# _process_file_source — constant reference post-processing
# ---------------------------------------------------------------------------

_SRC_WITH_CONST = (
    'TARGET = "crispen.before.X"\n\n'
    "@patch(TARGET)\n"
    "def test_f(mock_x):\n"
    "    pass\n"
)


@mock_patch(_PATCH_CALL_TOOL)
def test_process_const_same_file_update(mock_call, tmp_path):
    """Same-file const ref → same_file_const_map updates the const definition."""
    scan = str(tmp_path / "test_foo.py")
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after.X"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_CONST,
        {"crispen.before.X"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
    )
    assert changed is True
    # apply_patch_strings updates the const definition.
    assert '"crispen.after.X"' in result
    assert '"crispen.before.X"' not in result
    # No cross-file updates for same-file const.
    assert cross == {}


@mock_patch(_PATCH_CALL_TOOL)
def test_process_const_cross_file_update(mock_call, tmp_path):
    """Const ref from imported file → cross_file_patch_maps returned."""
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "crispen.before.X"\n', encoding="utf-8")
    src = "from .helpers import TARGET\n\n@patch(TARGET)\ndef test_f(mock):\n    pass\n"
    scan = str(tmp_path / "test_foo.py")
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after.X"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src,
        {"crispen.before.X"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
        repo_root=str(tmp_path),
    )
    helpers_abs = str(helpers.resolve())
    assert helpers_abs in cross
    assert cross[helpers_abs] == {"crispen.before.X": "crispen.after.X"}


@mock_patch(
    _PATCH_CALL_TOOL, return_value=_ok({"needs_rewrite": False, "patch_renames": {}})
)
def test_process_const_no_change_no_cross(mock_call, tmp_path):
    """LLM returns no renames → no change, cross is empty."""
    scan = str(tmp_path / "test_foo.py")
    result, changed, cross = _process_file_source(
        _SRC_WITH_CONST,
        {"crispen.before.X"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
    )
    assert changed is False
    assert cross == {}


@mock_patch(_PATCH_CALL_TOOL)
def test_process_cross_file_const_ref_not_in_renames(mock_call, tmp_path):
    """Cross-file const whose patch path is not in accepted renames → skipped.

    Scenario: function has two @patch decorators with different old paths. One
    is a cross-file const ref (path A) and the other is a literal (path B).
    Classify returns rename only for B; A is not in accepted renames.
    The const ref for A should be skipped.
    """
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET_A = "crispen.before.A"\n', encoding="utf-8")
    src = (
        "from .helpers import TARGET_A\n\n"
        '@patch(TARGET_A)\n@patch("crispen.before.B")\n'
        "def test_f(m1, m2):\n    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # Classify: only rename crispen.before.B → crispen.after.B;
    # crispen.before.A unchanged.
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.B": "crispen.after.B"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src,
        {"crispen.before.A", "crispen.before.B"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
        repo_root=str(tmp_path),
    )
    assert changed is True
    assert "crispen.after.B" in result
    # crispen.before.A not in accepted renames → no cross-file update for helpers.py.
    helpers_abs = str(helpers.resolve())
    assert helpers_abs not in cross


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_scan_file_no_const_processing(mock_call):
    """scan_file="" → const_map is empty, const post-processing skipped."""
    # Even with a const-ref style source, no scan_file means no const resolution.
    src = 'TARGET = "crispen.before.X"\n\n@patch(TARGET)\ndef test_f(m):\n    pass\n'
    # With scan_file="", const_map is empty, @patch(TARGET) is not collected.
    result, changed, cross = _process_file_source(
        src, {"crispen.before.X"}, "ctx", MagicMock(), _CFG, 1
    )
    assert result == src
    assert changed is False
    assert cross == {}
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL)
def test_process_attr_const_cross_file_update(mock_call, tmp_path):
    """@patch(constants.TARGET) resolved via import → cross-file proposal returned."""
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('TARGET = "crispen.before.X"\n', encoding="utf-8")
    src = (
        "import constants\n\n"
        "@patch(constants.TARGET)\n"
        "def test_f(mock_x):\n"
        "    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after.X"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src,
        {"crispen.before.X"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
        repo_root=str(tmp_path),
    )
    # Cross-file proposal recorded for constants.py.
    constants_abs = str(constants_file.resolve())
    assert constants_abs in cross
    assert cross[constants_abs] == {"crispen.before.X": "crispen.after.X"}


# ---------------------------------------------------------------------------
# _apply_cross_file_const_updates
# ---------------------------------------------------------------------------


def test_cross_file_empty_proposals():
    msgs = list(_apply_cross_file_const_updates({}, {}))
    assert msgs == []


def test_cross_file_conflicting_proposals(tmp_path):
    """Multiple new values for the same constant → resolved is empty → skip."""
    f = tmp_path / "helpers.py"
    f.write_text('TARGET = "old.val"\n', encoding="utf-8")
    proposals = {str(f.resolve()): {"old.val": {"new.val1", "new.val2"}}}
    msgs = list(_apply_cross_file_const_updates(proposals, {}))
    assert msgs == []
    # File unchanged.
    assert f.read_text(encoding="utf-8") == 'TARGET = "old.val"\n'


def test_cross_file_per_file_entry_updated(tmp_path):
    """Const source file is in per_file → updates in-memory source, no disk write."""
    f = tmp_path / "helpers.py"
    f.write_text('TARGET = "old.val"\n', encoding="utf-8")
    per_file = {str(f): {"source": 'TARGET = "old.val"\n', "msgs": []}}
    proposals = {str(f.resolve()): {"old.val": {"new.val"}}}
    msgs = list(_apply_cross_file_const_updates(proposals, per_file))
    assert msgs == []
    assert '"new.val"' in per_file[str(f)]["source"]
    assert any("constant definition" in m for m in per_file[str(f)]["msgs"])
    # Disk file unchanged.
    assert f.read_text(encoding="utf-8") == 'TARGET = "old.val"\n'


def test_cross_file_per_file_entry_no_change(tmp_path):
    """Resolved new value equals old → apply_patch_strings makes no change → no msg."""
    f = tmp_path / "helpers.py"
    src = 'TARGET = "new.val"\n'  # already has new value
    per_file = {str(f): {"source": src, "msgs": []}}
    proposals = {str(f.resolve()): {"old.val": {"new.val"}}}
    # apply_patch_strings("TARGET = "new.val"\n", {"old.val": "new.val"}) → unchanged
    msgs = list(_apply_cross_file_const_updates(proposals, per_file))
    assert msgs == []
    assert per_file[str(f)]["msgs"] == []


def test_cross_file_disk_file_updated(tmp_path):
    """Const source file is a disk file → written, message yielded."""
    f = tmp_path / "helpers.py"
    f.write_text('TARGET = "old.val"\n', encoding="utf-8")
    proposals = {str(f.resolve()): {"old.val": {"new.val"}}}
    msgs = list(_apply_cross_file_const_updates(proposals, {}))
    assert len(msgs) == 1
    assert "constant definition" in msgs[0]
    assert '"new.val"' in f.read_text(encoding="utf-8")


def test_cross_file_disk_file_no_change(tmp_path):
    """Disk file already has the new value → no write, no message."""
    f = tmp_path / "helpers.py"
    f.write_text('TARGET = "new.val"\n', encoding="utf-8")
    proposals = {str(f.resolve()): {"old.val": {"new.val"}}}
    msgs = list(_apply_cross_file_const_updates(proposals, {}))
    assert msgs == []


def test_cross_file_disk_oserror(tmp_path):
    """OSError reading disk file → skipped silently."""
    f = tmp_path / "helpers.py"
    f.write_text('TARGET = "old.val"\n', encoding="utf-8")
    f.chmod(0o000)
    try:
        proposals = {str(f.resolve()): {"old.val": {"new.val"}}}
        msgs = list(_apply_cross_file_const_updates(proposals, {}))
        assert msgs == []
    finally:
        f.chmod(0o644)


# ---------------------------------------------------------------------------
# apply_patch_rewrite — cross-file constant integration
# ---------------------------------------------------------------------------


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(_PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_cross_file_const_per_file(mock_key, mock_client, mock_call, tmp_path):
    """Cross-file const whose source is in per_file gets updated in-memory."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "pkg.big.A"\n', encoding="utf-8")
    # test_foo.py imports TARGET from helpers and uses it in @patch.
    test_src = (
        "from .helpers import TARGET\n\n@patch(TARGET)\ndef test_f(m):\n    pass\n"
    )
    helpers_state = {"source": 'TARGET = "pkg.big.A"\n', "msgs": []}
    per_file = {
        str(tmp_path / "test_foo.py"): {"source": test_src, "msgs": []},
        str(helpers): helpers_state,
    }
    mock_call.side_effect = [
        _ok({"needs_rewrite": False, "patch_renames": {"pkg.big.A": "pkg.sub_a.A"}}),
        _ok(_VERIFY_OK),
    ]
    list(apply_patch_rewrite([_make_fl_ctx()], per_file, None, _CFG))
    # The constant definition in helpers.py (per_file entry) should be updated.
    assert '"pkg.sub_a.A"' in helpers_state["source"]


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(_PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_cross_file_const_disk(mock_key, mock_client, mock_call, tmp_path):
    """Cross-file const on disk (not in per_file) gets written directly."""
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "pkg.big.A"\n', encoding="utf-8")
    test_src = (
        "from .helpers import TARGET\n\n@patch(TARGET)\ndef test_f(m):\n    pass\n"
    )
    test_file = tmp_path / "test_foo.py"
    test_file.write_text(test_src, encoding="utf-8")
    mock_call.side_effect = [
        _ok({"needs_rewrite": False, "patch_renames": {"pkg.big.A": "pkg.sub_a.A"}}),
        _ok(_VERIFY_OK),
    ]
    list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG))
    # The constant definition on disk should be updated.
    assert '"pkg.sub_a.A"' in helpers.read_text(encoding="utf-8")


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(_PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_cross_file_const_per_file_acc(
    mock_key, mock_client, mock_call, tmp_path
):
    """_acc.files_updated is incremented when a cross-file const in per_file changes."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "pkg.big.A"\n', encoding="utf-8")
    test_src = (
        "from .helpers import TARGET\n\n@patch(TARGET)\ndef test_f(m):\n    pass\n"
    )
    helpers_state = {"source": 'TARGET = "pkg.big.A"\n', "msgs": []}
    per_file = {
        str(tmp_path / "test_foo.py"): {"source": test_src, "msgs": []},
        str(helpers): helpers_state,
    }
    mock_call.side_effect = [
        _ok({"needs_rewrite": False, "patch_renames": {"pkg.big.A": "pkg.sub_a.A"}}),
        _ok(_VERIFY_OK),
    ]
    acc = RewriteAccumulator()
    list(apply_patch_rewrite([_make_fl_ctx()], per_file, None, _CFG, _acc=acc))
    # One file_updated for the test_foo.py source change, one for helpers const.
    assert acc.files_updated >= 1
    assert '"pkg.sub_a.A"' in helpers_state["source"]


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(_PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_cross_file_const_disk_acc(mock_key, mock_client, mock_call, tmp_path):
    """_acc.files_updated is incremented when a cross-file const on disk changes."""
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "pkg.big.A"\n', encoding="utf-8")
    test_src = (
        "from .helpers import TARGET\n\n@patch(TARGET)\ndef test_f(m):\n    pass\n"
    )
    test_file = tmp_path / "test_foo.py"
    test_file.write_text(test_src, encoding="utf-8")
    mock_call.side_effect = [
        _ok({"needs_rewrite": False, "patch_renames": {"pkg.big.A": "pkg.sub_a.A"}}),
        _ok(_VERIFY_OK),
    ]
    acc = RewriteAccumulator()
    list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG, _acc=acc))
    assert acc.files_updated >= 1


# ---------------------------------------------------------------------------
# Same-file constant: passthrough votes "keep old" — conflicts with rename
# ---------------------------------------------------------------------------


@mock_patch(_PATCH_CALL_TOOL)
def test_process_passthrough_votes_conflict_with_rename_proposal(mock_call, tmp_path):
    """One test (A) renames Y but not X → casts "keep old" vote for X.
    Another test (B) renames X → casts "rename" vote for X.
    "keep old" + "rename" → conflicting proposals → inline test_b with new value;
    test_a's decorator unchanged.  TARGET2 (Y) has a single rename vote → updated
    via same_file_const_map.

    Covers:
      - "keep old" vote (new_val is None) entered into same_file_proposals
      - conflict detection (len > 1) → conflicting_old_vals
      - per-function inline for test_b (existing_idx is None → append)
      - test_a in conflicting inline loop with new_val=None → inline_subs empty
        → continue
      - single-proposal for TARGET2 (value != old) → same_file_const_map update
    """
    src = (
        'TARGET = "crispen.before.X"\n'
        'TARGET2 = "crispen.before.Y"\n'
        "\n"
        "@patch(TARGET)\n"
        "@patch(TARGET2)\n"
        "def test_a(mock_y, mock_x):\n"
        "    pass\n"
        "\n"
        "@patch(TARGET)\n"
        "def test_b(mock_x):\n"
        "    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # test_a renames Y but NOT X → X gets a "keep old" vote, Y gets a rename vote.
    # test_b renames X → X gets a "rename to after.X" vote.
    # X proposals: {old, after.X} → conflicting → inline test_b, test_a unchanged.
    # Y proposals: {after.Y}      → single, != old → same_file_const_map update.
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.Y": "crispen.after.Y"},
            }
        ),
        _ok(_VERIFY_OK),
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after.X"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src,
        {"crispen.before.X", "crispen.before.Y"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
    )
    assert changed is True
    # X has conflicting votes → TARGET NOT updated globally.
    assert 'TARGET = "crispen.before.X"' in result
    # Y has single vote → TARGET2 updated via same_file_const_map.
    assert 'TARGET2 = "crispen.after.Y"' in result
    # test_b's X decorator is inlined individually.
    assert '@patch("crispen.after.X")' in result
    # test_a's decorator unchanged (its inline_subs were empty).
    assert "@patch(TARGET)" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_passthrough_identity_proposal_skipped(mock_call, tmp_path):
    """One test renames Y but not X.  X only receives a "keep old" identity vote.
    Expected: TARGET not updated (identity guard: proposed == old); TARGET2 updated.

    Covers the ``next(iter(new_set)) != old`` identity guard in same_file_const_map
    that drops entries where the sole proposal equals the existing value.
    """
    src = (
        'TARGET = "crispen.before.X"\n'
        'TARGET2 = "crispen.before.Y"\n'
        "\n"
        "@patch(TARGET)\n"
        "@patch(TARGET2)\n"
        "def test_a(mock_y, mock_x):\n"
        "    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # test_a: renames Y → after.Y, does not rename X.
    # X proposals: {"crispen.before.X"} → len==1, value==old → identity skip.
    # Y proposals: {"crispen.after.Y"}  → len==1, value!=old → const_map update.
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.Y": "crispen.after.Y"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src,
        {"crispen.before.X", "crispen.before.Y"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
    )
    assert changed is True
    # X got only an identity vote → not in same_file_const_map → TARGET unchanged.
    assert 'TARGET = "crispen.before.X"' in result
    # Y got a rename vote → TARGET2 updated.
    assert 'TARGET2 = "crispen.after.Y"' in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_conflict_two_renames_existing_splice(mock_call, tmp_path):
    """Two tests rename the same constant to *different* targets → conflict.
    test_a also renames a literal patch → it gets a func_splice from string_swap.
    Expected: both functions get inlined with their respective literals;
    test_a's existing splice is *updated in place* (existing_idx path).

    Covers:
      - lines 1763-1772  (loop, build inline_subs)
      - line 1787-False  (inlined != base_text)
      - line 1789-True   (existing_idx not None → update splice)
      - line 1792        (existing_idx is None → append splice, for test_b)
    """
    src = (
        'TARGET = "crispen.before.X"\n'
        "\n"
        "@patch(TARGET)\n"
        '@patch("crispen.before.Z")\n'
        "def test_a(mock_z, mock_x):\n"
        "    pass\n"
        "\n"
        "@patch(TARGET)\n"
        "def test_b(mock_x):\n"
        "    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # test_a renames X → after_a.X and Z → after.Z.
    # test_b renames X → after_b.X.
    # Two different targets for X → conflict → inline each function individually.
    # test_a's Z literal rename creates an existing func_splice; the inline step
    # must update that existing splice rather than appending a new one.
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {
                    "crispen.before.X": "crispen.after_a.X",
                    "crispen.before.Z": "crispen.after.Z",
                },
            }
        ),
        _ok(_VERIFY_OK),
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after_b.X"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src,
        {"crispen.before.X", "crispen.before.Z"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
    )
    assert changed is True
    # The shared TARGET constant must NOT be updated (conflict).
    assert 'TARGET = "crispen.before.X"' in result
    # test_a: Z literal renamed, X constant inlined.
    assert '@patch("crispen.after_a.X")' in result
    assert '@patch("crispen.after.Z")' in result
    # test_b: X constant inlined with its own target.
    assert '@patch("crispen.after_b.X")' in result
    # No original constant-style decorator survives.
    assert "@patch(TARGET)" not in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_conflict_two_proposals_passthrough_function_continue(
    mock_call, tmp_path
):
    """Two functions propose *different* values for TARGET → conflicting_old_vals.
    A third function also uses TARGET but only renames a different const (TARGET_Y).
    Expected: the third function is in string_swap_results but triggers the
    ``continue`` branch in the conflicting_old_vals inline loop (inline_subs
    empty for X); the other two get their decorators inlined individually.

    Covers the ``if not inline_subs: continue`` branch inside the
    ``if conflicting_old_vals:`` block (via two sub-paths):
      - ref.resolved_value NOT in conflicting_old_vals (Y ref → loop continues)
      - ref.resolved_value in conflicting_old_vals but new_val is None (X ref)
    """
    src = (
        'TARGET = "crispen.before.X"\n'
        'TARGET_Y = "crispen.before.Y"\n'
        "\n"
        "@patch(TARGET)\n"
        "def test_a(mock_x):\n"
        "    pass\n"
        "\n"
        "@patch(TARGET)\n"
        "def test_b(mock_x):\n"
        "    pass\n"
        "\n"
        "@patch(TARGET)\n"
        "@patch(TARGET_Y)\n"
        "def test_c(mock_y, mock_x):\n"
        "    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # test_a → after_a.X; test_b → after_b.X (two different proposals → conflicting)
    # test_c → renames Y only (not X) → in string_swap_results but inline_subs empty
    #   for X → continue.  Y gets a single proposal → same_file_const_map update.
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after_a.X"},
            }
        ),
        _ok(_VERIFY_OK),
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after_b.X"},
            }
        ),
        _ok(_VERIFY_OK),
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.Y": "crispen.after.Y"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src,
        {"crispen.before.X", "crispen.before.Y"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
    )
    assert changed is True
    # X: conflicting (two proposals) → const unchanged, test_a and test_b inlined.
    assert 'TARGET = "crispen.before.X"' in result
    assert '@patch("crispen.after_a.X")' in result
    assert '@patch("crispen.after_b.X")' in result
    # Y: single proposal → const updated via same_file_const_map.
    assert 'TARGET_Y = "crispen.after.Y"' in result
    # test_c: in string_swap_results (renamed Y) but X inline_subs empty → continue.
    assert "@patch(TARGET)" in result


# ---------------------------------------------------------------------------
# Call-graph helpers: _cg_collect_called_names
# ---------------------------------------------------------------------------


def test_cg_collect_called_names_name_and_attr():
    src = "foo()\nobj.bar()\n"
    result = _cg_collect_called_names(src)
    assert "foo" in result
    assert "bar" in result


def test_cg_collect_called_names_complex_func():
    # f()() — outer call's func is a Call node (neither Name nor Attribute).
    src = "f()()\n"
    result = _cg_collect_called_names(src)
    # Only the inner call's name is collected (f), the outer call is skipped.
    assert "f" in result


def test_cg_collect_called_names_parse_error():
    assert _cg_collect_called_names("def f(:\n") == set()


def test_cg_collect_called_names_no_calls():
    assert _cg_collect_called_names("x = 1\n") == set()


# ---------------------------------------------------------------------------
# _cg_collect_func_body_calls
# ---------------------------------------------------------------------------


def test_cg_collect_func_body_calls_found():
    src = "def helper(): foo()\ndef other(): bar()\n"
    result = _cg_collect_func_body_calls(src, "helper")
    assert "foo" in result
    assert "bar" not in result


def test_cg_collect_func_body_calls_not_found():
    src = "def helper(): foo()\n"
    assert _cg_collect_func_body_calls(src, "missing") == set()


def test_cg_collect_func_body_calls_parse_error():
    assert _cg_collect_func_body_calls("def f(:\n", "f") == set()


def test_cg_collect_func_body_calls_attribute_call():
    src = "def helper(): obj.method()\n"
    result = _cg_collect_func_body_calls(src, "helper")
    assert "method" in result


def test_cg_collect_func_body_calls_complex_func():
    # f()() inside a function body — outer call's func is a Call, not Name/Attribute.
    src = "def helper(): f()()\n"
    result = _cg_collect_func_body_calls(src, "helper")
    assert "f" in result  # inner call collected; outer (complex func) silently skipped


def test_cg_collect_func_body_calls_skips_non_function_nodes():
    # Module-level assignment before the function — should be skipped.
    src = "X = 1\ndef helper(): foo()\n"
    result = _cg_collect_func_body_calls(src, "helper")
    assert "foo" in result


# ---------------------------------------------------------------------------
# _cg_collect_called_names — alias-access form
# ---------------------------------------------------------------------------


def test_cg_collect_called_names_alias_access_emits_pair():
    # ``m.func()`` should emit both ``"func"`` and ``"m.func"``.
    src = "import mymod as m\nm.func()\n"
    result = _cg_collect_called_names(src)
    assert "func" in result
    assert "m.func" in result


def test_cg_collect_called_names_nested_attr_no_alias_pair():
    # ``a.b.c()`` — the receiver of ``.c`` is itself an Attribute, not a Name;
    # only the bare attr name is emitted (no alias pair for chained access).
    src = "a.b.c()\n"
    result = _cg_collect_called_names(src)
    assert "c" in result
    assert "b.c" not in result  # receiver is Attribute, not Name


# ---------------------------------------------------------------------------
# _cg_collect_func_body_calls — alias-access form
# ---------------------------------------------------------------------------


def test_cg_collect_func_body_calls_alias_access_emits_pair():
    src = "import mymod as m\ndef helper(): m.process()\n"
    result = _cg_collect_func_body_calls(src, "helper")
    assert "process" in result
    assert "m.process" in result


def test_cg_collect_func_body_calls_nested_attr_no_alias_pair():
    # Chained access ``a.b.c()`` — receiver of attr c is not a Name.
    src = "def helper(): a.b.c()\n"
    result = _cg_collect_func_body_calls(src, "helper")
    assert "c" in result
    assert "b.c" not in result


# ---------------------------------------------------------------------------
# _cg_resolve_call_to_import
# ---------------------------------------------------------------------------


def test_cg_resolve_call_plain_name():
    imports = {"foo": ("pkg.sub", "foo"), "bar": ("pkg.other", "bar")}
    assert _cg_resolve_call_to_import("foo", imports) == ("pkg.sub", "foo")


def test_cg_resolve_call_alias_attr():
    # ``m.process()`` — alias ``m`` maps to module ``mymod``; resolves to
    # ``(mymod, "process")``.
    imports = {"m": ("mymod", "mymod")}
    assert _cg_resolve_call_to_import("m.process", imports) == ("mymod", "process")


def test_cg_resolve_call_alias_attr_unknown_alias():
    # Alias not in imports → None.
    assert _cg_resolve_call_to_import("unknown.func", {"m": ("mymod", "mymod")}) is None


def test_cg_resolve_call_plain_not_found():
    assert _cg_resolve_call_to_import("missing", {"foo": ("pkg", "foo")}) is None


# ---------------------------------------------------------------------------
# _cg_collect_defined_names
# ---------------------------------------------------------------------------


def test_cg_collect_defined_names_functions_and_classes():
    src = "def foo(): pass\nclass Bar: pass\nasync def baz(): pass\n"
    result = _cg_collect_defined_names(src)
    assert result == {"foo", "Bar", "baz"}


def test_cg_collect_defined_names_parse_error():
    assert _cg_collect_defined_names("def f(:\n") == set()


def test_cg_collect_defined_names_empty():
    assert _cg_collect_defined_names("x = 1\n") == set()


# ---------------------------------------------------------------------------
# _cg_file_to_module_and_package
# ---------------------------------------------------------------------------


def test_cg_file_to_module_regular(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    f = pkg / "helpers.py"
    f.touch()
    mod, pkg_path = _cg_file_to_module_and_package(f, tmp_path)
    assert mod == "pkg.helpers"
    assert pkg_path == "pkg"


def test_cg_file_to_module_init(tmp_path):
    d = tmp_path / "pkg" / "utils"
    d.mkdir(parents=True)
    f = d / "__init__.py"
    f.touch()
    mod, pkg_path = _cg_file_to_module_and_package(f, tmp_path)
    assert mod == "pkg.utils"
    assert pkg_path == "pkg.utils"


def test_cg_file_to_module_top_level(tmp_path):
    f = tmp_path / "helpers.py"
    f.touch()
    mod, pkg_path = _cg_file_to_module_and_package(f, tmp_path)
    assert mod == "helpers"
    assert pkg_path == ""


def test_cg_file_to_module_nested(tmp_path):
    d = tmp_path / "a" / "b"
    d.mkdir(parents=True)
    f = d / "c.py"
    f.touch()
    mod, pkg_path = _cg_file_to_module_and_package(f, tmp_path)
    assert mod == "a.b.c"
    assert pkg_path == "a.b"


# ---------------------------------------------------------------------------
# _cg_parse_imports
# ---------------------------------------------------------------------------


def test_cg_parse_imports_from_import():
    assert _cg_parse_imports("from pkg.sub import foo\n", "pkg") == {
        "foo": ("pkg.sub", "foo")
    }


def test_cg_parse_imports_import_simple():
    result = _cg_parse_imports("import os\n", "pkg")
    assert result["os"] == ("os", "os")


def test_cg_parse_imports_import_dotted():
    result = _cg_parse_imports("import pkg.sub\n", "")
    assert result["pkg"] == ("pkg.sub", "pkg.sub")


def test_cg_parse_imports_import_as():
    result = _cg_parse_imports("import os as o\n", "pkg")
    assert result["o"] == ("os", "os")


def test_cg_parse_imports_from_import_as():
    assert _cg_parse_imports("from pkg import foo as bar\n", "pkg") == {
        "bar": ("pkg", "foo")
    }


def test_cg_parse_imports_relative_level1():
    # `from . import helper` with package "pkg.sub" → mod = "pkg.sub"
    assert _cg_parse_imports("from . import helper\n", "pkg.sub") == {
        "helper": ("pkg.sub", "helper")
    }


def test_cg_parse_imports_relative_level2():
    # `from .. import foo` with package "pkg.sub" → base = "pkg"
    assert _cg_parse_imports("from .. import foo\n", "pkg.sub") == {
        "foo": ("pkg", "foo")
    }


def test_cg_parse_imports_relative_with_module():
    # `from .utils import helper` with package "pkg" → mod = "pkg.utils"
    assert _cg_parse_imports("from .utils import helper\n", "pkg") == {
        "helper": ("pkg.utils", "helper")
    }


def test_cg_parse_imports_relative_with_empty_base():
    # `from .sub import foo` with empty package → base="" → mod = "sub"
    assert _cg_parse_imports("from .sub import foo\n", "") == {"foo": ("sub", "foo")}


def test_cg_parse_imports_relative_no_module():
    # `from . import bar` with package "pkg.sub" → mod = "pkg.sub"
    assert _cg_parse_imports("from . import bar\n", "pkg.sub") == {
        "bar": ("pkg.sub", "bar")
    }


def test_cg_parse_imports_star_skipped():
    assert _cg_parse_imports("from pkg import *\n", "pkg") == {}


def test_cg_parse_imports_syntax_error():
    assert _cg_parse_imports("def f(:\n", "pkg") == {}


def test_cg_parse_imports_too_deep_relative():
    # level=3 with package="pkg" → go_up=2 > len(["pkg"])=1 → skipped
    assert _cg_parse_imports("from ... import foo\n", "pkg") == {}


def test_cg_parse_imports_level2_with_submodule():
    # `from ..utils import foo` with package "pkg.sub" → base="pkg" → "pkg.utils"
    assert _cg_parse_imports("from ..utils import foo\n", "pkg.sub") == {
        "foo": ("pkg.utils", "foo")
    }


# ---------------------------------------------------------------------------
# _CgIndex.get_imports
# ---------------------------------------------------------------------------


def test_cg_index_get_imports_cached():
    index = _CgIndex(
        module_to_source={"pkg.mod": "from pkg.sub import foo\n"},
        module_to_package={"pkg.mod": "pkg"},
        module_to_defs={"pkg.mod": set()},
        file_to_module={},
    )
    r1 = index.get_imports("pkg.mod")
    r2 = index.get_imports("pkg.mod")  # second call — cached
    assert r1 == r2 == {"foo": ("pkg.sub", "foo")}
    assert "pkg.mod" in index._import_cache


def test_cg_index_get_imports_missing_module():
    index = _CgIndex(
        module_to_source={},
        module_to_package={},
        module_to_defs={},
        file_to_module={},
    )
    assert index.get_imports("nonexistent") == {}


# ---------------------------------------------------------------------------
# _cg_build_index
# ---------------------------------------------------------------------------


def test_cg_build_index_from_repo(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text("def foo(): pass\n", encoding="utf-8")
    index = _cg_build_index(str(tmp_path), {}, [])
    assert "pkg.mod" in index.module_to_source
    assert "foo" in index.module_to_defs["pkg.mod"]
    assert index.module_to_package["pkg.mod"] == "pkg"


def test_cg_build_index_per_file_override(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    f = pkg / "mod.py"
    f.write_text("def old(): pass\n", encoding="utf-8")
    abs_path = str(f.resolve())
    index = _cg_build_index(str(tmp_path), {abs_path: "def new(): pass\n"}, [])
    assert "new" in index.module_to_defs["pkg.mod"]
    assert "old" not in index.module_to_defs["pkg.mod"]


def test_cg_build_index_no_repo_root():
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="",
        modified_source="",
        new_files={"placement.py": "def helper(): pass\n"},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths=set(),
    )
    index = _cg_build_index(None, {}, [ctx])
    assert "pkg.placement" in index.module_to_source
    assert index.file_to_module == {}


def test_cg_build_index_excluded_dirs(tmp_path):
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "mod.py").write_text("def foo(): pass\n", encoding="utf-8")
    index = _cg_build_index(str(tmp_path), {}, [])
    assert "mod" not in index.module_to_source


def test_cg_build_index_new_files_from_context():
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="",
        modified_source="",
        new_files={"placement.py": "def helper(): pass\n"},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths=set(),
    )
    index = _cg_build_index(None, {}, [ctx])
    assert "helper" in index.module_to_defs["pkg.placement"]
    assert index.module_to_package["pkg.placement"] == "pkg"


def test_cg_build_index_init_package(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .sub import foo\n", encoding="utf-8")
    (pkg / "sub.py").write_text("def foo(): pass\n", encoding="utf-8")
    index = _cg_build_index(str(tmp_path), {}, [])
    assert "pkg" in index.module_to_source
    assert index.module_to_package["pkg"] == "pkg"


def test_cg_build_index_already_in_index():
    ctx1 = _FLContext(
        filepath="/proj/orig.py",
        old_module="orig",
        original_source="",
        modified_source="",
        new_files={"placement.py": "def first(): pass\n"},
        new_module_paths={"placement.py": "pkg.shared"},
        entity_to_target={},
        forking_old_paths=set(),
    )
    ctx2 = _FLContext(
        filepath="/proj/orig2.py",
        old_module="orig2",
        original_source="",
        modified_source="",
        new_files={"placement.py": "def second(): pass\n"},
        new_module_paths={"placement.py": "pkg.shared"},  # same module path
        entity_to_target={},
        forking_old_paths=set(),
    )
    index = _cg_build_index(None, {}, [ctx1, ctx2])
    assert "first" in index.module_to_defs["pkg.shared"]
    assert "second" not in index.module_to_defs["pkg.shared"]


def test_cg_build_index_oserror(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    bad = pkg / "bad.py"
    bad.write_text("def foo(): pass\n", encoding="utf-8")
    bad.chmod(0o000)
    try:
        index = _cg_build_index(str(tmp_path), {}, [])
        assert "pkg.bad" not in index.module_to_source
    finally:
        bad.chmod(0o644)


def test_cg_build_index_missing_module_path():
    ctx = _FLContext(
        filepath="/proj/orig.py",
        old_module="orig",
        original_source="",
        modified_source="",
        new_files={"placement.py": "def helper(): pass\n"},
        new_module_paths={},  # rel_path missing → new_mod = None → skip
        entity_to_target={},
        forking_old_paths=set(),
    )
    index = _cg_build_index(None, {}, [ctx])
    assert "placement.py" not in index.module_to_source


def test_cg_build_index_empty_src():
    ctx = _FLContext(
        filepath="/proj/orig.py",
        old_module="orig",
        original_source="",
        modified_source="",
        new_files={"placement.py": ""},  # empty src → skipped
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths=set(),
    )
    index = _cg_build_index(None, {}, [ctx])
    assert "pkg.placement" not in index.module_to_source


def test_cg_build_index_init_package_new_file():
    # __init__.py as a new file: pkg = new_mod (not rsplit)
    ctx = _FLContext(
        filepath="/proj/orig.py",
        old_module="orig",
        original_source="",
        modified_source="",
        new_files={"__init__.py": "def init_fn(): pass\n"},
        new_module_paths={"__init__.py": "pkg.sub"},
        entity_to_target={},
        forking_old_paths=set(),
    )
    index = _cg_build_index(None, {}, [ctx])
    assert index.module_to_package["pkg.sub"] == "pkg.sub"


def test_cg_build_index_top_level_new_file():
    # new_mod without a dot → package = ""
    ctx = _FLContext(
        filepath="/proj/orig.py",
        old_module="orig",
        original_source="",
        modified_source="",
        new_files={"placement.py": "def helper(): pass\n"},
        new_module_paths={"placement.py": "placement"},  # no dot
        entity_to_target={},
        forking_old_paths=set(),
    )
    index = _cg_build_index(None, {}, [ctx])
    assert index.module_to_package.get("placement") == ""


# ---------------------------------------------------------------------------
# _resolve_forking_path_via_callgraph — BFS helpers
# ---------------------------------------------------------------------------


def _make_bfs_ctx() -> _FLContext:
    """Context with placement.py (helper) and conflict.py (resolve) using use_fn."""
    return _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="from .placement import helper\n",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
            "conflict.py": "from external import use_fn\ndef resolve(): use_fn()\n",
        },
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={"helper": "placement.py"},
        forking_old_paths={"pkg.orig.use_fn"},
    )


def _make_bfs_index(test_src: str, calling_module: str = "pkg.test_mod") -> _CgIndex:
    """Minimal index: only the calling module's source (for import resolution)."""
    parts = calling_module.split(".")
    pkg = ".".join(parts[:-1]) if len(parts) > 1 else ""
    return _CgIndex(
        module_to_source={calling_module: test_src},
        module_to_package={calling_module: pkg},
        module_to_defs={calling_module: set()},
        file_to_module={},
    )


# ---------------------------------------------------------------------------
# _resolve_forking_path_via_callgraph — tests
# ---------------------------------------------------------------------------


def test_resolve_callgraph_no_calling_module():
    ctx = _make_bfs_ctx()
    index = _make_bfs_index("from pkg.placement import helper\n")
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper()\n", ctx, index, ""
    )
    assert result is None


def test_resolve_callgraph_pre_check_fails():
    # original_source has no external import of 'use_fn' → pre-check fails
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="def helper(): use_fn()\n",  # not imported externally
        modified_source="",
        new_files={"placement.py": "def helper(): use_fn()\n"},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    index = _make_bfs_index("from pkg.placement import helper\n")
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_no_terminal():
    # New files don't reference 'use_fn' → terminal empty → None
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": "def helper(): pass\n"},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    index = _make_bfs_index("from pkg.placement import helper\n")
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_direct_call():
    # Test directly calls 'helper'; helper in placement uses use_fn.
    ctx = _make_bfs_ctx()
    index = _make_bfs_index("from pkg.placement import helper\n")
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert result == "pkg.placement.use_fn"


def test_resolve_callgraph_multi_hop():
    # Test → intermediary → helper → terminal (placement.use_fn)
    ctx = _make_bfs_ctx()
    middle_src = "from pkg.placement import helper\ndef intermediary(): helper()\n"
    test_src = "from pkg.middle import intermediary\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.middle": middle_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.middle": "pkg"},
        module_to_defs={"pkg.test_mod": set(), "pkg.middle": {"intermediary"}},
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): intermediary()\n", ctx, index, "pkg.test_mod"
    )
    assert result == "pkg.placement.use_fn"


def test_resolve_callgraph_reexport():
    # Test imports helper from pkg.orig; pkg.orig re-exports helper from placement.
    # Re-export is followed without incrementing depth.
    ctx = _make_bfs_ctx()
    orig_src = "from .placement import helper\n"  # re-exports (fn not defined)
    test_src = "from pkg.orig import helper\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.orig": orig_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.orig": "pkg"},
        module_to_defs={"pkg.test_mod": set(), "pkg.orig": set()},  # helper not defined
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert result == "pkg.placement.use_fn"


def test_resolve_callgraph_multiple_candidates():
    # Both placement.helper and conflict.resolve are reachable → ambiguous → None.
    ctx = _make_bfs_ctx()
    test_src = "from pkg.placement import helper\n" "from pkg.conflict import resolve\n"
    index = _make_bfs_index(test_src)
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper(); resolve()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_not_reachable():
    # Test doesn't import anything relevant → BFS queue empty → None.
    ctx = _make_bfs_ctx()
    index = _make_bfs_index("")  # no imports
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): unrelated()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_new_submodule_non_terminal():
    # Test imports 'other' from placement; 'other' doesn't use use_fn.
    # 'other' is in a new sub-module but NOT in terminal → skipped.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": (
                "from external import use_fn\n"
                "def helper(): use_fn()\n"
                "def other(): pass\n"
            )
        },
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "from pkg.placement import other\n"
    index = _make_bfs_index(test_src)
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): other()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_init_reexport():
    # pkg/orig.py split into pkg/orig/__init__.py (re-exports helper) and
    # pkg/orig/placement.py (defines helper, uses use_fn).
    # Test imports helper from pkg.orig (the new __init__).
    # __init__ is excluded from new_module_set so BFS traverses through it
    # and follows the re-export to placement.py, finding the terminal.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "orig/__init__.py": "from .placement import helper\n",
            "orig/placement.py": (
                "from external import use_fn\ndef helper(): use_fn()\n"
            ),
        },
        new_module_paths={
            "orig/__init__.py": "pkg.orig",
            "orig/placement.py": "pkg.orig.placement",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    init_src = "from .placement import helper\n"
    test_src = "from pkg.orig import helper\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.orig": init_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.orig": "pkg.orig"},
        module_to_defs={"pkg.test_mod": set(), "pkg.orig": set()},
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert result == "pkg.orig.placement.use_fn"


def test_resolve_callgraph_visited_dedup():
    # 'intermediary' and 'inter2' both map to same (module, func); processed once.
    ctx = _make_bfs_ctx()
    middle_src = "from pkg.placement import helper\ndef intermediary(): helper()\n"
    test_src = (
        "from pkg.middle import intermediary\n"
        "from pkg.middle import intermediary as inter2\n"
    )
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.middle": middle_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.middle": "pkg"},
        module_to_defs={"pkg.test_mod": set(), "pkg.middle": {"intermediary"}},
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn",
        "def test_f(): intermediary(); inter2()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    assert result == "pkg.placement.use_fn"


def test_resolve_callgraph_empty_new_file():
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "empty.py": "",
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
        },
        new_module_paths={"empty.py": "pkg.empty", "placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    index = _make_bfs_index("from pkg.placement import helper\n")
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert result == "pkg.placement.use_fn"


def test_resolve_callgraph_missing_module_path():
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": "from external import use_fn\ndef f(): use_fn()\n"},
        new_module_paths={},  # missing → terminal empty → None
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    index = _make_bfs_index("from pkg.placement import f\n")
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): f()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_missing_source():
    # Called function's module is not in the index → src=None → continue
    ctx = _make_bfs_ctx()
    test_src = "from pkg.missing import something\n"
    index = _make_bfs_index(test_src)
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): something()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_func_defined_no_calls():
    # Function IS defined but has no imported calls → BFS dead-end → None
    ctx = _make_bfs_ctx()
    middle_src = "def standalone(): pass\n"
    test_src = "from pkg.middle import standalone\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.middle": middle_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.middle": "pkg"},
        module_to_defs={"pkg.test_mod": set(), "pkg.middle": {"standalone"}},
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): standalone()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_body_call_not_importable():
    # Function's body calls something not in its import map → BFS dead-end → None
    ctx = _make_bfs_ctx()
    middle_src = "def fn(): bar()\n"  # bar not imported
    test_src = "from pkg.middle import fn\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.middle": middle_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.middle": "pkg"},
        module_to_defs={"pkg.test_mod": set(), "pkg.middle": {"fn"}},
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): fn()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_func_not_defined_not_reexported():
    # func_name not defined and not re-exported → BFS dead-end → None
    ctx = _make_bfs_ctx()
    middle_src = "def other(): pass\n"  # 'fn' not defined, not re-exported
    test_src = "from pkg.middle import fn\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.middle": middle_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.middle": "pkg"},
        module_to_defs={"pkg.test_mod": set(), "pkg.middle": {"other"}},
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): fn()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_body_call_already_visited():
    # fn_a → fn_b → fn_a (mutual recursion); fn_a already visited when fn_b adds it
    ctx = _make_bfs_ctx()
    m_a = "from pkg.m_b import fn_b\ndef fn_a(): fn_b()\n"
    m_b = "from pkg.m_a import fn_a\ndef fn_b(): fn_a()\n"
    test_src = "from pkg.m_a import fn_a\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.m_a": m_a, "pkg.m_b": m_b},
        module_to_package={
            "pkg.test_mod": "pkg",
            "pkg.m_a": "pkg",
            "pkg.m_b": "pkg",
        },
        module_to_defs={
            "pkg.test_mod": set(),
            "pkg.m_a": {"fn_a"},
            "pkg.m_b": {"fn_b"},
        },
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): fn_a()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_reexport_already_visited():
    # m_x re-exports fn_x from m_y; m_y re-exports fn_x from m_x (cycle).
    # When m_y checks re-export, (m_x, fn_x) is already visited → skip.
    ctx = _make_bfs_ctx()
    m_x = "from pkg.m_y import fn_x\n"  # re-exports fn_x from m_y
    m_y = "from pkg.m_x import fn_x\n"  # re-exports fn_x from m_x (cycle)
    test_src = "from pkg.m_x import fn_x\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.m_x": m_x, "pkg.m_y": m_y},
        module_to_package={
            "pkg.test_mod": "pkg",
            "pkg.m_x": "pkg",
            "pkg.m_y": "pkg",
        },
        module_to_defs={"pkg.test_mod": set(), "pkg.m_x": set(), "pkg.m_y": set()},
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): fn_x()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_depth_limit():
    # Chain of _CG_MAX_DEPTH + 1 hops; last function calls terminal but is cut off.
    n = _CG_MAX_DEPTH + 1  # 13 intermediate functions
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"end.py": "from external import use_fn\ndef end_fn(): use_fn()\n"},
        new_module_paths={"end.py": "pkg.end"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    modules = {}
    for i in range(n):
        if i < n - 1:
            src = f"from pkg.m{i + 1} import f{i + 1}\ndef f{i}(): f{i + 1}()\n"
        else:
            src = f"from pkg.end import end_fn\ndef f{i}(): end_fn()\n"
        modules[f"pkg.m{i}"] = src
    modules["pkg.test_mod"] = "from pkg.m0 import f0\n"
    defs = {m: _cg_collect_defined_names(s) for m, s in modules.items()}
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs=defs,
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): f0()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_module_limit():
    # Re-export chain of _CG_MAX_MODULES + 1 unique modules; 51st is cut off.
    n = _CG_MAX_MODULES  # 50 re-export hops before the cut-off
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"end.py": "from external import use_fn\ndef final(): use_fn()\n"},
        new_module_paths={"end.py": "pkg.end"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    modules = {}
    for i in range(n + 1):
        if i < n:
            modules[f"pkg.m{i}"] = f"from pkg.m{i + 1} import fn\n"
        else:
            modules[f"pkg.m{i}"] = "from pkg.end import final\ndef fn(): final()\n"
    modules["pkg.test_mod"] = "from pkg.m0 import fn\n"
    defs = {m: _cg_collect_defined_names(s) for m, s in modules.items()}
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs=defs,
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): fn()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_same_module_twice():
    """Two called names both resolve to the same intermediate module.

    The second BFS entry hits the 'module already in modules_seen' fast path
    (branch 1250->1255 in patch_rewriter.py).
    """
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"end.py": "from external import use_fn\ndef final(): use_fn()\n"},
        new_module_paths={"end.py": "pkg.end"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    # Both fn_a and fn_b live in pkg.middle; neither calls anything reachable.
    middle_src = "def fn_a(): pass\ndef fn_b(): pass\n"
    modules = {
        "pkg.test_mod": "from pkg.middle import fn_a, fn_b\n",
        "pkg.middle": middle_src,
    }
    defs = {m: _cg_collect_defined_names(s) for m, s in modules.items()}
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs=defs,
        file_to_module={},
    )
    # BFS enqueues (pkg.middle, fn_a, 0) and (pkg.middle, fn_b, 0).
    # First pop adds pkg.middle to modules_seen; second pop hits the fast path.
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): fn_a(); fn_b()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


# ---------------------------------------------------------------------------
# _callgraph_update_file — helpers
# ---------------------------------------------------------------------------


def _make_cuf_contexts() -> list:
    """FL context with placement (helper) and conflict (resolve) using use_fn."""
    return [
        _FLContext(
            filepath="/proj/pkg/orig.py",
            old_module="pkg.orig",
            original_source="from external import use_fn\ndef helper(): use_fn()\n",
            modified_source="from .placement import helper\n",
            new_files={
                "placement.py": (
                    "from external import use_fn\ndef helper(): use_fn()\n"
                ),
                "conflict.py": (
                    "from external import use_fn\ndef resolve(): use_fn()\n"
                ),
            },
            new_module_paths={
                "placement.py": "pkg.placement",
                "conflict.py": "pkg.conflict",
            },
            entity_to_target={"helper": "placement.py"},
            forking_old_paths={"pkg.orig.use_fn"},
        )
    ]


def _make_cuf_index(scan_abs: str, test_src: str) -> _CgIndex:
    """Minimal index for _callgraph_update_file: maps scan_abs → 'pkg.test_mod'."""
    return _CgIndex(
        module_to_source={"pkg.test_mod": test_src},
        module_to_package={"pkg.test_mod": "pkg"},
        module_to_defs={"pkg.test_mod": set()},
        file_to_module={scan_abs: "pkg.test_mod"},
    )


# ---------------------------------------------------------------------------
# _callgraph_update_file — tests
# ---------------------------------------------------------------------------


def test_callgraph_update_file_no_functions():
    src = "x = 1\n"
    result, changed, _unresolved = _callgraph_update_file(
        src, {"pkg.orig.use_fn"}, _make_cuf_contexts()
    )
    assert not changed
    assert result == src


def test_callgraph_update_file_index_none(tmp_path):
    # index=None → BFS skipped → no resolution even if test calls helper.
    test_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=None,
    )
    assert not changed


def test_callgraph_update_file_string_literal_resolved(tmp_path):
    test_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.placement.use_fn")' in result


def test_callgraph_update_file_acc_cg_resolved(tmp_path):
    # _acc.cg_resolved incremented for each resolved path.
    test_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    acc = RewriteAccumulator()
    _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
        _acc=acc,
    )
    assert acc.cg_resolved == 1


def test_callgraph_update_file_no_resolution(tmp_path):
    # Test calls 'unrelated' — not imported → BFS queue empty → no resolution.
    # Static fallback has 2 candidates (placement + conflict) → unresolved saved.
    test_src = (
        '@patch("pkg.orig.use_fn")\n' "def test_f(mock_use_fn):\n" "    unrelated()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert not changed
    cands = unresolved.get("test_f", {}).get("pkg.orig.use_fn", [])
    assert sorted(cands) == ["pkg.conflict.use_fn", "pkg.placement.use_fn"]


def test_callgraph_update_file_zero_cands_single_static_auto_resolve(tmp_path):
    # BFS finds 0 candidates but static terminal has exactly 1 → auto-resolve.
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        '@patch("pkg.orig.use_fn")\n' "def test_f(mock_use_fn):\n" "    unrelated()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.placement.use_fn")' in result
    assert "test_f" not in unresolved


def test_callgraph_update_file_zero_cands_single_static_clears_unresolved(tmp_path):
    # ctx_ambig: BFS finds 2 candidates (saves to unresolved).
    # ctx_uniq_static: BFS finds 0, static has 1 → auto-resolves AND clears the
    # previously saved unresolved entry (exercises the delete-entry branch).
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
        "    resolve()\n"
    )
    ctx_ambig = _make_cuf_contexts()[0]  # placement + conflict → 2 BFS candidates
    ctx_uniq_static = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"singleton.py": "from external import use_fn\ndef fn(): use_fn()\n"},
        new_module_paths={"singleton.py": "pkg.singleton"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    scan = str(tmp_path / "test_foo.py")
    # Index only knows pkg.test_mod; pkg.placement/conflict have no source so
    # ctx_uniq_static's BFS reaches 0 candidates while static_cands = 1.
    index = _make_cuf_index(scan, test_src)
    result, changed, unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx_ambig, ctx_uniq_static],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.singleton.use_fn")' in result
    assert "test_f" not in unresolved  # static single-cand cleared the entry


def test_callgraph_update_file_const_ref_unanimous(tmp_path):
    test_src = (
        "from pkg.placement import helper\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_a(mock_use_fn):\n"
        "    helper()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_b(mock_use_fn):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '_PATCH_USE = "pkg.placement.use_fn"' in result
    assert "@patch(_PATCH_USE)" in result


def test_callgraph_update_file_const_ref_conflicting(tmp_path):
    # test_a: helper() → placement; test_b: resolve() → conflict → conflicting.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
            "conflict.py": "from external import use_fn\ndef resolve(): use_fn()\n",
        },
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_a(mock_use_fn):\n"
        "    helper()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_b(mock_use_fn):\n"
        "    resolve()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src, {"pkg.orig.use_fn"}, [ctx], scan_file=scan, index=index
    )
    assert changed
    assert '_PATCH_USE = "pkg.orig.use_fn"' in result  # const def unchanged
    assert '@patch("pkg.placement.use_fn")' in result  # test_a inlined
    assert '@patch("pkg.conflict.use_fn")' in result  # test_b inlined


def test_callgraph_update_file_non_forking_path_skipped(tmp_path):
    test_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.stable.some_func")\n'
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn, mock_some):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.placement.use_fn")' in result
    assert '@patch("pkg.stable.some_func")' in result  # unchanged


def test_callgraph_update_file_multi_context_second_matches(tmp_path):
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx_a = _FLContext(
        filepath="/proj/pkg/other.py",
        old_module="pkg.other",
        original_source="from external import other_fn\n",
        modified_source="",
        new_files={},
        new_module_paths={},
        entity_to_target={},
        forking_old_paths={"pkg.other.other_fn"},
    )
    ctx_b = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn", "pkg.other.other_fn"},
        [ctx_a, ctx_b],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.placement.use_fn")' in result


def test_callgraph_update_file_new_path_equals_old_path_continues(tmp_path):
    # Context 1 (outer subdir split): terminal includes intermediate __init__.py
    # which still has helper() calling use_fn → BFS returns old path as single
    # candidate.  The fix skips that "resolution" and tries context 2.
    # Context 2 (inner split): terminal maps helper → pkg.sub.use_fn (new path).
    intermediate_init = "from external import use_fn\ndef helper(): use_fn()\n"
    final_init = "from .sub import helper\n"
    sub_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx1 = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source=intermediate_init,
        modified_source=intermediate_init,
        new_files={
            "orig/__init__.py": intermediate_init,
            "orig/models.py": "class M: pass\n",
        },
        new_module_paths={
            "orig/__init__.py": "pkg.orig",
            "orig/models.py": "pkg.orig.models",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    ctx2 = _FLContext(
        filepath="/proj/pkg/orig/__init__.py",
        old_module="pkg.orig",
        original_source=intermediate_init,
        modified_source=final_init,
        new_files={"sub.py": sub_src},
        new_module_paths={"sub.py": "pkg.orig.sub"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.orig import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_f.py")
    index = _CgIndex(
        module_to_source={
            "pkg.orig": final_init,
            "pkg.orig.sub": sub_src,
            "pkg.test_mod": test_src,
        },
        module_to_package={
            "pkg.orig": "pkg.orig",
            "pkg.orig.sub": "pkg.orig",
            "pkg.test_mod": "pkg",
        },
        module_to_defs={
            "pkg.orig": set(),
            "pkg.orig.sub": {"helper"},
            "pkg.test_mod": set(),
        },
        file_to_module={scan: "pkg.test_mod"},
    )
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx1, ctx2],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.orig.sub.use_fn")' in result


def test_callgraph_update_file_static_fallback_old_path_continues(tmp_path):
    # BFS finds 0 reachable candidates in context 1, static_cands = [old_path].
    # The fix skips that no-op resolution and tries context 2, where
    # static_cands = [new_path] → auto-resolves to the new path.
    intermediate_init = "from external import use_fn\ndef helper(): use_fn()\n"
    sub_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx1 = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source=intermediate_init,
        modified_source=intermediate_init,
        new_files={"orig/__init__.py": intermediate_init},
        new_module_paths={"orig/__init__.py": "pkg.orig"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    ctx2 = _FLContext(
        filepath="/proj/pkg/orig/__init__.py",
        old_module="pkg.orig",
        original_source=intermediate_init,
        modified_source="from .sub import helper\n",
        new_files={"sub.py": sub_src},
        new_module_paths={"sub.py": "pkg.orig.sub"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    # Test calls unrelated() — BFS finds 0 reachable candidates in both contexts.
    test_src = '@patch("pkg.orig.use_fn")\n' "def test_f(m):\n" "    unrelated()\n"
    scan = str(tmp_path / "test_f.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx1, ctx2],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.orig.sub.use_fn")' in result


def test_callgraph_update_file_old_path_valid_casts_keep_old_vote(tmp_path):
    # worker stayed in pkg.orig (uses use_fn) → BFS returns old_path for test_a.
    # helper moved to pkg.sub (uses use_fn) → BFS returns pkg.sub.use_fn for test_b.
    # Both share _PATCH_USE const.  The keep-old vote from test_a must create a
    # conflict so the const definition is NOT silently rewritten to pkg.sub.use_fn
    # (which would break test_a).  test_b should be inlined instead.
    orig_src = (
        "from external import use_fn\ndef helper(): use_fn()\ndef worker(): use_fn()\n"
    )
    # After split: worker stayed, helper moved.
    modified_orig = (
        "from external import use_fn\n"
        "from .sub import helper\n"
        "def worker(): use_fn()\n"
    )
    sub_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source=orig_src,
        modified_source=modified_orig,
        new_files={"sub.py": sub_src},
        new_module_paths={"sub.py": "pkg.sub"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.orig import worker, helper\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_a(m):\n"
        "    worker()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_b(m):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # Index: pkg.orig = modified source (re-exports helper from sub), pkg.sub = sub_src.
    index = _CgIndex(
        module_to_source={
            "pkg.orig": modified_orig,
            "pkg.sub": sub_src,
            "pkg.test_mod": test_src,
        },
        module_to_package={
            "pkg.orig": "pkg",
            "pkg.sub": "pkg",
            "pkg.test_mod": "pkg",
        },
        module_to_defs={
            "pkg.orig": {"worker"},
            "pkg.sub": {"helper"},
            "pkg.test_mod": set(),
        },
        file_to_module={scan: "pkg.test_mod"},
    )
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert (
        '_PATCH_USE = "pkg.orig.use_fn"' in result
    )  # const def NOT updated (conflict)
    assert "@patch(_PATCH_USE)" in result  # test_a still uses const (not inlined)
    assert '@patch("pkg.sub.use_fn")' in result  # test_b inlined to new path


def test_callgraph_update_file_const_ref_no_resolution_passthrough(tmp_path):
    test_src = (
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_f(mock_use_fn):\n"
        "    unrelated()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert not changed
    assert '_PATCH_USE = "pkg.orig.use_fn"' in result


def test_callgraph_update_file_const_ref_passthrough_single_proposal_updates_const(
    tmp_path,
):
    # test_a: BFS fails (calls unrelated()) → passthrough (if not resolved → continue).
    # test_b: BFS → placement → single proposal for _PATCH_USE.
    # Old: passthrough + single proposal → conflicting → inline test_b.
    # New: single proposal (passthrough no longer blocks) → const def updated, no
    #   inline.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
        },
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_a(m):\n"
        "    unrelated()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_b(m):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src, {"pkg.orig.use_fn"}, [ctx], scan_file=scan, index=index
    )
    assert changed
    # Const definition updated (single proposal, passthrough no longer blocks).
    assert '_PATCH_USE = "pkg.placement.use_fn"' in result
    # Decorators stay as const refs — no per-function inlining.
    assert "@patch(_PATCH_USE)" in result
    assert '@patch("pkg.placement.use_fn")' not in result


def test_callgraph_update_file_const_ref_partial_resolution(tmp_path):
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn, other_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n"
        },
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn", "pkg.orig.other_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        '_PATCH_OTHER = "pkg.orig.other_fn"\n'
        "@patch(_PATCH_OTHER)\n"
        "@patch(_PATCH_USE)\n"
        "def test_f(mock_use, mock_other):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn", "pkg.orig.other_fn"},
        [ctx],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '_PATCH_USE = "pkg.placement.use_fn"' in result
    assert '_PATCH_OTHER = "pkg.orig.other_fn"' in result


def test_callgraph_update_file_inline_no_inline_subs_continue(tmp_path):
    # test_a: string literal (no const_refs → inline_subs empty → continue)
    # test_b: const ref → placement; test_c: const ref → conflict → conflicting
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
            "conflict.py": "from external import use_fn\ndef resolve(): use_fn()\n",
        },
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_a(m):\n"
        "    helper()\n"
        "\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_b(m):\n"
        "    helper()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_c(m):\n"
        "    resolve()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src, {"pkg.orig.use_fn"}, [ctx], scan_file=scan, index=index
    )
    assert changed
    assert '@patch("pkg.placement.use_fn")' in result  # test_a updated
    assert '@patch("pkg.conflict.use_fn")' in result  # test_c inlined


def test_callgraph_update_file_inline_ref_from_different_file(tmp_path):
    # Const ref from constants.py (≠ scan_file) → inline_subs empty → no change.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
            "conflict.py": "from external import use_fn\ndef resolve(): use_fn()\n",
        },
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('_PATCH_USE = "pkg.orig.use_fn"\n', encoding="utf-8")
    test_src = (
        "from constants import _PATCH_USE\n"
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        "@patch(_PATCH_USE)\n"
        "def test_b(m):\n"
        "    helper()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_c(m):\n"
        "    resolve()\n"
    )
    scan_file = tmp_path / "test_cases.py"
    scan_file.write_text(test_src, encoding="utf-8")
    scan = str(scan_file)
    # Build index from disk so file_to_module is populated for test_cases.py
    index = _cg_build_index(str(tmp_path), {}, [ctx])
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx],
        scan_file=scan,
        repo_root=str(tmp_path),
        index=index,
    )
    assert not changed


def test_callgraph_update_file_inline_new_val_same_as_old(tmp_path):
    # placement.py → "pkg.orig" (same as old_module); test_b→helper→same val; skipped.
    # test_c → resolve → "pkg.conflict" → different val → inlined → changed.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
            "conflict.py": "from external import use_fn\ndef resolve(): use_fn()\n",
        },
        new_module_paths={
            "placement.py": "pkg.orig",  # same as old_module → new_val == old_val
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.orig import helper\n"
        "from pkg.conflict import resolve\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_b(m):\n"
        "    helper()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_c(m):\n"
        "    resolve()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src, {"pkg.orig.use_fn"}, [ctx], scan_file=scan, index=index
    )
    assert changed  # test_c inlined to pkg.conflict.use_fn


def test_callgraph_update_file_inline_existing_splice_updated(tmp_path):
    # test_a: use_fn → func_splice; other_fn const ref conflicting → inline.
    # Inline finds existing splice and updates it.  test_b: const → new splice.
    placement_src = (
        "from external import use_fn, other_fn\n" "def helper(): use_fn(); other_fn()\n"
    )
    conflict2_src = "from external import other_fn\ndef resolve2(): other_fn()\n"
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn, other_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src, "conflict2.py": conflict2_src},
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict2.py": "pkg.conflict2",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn", "pkg.orig.other_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict2 import resolve2\n"
        '_PATCH_OTHER = "pkg.orig.other_fn"\n'
        '@patch("pkg.orig.use_fn")\n'
        "@patch(_PATCH_OTHER)\n"
        "def test_a(m_other, m_use):\n"
        "    helper()\n"
        "\n"
        "@patch(_PATCH_OTHER)\n"
        "def test_b(m_other):\n"
        "    resolve2()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn", "pkg.orig.other_fn"},
        [ctx],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.placement.use_fn")' in result
    assert '@patch("pkg.placement.other_fn")' in result
    assert '@patch("pkg.conflict2.other_fn")' in result


def test_callgraph_update_file_verbose(tmp_path, capsys):
    test_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "patch_callgraph" in captured.err


def test_callgraph_update_file_truncated_warns(tmp_path, capsys):
    # Depth limit of 0 forces truncation for indirect calls; warning must be printed.
    # Test calls an intermediate function (not a terminal); with max_depth=0 the
    # first BFS hop immediately hits the limit before reaching the terminal.
    test_src = (
        "from pkg.middle import middle_fn\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn):\n"
        "    middle_fn()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    scan_abs = str((tmp_path / "test_foo.py").resolve())
    # middle_fn → helper (terminal in pkg.placement), but BFS cuts off before that.
    middle_src = "from pkg.placement import helper\ndef middle_fn(): helper()\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.middle": middle_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.middle": "pkg"},
        module_to_defs={
            "pkg.test_mod": set(),
            "pkg.middle": _cg_collect_defined_names(middle_src),
        },
        file_to_module={scan_abs: "pkg.test_mod"},
    )
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
        max_depth=0,
    )
    assert not changed
    captured = capsys.readouterr()
    assert "traversal limit reached" in captured.err
    assert "pkg.orig.use_fn" in captured.err


# ---------------------------------------------------------------------------
# _resolve_forking_path_candidates — full result (truncation / candidates)
# ---------------------------------------------------------------------------


def test_resolve_forking_path_candidates_single():
    # Single candidate: path returned, candidates=[path], truncated=False.
    ctx = _make_bfs_ctx()
    index = _make_bfs_index("from pkg.placement import helper\n")
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert path == "pkg.placement.use_fn"
    assert cands == ["pkg.placement.use_fn"]
    assert not truncated


def test_resolve_forking_path_candidates_multiple():
    # Multiple candidates → path=None, cands=[...], truncated=False.
    ctx = _make_bfs_ctx()
    test_src = "from pkg.placement import helper\nfrom pkg.conflict import resolve\n"
    index = _make_bfs_index(test_src)
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): helper(); resolve()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    assert path is None
    assert sorted(cands) == ["pkg.conflict.use_fn", "pkg.placement.use_fn"]
    assert not truncated


def test_resolve_forking_path_candidates_no_calling_module():
    ctx = _make_bfs_ctx()
    index = _make_bfs_index("from pkg.placement import helper\n")
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn", "def test_f(): helper()\n", ctx, index, ""
    )
    assert path is None
    assert cands == []
    assert not truncated


def test_resolve_forking_path_candidates_truncated_depth():
    # Chain of exactly _CG_MAX_DEPTH + 1 hops; the last hop is cut off → truncated=True.
    # Chain: test_mod -[f0]-> mid0 -> mid1 -> ... -> mid{n-1} -[helper]-> placement
    # n = _CG_MAX_DEPTH + 1 intermediate modules; helper is at depth n-1 = 13,
    # but the depth limit cuts off at depth 12 before enqueuing helper.
    n = _CG_MAX_DEPTH + 1  # 13 hops from test_mod to placement
    ctx = _make_bfs_ctx()
    all_src: dict = {}
    all_src["pkg.test_mod"] = "from pkg.mid0 import f0\n"
    for i in range(n):
        caller = f"f{i}"
        if i < n - 1:
            callee = f"f{i + 1}"
            callee_mod = f"pkg.mid{i + 1}"
        else:
            callee = "helper"
            callee_mod = "pkg.placement"
        all_src[f"pkg.mid{i}"] = (
            f"from {callee_mod} import {callee}\n" f"def {caller}(): {callee}()\n"
        )
    all_src["pkg.placement"] = "from external import use_fn\ndef helper(): use_fn()\n"
    index = _CgIndex(
        module_to_source=all_src,
        module_to_package={m: "pkg" for m in all_src},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in all_src.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn", "def test_f(): f0()\n", ctx, index, "pkg.test_mod"
    )
    assert path is None
    assert truncated


def test_resolve_forking_path_candidates_truncated_modules():
    # Re-export chain of _CG_MAX_MODULES + 1 intermediate modules; the last one
    # is cut off before pkg.placement (a terminal) is ever reached.
    n = _CG_MAX_MODULES + 1
    ctx = _make_bfs_ctx()
    src_map: dict = {}
    for i in range(n):
        next_mod = f"pkg.m{i + 1}" if i < n - 1 else "pkg.placement"
        src_map[f"pkg.m{i}"] = f"from {next_mod} import helper\n"
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    src_map["pkg.placement"] = placement_src
    test_src = "from pkg.m0 import helper\n"
    all_src = {"pkg.test_mod": test_src, **src_map}
    index = _CgIndex(
        module_to_source=all_src,
        module_to_package={m: "pkg" for m in all_src},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in all_src.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert path is None
    assert truncated


def test_resolve_forking_path_candidates_original_module_only():
    # modified_source still has a function using use_fn; no new sub-file uses it.
    # → only terminal is (pkg.orig, func_a) → unique resolution to original path.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source=("from external import use_fn\n" "def func_a(): use_fn()\n"),
        new_files={
            "placement.py": "from external import other\ndef helper(): other()\n"
        },
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    orig_src = ctx.modified_source
    test_src = "from pkg.orig import func_a\n"
    modules = {"pkg.test_mod": test_src, "pkg.orig": orig_src}
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn", "def test_f(): func_a()\n", ctx, index, "pkg.test_mod"
    )
    assert path == "pkg.orig.use_fn"
    assert cands == ["pkg.orig.use_fn"]
    assert not truncated


def test_resolve_forking_path_candidates_original_and_new_both_candidates():
    # modified_source keeps func_a (uses use_fn); placement.py moves func_b
    # (also uses use_fn).  Test calls both → 2 candidates → ambiguous.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source=("from external import use_fn\n" "def func_a(): use_fn()\n"),
        new_files={
            "placement.py": (
                "from external import use_fn\n" "def func_b(): use_fn()\n"
            ),
        },
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    orig_src = ctx.modified_source
    placement_src = ctx.new_files["placement.py"]
    test_src = "from pkg.orig import func_a\nfrom pkg.placement import func_b\n"
    modules = {
        "pkg.test_mod": test_src,
        "pkg.orig": orig_src,
        "pkg.placement": placement_src,
    }
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): func_a(); func_b()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    assert path is None  # ambiguous
    assert sorted(cands) == ["pkg.orig.use_fn", "pkg.placement.use_fn"]
    assert not truncated


# ---------------------------------------------------------------------------
# _expand_module_terminals
# ---------------------------------------------------------------------------


def test_expand_module_terminals_no_direct():
    # No direct terminal in this module → nothing added.
    terminal: dict = {}
    _expand_module_terminals(
        "def a(): b()\ndef b(): pass\n", "pkg.mod", "use_fn", terminal
    )
    assert terminal == {}


def test_expand_module_terminals_direct_only():
    # A direct terminal is seeded before calling; only transitive callers are added.
    terminal: dict = {("pkg.mod", "b"): "pkg.mod.use_fn"}
    _expand_module_terminals(
        "def a(): b()\ndef b(): use_fn()\n", "pkg.mod", "use_fn", terminal
    )
    # a calls b (direct terminal) → a becomes transitive terminal.
    assert terminal[("pkg.mod", "a")] == "pkg.mod.use_fn"
    # Original direct entry unchanged.
    assert terminal[("pkg.mod", "b")] == "pkg.mod.use_fn"


def test_expand_module_terminals_multi_level():
    # c → b → a (direct); all three end up in terminal.
    terminal: dict = {("pkg.mod", "a"): "pkg.mod.use_fn"}
    src = "def a(): use_fn()\ndef b(): a()\ndef c(): b()\n"
    _expand_module_terminals(src, "pkg.mod", "use_fn", terminal)
    assert ("pkg.mod", "b") in terminal
    assert ("pkg.mod", "c") in terminal


def test_expand_module_terminals_syntax_error():
    # Unparseable source → silently returns without modifying terminal.
    terminal: dict = {("pkg.mod", "a"): "pkg.mod.use_fn"}
    _expand_module_terminals("def (broken\n", "pkg.mod", "use_fn", terminal)
    # Only the original entry remains.
    assert list(terminal.keys()) == [("pkg.mod", "a")]


def test_expand_module_terminals_unrelated_module():
    # Direct terminal is in a different module → nothing added for pkg.other.
    terminal: dict = {("pkg.mod", "a"): "pkg.mod.use_fn"}
    _expand_module_terminals("def b(): a()\n", "pkg.other", "use_fn", terminal)
    # b is in pkg.other which has no direct terminals → not added.
    assert ("pkg.other", "b") not in terminal


# ---------------------------------------------------------------------------
# BFS local-call following (intra-module)
# ---------------------------------------------------------------------------


def test_resolve_forking_path_candidates_intra_module_chain():
    # BFS follows locally-defined calls within a non-terminal intermediate module.
    # Chain: test_mod → pkg.service.public_func
    #                        (local) ↓
    #                   pkg.service._local_helper
    #                        (import) ↓
    #                   pkg.placement.use_target  ← terminal (calls use_fn)
    #
    # pkg.service is neither orig nor a new sub-file, so _expand_module_terminals
    # never seeds it.  The elif branch in the BFS must queue _local_helper from
    # public_func's body so we eventually reach the terminal in pkg.placement.
    placement_src = "from external import use_fn\ndef use_target(): use_fn()\n"
    service_src = (
        "from pkg.placement import use_target\n"
        "def _local_helper(): use_target()\n"
        "def public_func(): _local_helper()\n"
    )
    ctx2 = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="from .placement import use_target\n",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={"use_target": "placement.py"},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "from pkg.service import public_func\n"
    modules = {
        "pkg.test_mod": test_src,
        "pkg.service": service_src,
        "pkg.placement": placement_src,
    }
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): public_func()\n",
        ctx2,
        index,
        "pkg.test_mod",
    )
    assert path == "pkg.placement.use_fn"
    assert cands == ["pkg.placement.use_fn"]
    assert not truncated


def test_resolve_forking_path_candidates_intra_module_local_already_visited():
    # The elif branch fires but the visited guard suppresses re-queuing.
    # A recursive function calls itself: when processing its body calls, itself
    # is already in visited → (module, called_name) in visited → branch skipped.
    placement_src = "from external import use_fn\ndef use_target(): use_fn()\n"
    service_src = (
        "from pkg.placement import use_target\n"
        # recursive_func calls use_target (imported) AND itself (local, recursive)
        "def recursive_func(n): use_target() if n <= 0 else recursive_func(n-1)\n"
    )
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="from .placement import use_target\n",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={"use_target": "placement.py"},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "from pkg.service import recursive_func\n"
    modules = {
        "pkg.test_mod": test_src,
        "pkg.service": service_src,
        "pkg.placement": placement_src,
    }
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): recursive_func(5)\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    # recursive_func's body calls reach use_target (terminal in pkg.placement).
    assert path == "pkg.placement.use_fn"
    assert not truncated


def test_resolve_forking_path_candidates_new_module_intra_chain():
    # BFS follows intra-module calls within a new submodule to reach a terminal.
    # Chain: test_mod → pkg.placement.wrapper (new-module, not terminal)
    #                        (local) ↓
    #                   pkg.placement._inner  ← terminal (calls use_fn directly)
    #
    # Before the fix, the BFS hit pkg.placement in new_module_set and stopped at
    # wrapper without following _inner — no candidate was found.  After the fix,
    # it follows the local call to _inner and discovers pkg.placement.use_fn.
    placement_src = (
        "from external import use_fn\n"
        "def _inner(): use_fn()\n"
        "def wrapper(): _inner()\n"
    )
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={"wrapper": "placement.py", "_inner": "placement.py"},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "from pkg.placement import wrapper\n"
    modules = {
        "pkg.test_mod": test_src,
        "pkg.placement": placement_src,
    }
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): wrapper()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    assert path == "pkg.placement.use_fn"
    assert cands == ["pkg.placement.use_fn"]
    assert not truncated


def test_resolve_forking_path_candidates_new_module_intra_chain_cycle():
    # Intra-module traversal inside a new submodule respects the visited guard:
    # a mutually recursive pair (a calls b, b calls a) does not loop.
    placement_src = (
        "from external import use_fn\n"
        "def _inner(): use_fn()\n"
        "def a(): b()\n"
        "def b(): a(); _inner()\n"  # b is terminal (uses use_fn via _inner)
    )
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={"a": "placement.py"},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "from pkg.placement import a\n"
    modules = {
        "pkg.test_mod": test_src,
        "pkg.placement": placement_src,
    }
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): a()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    assert path == "pkg.placement.use_fn"
    assert not truncated


def test_resolve_forking_path_candidates_new_module_cross_module_terminal():
    # BFS follows cross-module calls FROM a terminal function inside a new submodule.
    # Scenario:
    #   pkg.main:  _run_step() calls use_fn() directly (terminal)
    #              orchestrate() calls _run_step() [local] + do_step() [pkg.steps]
    #              _expand_module_terminals makes orchestrate terminal for pkg.main.use_fn  # noqa: E501
    #   pkg.steps: do_step() calls use_fn() (terminal for pkg.steps.use_fn)
    #
    # When BFS hits (pkg.main, orchestrate) — which IS in terminal — it should
    # record pkg.main.use_fn AND then follow the cross-module call to
    # (pkg.steps, do_step), discovering pkg.steps.use_fn as a second candidate.
    # Line 1472 in the BFS is covered only by this cross-module append.
    main_src = (
        "from external import use_fn\n"
        "from pkg.steps import do_step\n"
        "def _run_step(): use_fn()\n"
        "def orchestrate(): _run_step(); do_step()\n"
    )
    steps_src = "from external import use_fn\n" "def do_step(): use_fn()\n"
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"main.py": main_src, "steps.py": steps_src},
        new_module_paths={"main.py": "pkg.main", "steps.py": "pkg.steps"},
        entity_to_target={
            "_run_step": "main.py",
            "orchestrate": "main.py",
            "do_step": "steps.py",
        },
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "from pkg.main import orchestrate\n"
    modules = {
        "pkg.test_mod": test_src,
        "pkg.main": main_src,
        "pkg.steps": steps_src,
    }
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): orchestrate()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    # Both submodules use use_fn — two candidates, no single resolved path.
    assert path is None
    assert sorted(cands) == ["pkg.main.use_fn", "pkg.steps.use_fn"]
    assert not truncated


# ---------------------------------------------------------------------------
# BFS — import alias traversal
# ---------------------------------------------------------------------------


def test_resolve_forking_path_candidates_import_alias_direct():
    # Test uses ``import pkg.placement as pl; pl.helper()`` to call the terminal.
    # The BFS must follow ``pl.helper`` by resolving alias ``pl`` → ``pkg.placement``.
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="from .placement import helper\n",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={"helper": "placement.py"},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "import pkg.placement as pl\n"
    # Import map: "pl" → ("pkg.placement", "pkg.placement"); call "pl.helper()"
    # → _cg_collect_called_names emits "pl.helper"; BFS resolves alias pl →
    # module pkg.placement, queues (pkg.placement, "helper") → terminal hit.
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src},
        module_to_package={"pkg.test_mod": "pkg"},
        module_to_defs={"pkg.test_mod": set()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): pl.helper()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    assert path == "pkg.placement.use_fn"
    assert cands == ["pkg.placement.use_fn"]
    assert not truncated


def test_resolve_forking_path_candidates_body_call_via_alias():
    # An intermediate function uses ``mod.helper()`` (module alias) to reach
    # the terminal.  The BFS body-call step must follow the alias.
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    service_src = "import pkg.placement as pl\n" "def public_func(): pl.helper()\n"
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="from .placement import helper\n",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={"helper": "placement.py"},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "from pkg.service import public_func\n"
    modules = {
        "pkg.test_mod": test_src,
        "pkg.service": service_src,
        "pkg.placement": placement_src,
    }
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): public_func()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    assert path == "pkg.placement.use_fn"
    assert cands == ["pkg.placement.use_fn"]
    assert not truncated


# ---------------------------------------------------------------------------
# _candidates_check
# ---------------------------------------------------------------------------


def test_candidates_check_no_candidates():
    # No candidates for any path → None.
    assert _candidates_check({"pkg.orig.A": "pkg.sub.A"}, ["pkg.orig.A"], {}) is None


def test_candidates_check_rename_valid():
    # Rename is in candidates → None.
    cands = {"pkg.orig.A": ["pkg.placement.A", "pkg.helpers.A"]}
    assert (
        _candidates_check({"pkg.orig.A": "pkg.placement.A"}, ["pkg.orig.A"], cands)
        is None
    )


def test_candidates_check_rename_invalid():
    # Rename proposes a path not in candidates → error message.
    cands = {"pkg.orig.A": ["pkg.placement.A"]}
    result = _candidates_check({"pkg.orig.A": "pkg.wrong.A"}, ["pkg.orig.A"], cands)
    assert result is not None
    assert "pkg.wrong.A" in result
    assert "pkg.placement.A" in result


def test_candidates_check_no_change_with_candidates():
    # No rename proposed for a path that has candidates → error message.
    cands = {"pkg.orig.A": ["pkg.placement.A"]}
    result = _candidates_check({}, ["pkg.orig.A"], cands)
    assert result is not None
    assert "pkg.orig.A" in result
    assert "pkg.placement.A" in result


def test_candidates_check_path_not_in_candidates():
    # Another path has no candidates → passes; only paths with candidates are checked.
    cands = {"pkg.orig.A": ["pkg.placement.A"]}
    # pkg.orig.B has no candidates; even though no rename proposed → None
    assert _candidates_check({}, ["pkg.orig.B"], cands) is None


def test_candidates_check_no_change_when_old_in_candidates():
    # No rename proposed but old path is itself one of the candidates (e.g. the entity
    # is still accessible at the original module via __init__.py re-export) → None.
    cands = {"pkg.orig.A": ["pkg.orig.A", "pkg.resolver.A"]}
    assert _candidates_check({}, ["pkg.orig.A"], cands) is None


# ---------------------------------------------------------------------------
# _callgraph_update_file — candidates collected when multiple found
# ---------------------------------------------------------------------------


def test_callgraph_update_file_multiple_candidates_saved(tmp_path):
    # Both placement and conflict are reachable → 2 candidates → saved.
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
        "    resolve()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert not changed  # ambiguous → no update
    assert "test_f" in unresolved
    assert "pkg.orig.use_fn" in unresolved["test_f"]
    cands = unresolved["test_f"]["pkg.orig.use_fn"]
    assert sorted(cands) == ["pkg.conflict.use_fn", "pkg.placement.use_fn"]


def test_callgraph_update_file_resolved_clears_candidates(tmp_path):
    # Single ctx with unique resolution → no candidates saved.
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert "test_f" not in unresolved  # unique resolution → no candidates saved


def test_callgraph_update_file_resolved_clears_function_entry(tmp_path):
    # ctx_ambig gives 2 candidates (saves to unresolved); ctx_uniq resolves uniquely →
    # unresolved entry for the function is deleted (line 2695).
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
        "    resolve()\n"
    )
    ctx_ambig = _make_cuf_contexts()[0]  # both placement and conflict → 2 candidates
    ctx_uniq = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="from .placement import helper\n",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
        },
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx_ambig, ctx_uniq],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert "test_f" not in unresolved  # ctx_uniq resolved → entry deleted


# ---------------------------------------------------------------------------
# apply_patch_callgraph — candidates_out parameter
# ---------------------------------------------------------------------------


def test_apply_patch_callgraph_candidates_out_per_file(tmp_path):
    # Multiple candidates → saved in candidates_out for per_file entry.
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
        "    resolve()\n"
    )
    test_file = tmp_path / "test_orig.py"
    test_file.write_text(test_src, encoding="utf-8")
    per_file = {str(test_file): {"source": test_src, "msgs": []}}
    candidates_out: dict = {}
    list(
        apply_patch_callgraph(
            _make_cuf_contexts(), per_file, str(tmp_path), candidates_out=candidates_out
        )
    )
    abs_fp = str(test_file.resolve())
    assert abs_fp in candidates_out
    assert "test_f" in candidates_out[abs_fp]


def test_apply_patch_callgraph_candidates_out_disk_file(tmp_path):
    # Multiple candidates → saved in candidates_out for disk file.
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
        "    resolve()\n"
    )
    test_file = tmp_path / "test_orig.py"
    test_file.write_text(test_src, encoding="utf-8")
    candidates_out: dict = {}
    list(
        apply_patch_callgraph(
            _make_cuf_contexts(), {}, str(tmp_path), candidates_out=candidates_out
        )
    )
    abs_fp = str(test_file.resolve())
    assert abs_fp in candidates_out
    assert "test_f" in candidates_out[abs_fp]


# ---------------------------------------------------------------------------
# Prompt builders — candidates_per_path parameter
# ---------------------------------------------------------------------------


def _make_fl_ctx_simple():
    """Minimal FLContext for prompt builder tests."""
    return _FLContext(
        filepath="/repo/pkg/big.py",
        old_module="pkg.big",
        original_source="from external import A\ndef f(): A()\n",
        modified_source="from .sub_a import f\n",
        new_files={"sub_a.py": "from external import A\ndef f(): A()\n"},
        new_module_paths={"sub_a.py": "pkg.sub_a"},
        entity_to_target={"f": "sub_a.py"},
        forking_old_paths={"pkg.big.A"},
    )


def test_build_classify_prompt_with_candidates():
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    prompt = _build_classify_prompt(
        context_msg,
        "def test_f(): pass\n",
        ["pkg.big.A"],
        candidates_per_path={"pkg.big.A": ["pkg.sub_a.A", "pkg.sub_b.A"]},
    )
    assert "Call-graph candidate paths" in prompt
    assert "pkg.sub_a.A" in prompt
    assert "pkg.sub_b.A" in prompt


def test_build_classify_prompt_candidates_above_threshold():
    # Candidates count > threshold → section not included.
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    many_cands = [f"pkg.sub_{i}.A" for i in range(_CG_CANDIDATES_LLM_THRESHOLD + 1)]
    prompt = _build_classify_prompt(
        context_msg,
        "def test_f(): pass\n",
        ["pkg.big.A"],
        candidates_per_path={"pkg.big.A": many_cands},
    )
    assert "Call-graph candidate paths" not in prompt


def test_build_func_verify_prompt_with_candidates():
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    prompt = _build_func_verify_prompt(
        context_msg,
        "def test_f(): pass\n",
        {"pkg.big.A": "pkg.sub_a.A"},
        candidates_per_path={"pkg.big.A": ["pkg.sub_a.A", "pkg.sub_b.A"]},
    )
    assert "Call-graph candidate paths" in prompt
    assert "pkg.sub_a.A" in prompt


def test_build_func_verify_prompt_candidates_above_threshold():
    # All candidate lists exceed the threshold → section not included.
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    many_cands = [f"pkg.sub_{i}.A" for i in range(_CG_CANDIDATES_LLM_THRESHOLD + 1)]
    prompt = _build_func_verify_prompt(
        context_msg,
        "def test_f(): pass\n",
        {"pkg.big.A": "pkg.sub_a.A"},
        candidates_per_path={"pkg.big.A": many_cands},
    )
    assert "Call-graph candidate paths" not in prompt


def test_build_no_change_verify_prompt_with_candidates():
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    prompt = _build_no_change_verify_prompt(
        context_msg,
        "def test_f(): pass\n",
        ["pkg.big.A"],
        candidates_per_path={"pkg.big.A": ["pkg.sub_a.A"]},
    )
    assert "Call-graph candidate paths" in prompt
    assert "pkg.sub_a.A" in prompt


def test_build_no_change_verify_prompt_candidates_above_threshold():
    # All candidate lists exceed the threshold → section not included.
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    many_cands = [f"pkg.sub_{i}.A" for i in range(_CG_CANDIDATES_LLM_THRESHOLD + 1)]
    prompt = _build_no_change_verify_prompt(
        context_msg,
        "def test_f(): pass\n",
        ["pkg.big.A"],
        candidates_per_path={"pkg.big.A": many_cands},
    )
    assert "Call-graph candidate paths" not in prompt


def test_build_rewrite_func_prompt_with_candidates():
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    prompt = _build_rewrite_func_prompt(
        context_msg,
        "def test_f(): pass\n",
        ["pkg.big.A"],
        candidates_per_path={"pkg.big.A": ["pkg.sub_a.A", "pkg.helpers.A"]},
    )
    assert "Call-graph candidate paths" in prompt
    assert "pkg.sub_a.A" in prompt
    assert "pkg.helpers.A" in prompt


def test_build_rewrite_func_prompt_candidates_above_threshold():
    # All candidate lists exceed the threshold → section not included.
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    many_cands = [f"pkg.sub_{i}.A" for i in range(_CG_CANDIDATES_LLM_THRESHOLD + 1)]
    prompt = _build_rewrite_func_prompt(
        context_msg,
        "def test_f(): pass\n",
        ["pkg.big.A"],
        candidates_per_path={"pkg.big.A": many_cands},
    )
    assert "Call-graph candidate paths" not in prompt


# ---------------------------------------------------------------------------
# _patch_strings_in_text
# ---------------------------------------------------------------------------


def test_patch_strings_in_text_decorator():
    text = '@patch("pkg.mod.A")\ndef test_f(m): pass\n'
    assert _patch_strings_in_text(text) == {"pkg.mod.A"}


def test_patch_strings_in_text_attribute_decorator():
    text = '@mock.patch("pkg.mod.B")\ndef test_f(m): pass\n'
    assert _patch_strings_in_text(text) == {"pkg.mod.B"}


def test_patch_strings_in_text_context_manager():
    text = 'def test_f():\n    with patch("pkg.mod.C") as m: pass\n'
    assert _patch_strings_in_text(text) == {"pkg.mod.C"}


def test_patch_strings_in_text_multiple():
    text = (
        '@patch("pkg.mod.A")\n' '@mock.patch("pkg.mod.B")\n' "def test_f(a, b): pass\n"
    )
    assert _patch_strings_in_text(text) == {"pkg.mod.A", "pkg.mod.B"}


def test_patch_strings_in_text_empty():
    assert _patch_strings_in_text("def test_f(): pass\n") == set()


# ---------------------------------------------------------------------------
# _rewrite_candidates_check
# ---------------------------------------------------------------------------


def test_rewrite_candidates_check_no_candidates():
    # No candidates for any path → None.
    text = '@patch("pkg.mod.A")\ndef test_f(m): pass\n'
    assert _rewrite_candidates_check(["pkg.mod.A"], text, {}) is None


def test_rewrite_candidates_check_valid_rename():
    # Old path absent, one candidate present → None.
    text = '@patch("pkg.placement.A")\ndef test_f(m): pass\n'
    cands = {"pkg.mod.A": ["pkg.placement.A", "pkg.other.A"]}
    assert _rewrite_candidates_check(["pkg.mod.A"], text, cands) is None


def test_rewrite_candidates_check_old_still_present():
    # Old path still present even though candidates exist → error.
    text = '@patch("pkg.mod.A")\ndef test_f(m): pass\n'
    cands = {"pkg.mod.A": ["pkg.placement.A"]}
    result = _rewrite_candidates_check(["pkg.mod.A"], text, cands)
    assert result is not None
    assert "pkg.mod.A" in result
    assert "pkg.placement.A" in result


def test_rewrite_candidates_check_renamed_to_unknown():
    # Old path absent, no known candidate appears — could be a wrong rename or a
    # dead-code removal. Let the LLM verify step decide; no error returned here.
    text = '@patch("pkg.wrong.A")\ndef test_f(m): pass\n'
    cands = {"pkg.mod.A": ["pkg.placement.A", "pkg.other.A"]}
    assert _rewrite_candidates_check(["pkg.mod.A"], text, cands) is None


def test_rewrite_candidates_check_deleted_patch():
    # Old path absent and decorator was removed entirely → dead-code removal is
    # allowed; let the LLM verify step confirm correctness.
    text = "def test_f(): pass\n"
    cands = {"pkg.mod.A": ["pkg.placement.A", "pkg.other.A"]}
    assert _rewrite_candidates_check(["pkg.mod.A"], text, cands) is None


def test_rewrite_candidates_check_path_without_candidates_ignored():
    # A path with no candidates in the dict → skip it.
    text = '@patch("pkg.mod.B")\ndef test_f(m): pass\n'
    cands = {"pkg.mod.A": ["pkg.placement.A"]}  # A has candidates, B does not
    assert _rewrite_candidates_check(["pkg.mod.B"], text, cands) is None


# ---------------------------------------------------------------------------
# _build_rewrite_verify_prompt — candidates_per_path
# ---------------------------------------------------------------------------


def test_build_rewrite_verify_prompt_with_candidates():
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    prompt = _build_rewrite_verify_prompt(
        context_msg,
        "def test_f(): pass\n",
        "def test_f(): pass\n",
        candidates_per_path={"pkg.big.A": ["pkg.sub_a.A", "pkg.sub_b.A"]},
    )
    assert "Call-graph candidate paths" in prompt
    assert "pkg.sub_a.A" in prompt


def test_build_rewrite_verify_prompt_candidates_above_threshold():
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    many_cands = [f"pkg.sub_{i}.A" for i in range(_CG_CANDIDATES_LLM_THRESHOLD + 1)]
    prompt = _build_rewrite_verify_prompt(
        context_msg,
        "def test_f(): pass\n",
        "def test_f(): pass\n",
        candidates_per_path={"pkg.big.A": many_cands},
    )
    assert "Call-graph candidate paths" not in prompt


def test_build_rewrite_verify_prompt_no_candidates():
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    prompt = _build_rewrite_verify_prompt(
        context_msg,
        "def test_f(): pass\n",
        "def test_f(): pass\n",
    )
    assert "Call-graph candidate paths" not in prompt
    assert "Verify that the rewrite is correct" in prompt


# ---------------------------------------------------------------------------
# _process_file_source — candidates pre-check
# ---------------------------------------------------------------------------


_PATCH_MAKE_CLIENT = "crispen.patch_rewriter.make_client"
_PATCH_GET_KEY_PR = "crispen.patch_rewriter.get_api_key"
_PATCH_CALL_PR = "crispen.patch_rewriter.call_with_tool"


def _make_process_cfg():
    return CrispenConfig(patch_update_retries=1, llm_verify_retries=0)


@mock_patch(_PATCH_CALL_PR)
@mock_patch(_PATCH_MAKE_CLIENT)
@mock_patch(_PATCH_GET_KEY_PR, return_value="key")
def test_process_file_source_candidates_reject_no_change(
    mock_key, mock_client, mock_call
):
    # LLM proposes no change but candidates exist → reject and retry.
    # First classify: no rename → rejected by candidates check.
    # Second classify: correct rename in candidates → verify → accepted.
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    ctx = _FLContext(
        filepath="/repo/pkg/big.py",
        old_module="pkg.big",
        original_source="from external import A\ndef f(): A()\n",
        modified_source="from .sub_a import f\n",
        new_files={"sub_a.py": "from external import A\ndef f(): A()\n"},
        new_module_paths={"sub_a.py": "pkg.sub_a"},
        entity_to_target={"f": "sub_a.py"},
        forking_old_paths={"pkg.big.A"},
    )
    context_msg = _build_context_message([ctx])
    mock_call.side_effect = [
        # First classify: no rename (LLM says no change needed)
        LLMCallResult(
            tool_input={"needs_rewrite": False, "patch_renames": {}},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Second classify (after candidates rejection): correct rename
        LLMCallResult(
            tool_input={
                "needs_rewrite": False,
                "patch_renames": {"pkg.big.A": "pkg.sub_a.A"},
            },
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Verify rename
        LLMCallResult(
            tool_input={"correct": True, "corrections": {}, "issue": ""},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
    ]
    cg_candidates = {"test_f": {"pkg.big.A": ["pkg.sub_a.A"]}}
    new_src, changed, _ = _process_file_source(
        src,
        {"pkg.big.A"},
        context_msg,
        mock_client.return_value,
        _make_process_cfg(),
        max_attempts=2,
        cg_candidates=cg_candidates,
    )
    assert changed
    assert "pkg.sub_a.A" in new_src
    # Two classify calls + one verify call = 3
    assert mock_call.call_count == 3


@mock_patch(_PATCH_CALL_PR)
@mock_patch(_PATCH_MAKE_CLIENT)
@mock_patch(_PATCH_GET_KEY_PR, return_value="key")
def test_process_file_source_candidates_reject_verbose(
    mock_key, mock_client, mock_call, capsys
):
    # verbose=True prints 'candidates check rejected' when cand_issue fires.
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    ctx = _FLContext(
        filepath="/repo/pkg/big.py",
        old_module="pkg.big",
        original_source="from external import A\ndef f(): A()\n",
        modified_source="from .sub_a import f\n",
        new_files={"sub_a.py": "from external import A\ndef f(): A()\n"},
        new_module_paths={"sub_a.py": "pkg.sub_a"},
        entity_to_target={"f": "sub_a.py"},
        forking_old_paths={"pkg.big.A"},
    )
    context_msg = _build_context_message([ctx])
    mock_call.side_effect = [
        # First classify: no rename → rejected by candidates check.
        LLMCallResult(
            tool_input={"needs_rewrite": False, "patch_renames": {}},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Second classify: correct rename
        LLMCallResult(
            tool_input={
                "needs_rewrite": False,
                "patch_renames": {"pkg.big.A": "pkg.sub_a.A"},
            },
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Verify rename
        LLMCallResult(
            tool_input={"correct": True, "corrections": {}, "issue": ""},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
    ]
    cg_candidates = {"test_f": {"pkg.big.A": ["pkg.sub_a.A"]}}
    _process_file_source(
        src,
        {"pkg.big.A"},
        context_msg,
        mock_client.return_value,
        _make_process_cfg(),
        max_attempts=2,
        cg_candidates=cg_candidates,
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "candidates check rejected" in err


@mock_patch(_PATCH_CALL_PR)
@mock_patch(_PATCH_MAKE_CLIENT)
@mock_patch(_PATCH_GET_KEY_PR, return_value="key")
def test_process_file_source_candidates_reject_bad_rename(
    mock_key, mock_client, mock_call
):
    # LLM proposes a rename not in candidates → rejected → retry with correct answer.
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    ctx = _FLContext(
        filepath="/repo/pkg/big.py",
        old_module="pkg.big",
        original_source="from external import A\ndef f(): A()\n",
        modified_source="from .sub_a import f\n",
        new_files={
            "sub_a.py": "from external import A\ndef f(): A()\n",
            "sub_b.py": "from external import A\ndef g(): A()\n",
        },
        new_module_paths={"sub_a.py": "pkg.sub_a", "sub_b.py": "pkg.sub_b"},
        entity_to_target={"f": "sub_a.py"},
        forking_old_paths={"pkg.big.A"},
    )
    context_msg = _build_context_message([ctx])
    mock_call.side_effect = [
        # First classify: wrong rename (not in candidates)
        LLMCallResult(
            tool_input={
                "needs_rewrite": False,
                "patch_renames": {"pkg.big.A": "pkg.sub_b.A"},
            },
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Second classify: correct rename
        LLMCallResult(
            tool_input={
                "needs_rewrite": False,
                "patch_renames": {"pkg.big.A": "pkg.sub_a.A"},
            },
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Verify
        LLMCallResult(
            tool_input={"correct": True, "corrections": {}, "issue": ""},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
    ]
    cg_candidates = {"test_f": {"pkg.big.A": ["pkg.sub_a.A"]}}
    new_src, changed, _ = _process_file_source(
        src,
        {"pkg.big.A"},
        context_msg,
        mock_client.return_value,
        _make_process_cfg(),
        max_attempts=2,
        cg_candidates=cg_candidates,
    )
    assert changed
    assert "pkg.sub_a.A" in new_src
    assert mock_call.call_count == 3


@mock_patch(_PATCH_CALL_PR)
@mock_patch(_PATCH_MAKE_CLIENT)
@mock_patch(_PATCH_GET_KEY_PR, return_value="key")
def test_process_file_source_rewrite_candidates_reject_and_retry(
    mock_key, mock_client, mock_call
):
    # Rewrite returns old path still present → rewrite candidates check rejects
    # without calling verify → retry; second rewrite uses valid candidate → accepted.
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    complex_logic()\n'
    ctx = _FLContext(
        filepath="/repo/pkg/big.py",
        old_module="pkg.big",
        original_source="from external import A\ndef f(): A()\n",
        modified_source="from .sub_a import f\n",
        new_files={"sub_a.py": "from external import A\ndef f(): A()\n"},
        new_module_paths={"sub_a.py": "pkg.sub_a"},
        entity_to_target={"f": "sub_a.py"},
        forking_old_paths={"pkg.big.A"},
    )
    context_msg = _build_context_message([ctx])
    bad_rewrite = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    complex_logic()\n'
    good_rewrite = '@patch("pkg.sub_a.A")\ndef test_f(mock_a):\n    complex_logic()\n'
    mock_call.side_effect = [
        # classify → needs rewrite
        LLMCallResult(
            tool_input={"needs_rewrite": True},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # rewrite 1: old path still present → rejected by _rewrite_candidates_check
        LLMCallResult(
            tool_input={"rewritten_function": bad_rewrite},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # rewrite 2: valid candidate
        LLMCallResult(
            tool_input={"rewritten_function": good_rewrite},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # verify
        LLMCallResult(
            tool_input={"correct": True, "issue": ""},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
    ]
    cg_candidates = {"test_f": {"pkg.big.A": ["pkg.sub_a.A"]}}
    new_src, changed, _ = _process_file_source(
        src,
        {"pkg.big.A"},
        context_msg,
        mock_client.return_value,
        _make_process_cfg(),
        max_attempts=2,
        cg_candidates=cg_candidates,
    )
    assert changed
    assert "pkg.sub_a.A" in new_src
    assert mock_call.call_count == 4  # classify + bad_rw + good_rw + verify


@mock_patch(_PATCH_CALL_PR)
@mock_patch(_PATCH_MAKE_CLIENT)
@mock_patch(_PATCH_GET_KEY_PR, return_value="key")
def test_process_file_source_candidates_all_retries_escalates_to_rewrite(
    mock_key, mock_client, mock_call, capsys
):
    # All classify retries exhausted with persistent candidates check rejections →
    # escalate to full rewrite rather than silently leaving the test broken.
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    ctx = _FLContext(
        filepath="/repo/pkg/big.py",
        old_module="pkg.big",
        original_source="from external import A\ndef f(): A()\n",
        modified_source="from .sub_a import f\n",
        new_files={"sub_a.py": "from external import A\ndef f(): A()\n"},
        new_module_paths={"sub_a.py": "pkg.sub_a"},
        entity_to_target={"f": "sub_a.py"},
        forking_old_paths={"pkg.big.A"},
    )
    context_msg = _build_context_message([ctx])
    good_rewrite = '@patch("pkg.sub_a.A")\ndef test_f(mock_a):\n    pass\n'
    mock_call.side_effect = [
        # First classify: no rename → rejected by candidates check (not last attempt).
        LLMCallResult(
            tool_input={"needs_rewrite": False, "patch_renames": {}},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Second classify: still no rename → last attempt → escalate to rewrite.
        LLMCallResult(
            tool_input={"needs_rewrite": False, "patch_renames": {}},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Rewrite (escalated from candidates check failure):
        LLMCallResult(
            tool_input={"rewritten_function": good_rewrite},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Rewrite verify:
        LLMCallResult(
            tool_input={"correct": True, "issue": ""},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
    ]
    # Two candidates → ambiguous → LLM keeps returning no_change.
    cg_candidates = {"test_f": {"pkg.big.A": ["pkg.sub_a.A", "pkg.sub_b.A"]}}
    new_src, changed, _ = _process_file_source(
        src,
        {"pkg.big.A"},
        context_msg,
        mock_client.return_value,
        _make_process_cfg(),
        max_attempts=2,
        cg_candidates=cg_candidates,
        verbose=True,
    )
    assert changed
    assert "pkg.sub_a.A" in new_src
    assert mock_call.call_count == 4  # classify x2 + rewrite + verify
    err = capsys.readouterr().err
    assert "candidates check retries exhausted" in err


# ---------------------------------------------------------------------------
# apply_patch_callgraph
# ---------------------------------------------------------------------------


def test_apply_patch_callgraph_empty_contexts():
    result = list(apply_patch_callgraph([], {}, "/repo"))
    assert result == []


def test_apply_patch_callgraph_no_forking_paths():
    ctx = _make_fl_ctx(forking_old_paths=set())
    result = list(apply_patch_callgraph([ctx], {}, "/repo"))
    assert result == []


def test_apply_patch_callgraph_per_file_update(tmp_path):
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    conflict_src = "from external import use_fn\ndef resolve(): use_fn()\n"
    ctx = _FLContext(
        filepath=str(tmp_path / "pkg" / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src, "conflict.py": conflict_src},
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    file_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn):\n"
        "    helper()\n"
    )
    test_file = tmp_path / "test_orig.py"
    test_file.write_text(file_src, encoding="utf-8")
    per_file = {str(test_file): {"source": file_src, "msgs": []}}
    list(apply_patch_callgraph([ctx], per_file, str(tmp_path)))
    assert '@patch("pkg.placement.use_fn")' in per_file[str(test_file)]["source"]


def test_apply_patch_callgraph_repo_scan(tmp_path):
    test_file = tmp_path / "test_something.py"
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    test_file.write_text(
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn):\n"
        "    helper()\n",
        encoding="utf-8",
    )
    ctx = _FLContext(
        filepath=str(tmp_path / "pkg" / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    msgs = list(apply_patch_callgraph([ctx], {}, str(tmp_path)))
    updated = test_file.read_text(encoding="utf-8")
    assert '@patch("pkg.placement.use_fn")' in updated
    assert any("call-graph" in m for m in msgs)


def test_apply_patch_callgraph_repo_scan_no_change(tmp_path):
    test_file = tmp_path / "test_something.py"
    test_file.write_text("def test_f(): pass\n", encoding="utf-8")
    ctx = _FLContext(
        filepath=str(tmp_path / "pkg" / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": "from external import use_fn\ndef f(): use_fn()\n"},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    msgs = list(apply_patch_callgraph([ctx], {}, str(tmp_path)))
    assert msgs == []


def test_apply_patch_callgraph_repo_root_none():
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={},
        new_module_paths={},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    result = list(apply_patch_callgraph([ctx], {}, None))
    assert result == []


def test_apply_patch_callgraph_per_file_no_change(tmp_path):
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath=str(tmp_path / "pkg" / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    file_src = "# pkg.orig.use_fn mentioned here but no test functions.\nx = 1\n"
    key = str(tmp_path / "module.py")
    per_file = {key: {"source": file_src, "msgs": []}}
    list(apply_patch_callgraph([ctx], per_file, None))
    assert per_file[key]["source"] == file_src


def test_apply_patch_callgraph_per_file_no_match(tmp_path):
    """per_file entry whose source contains no forking path string → continue."""
    ctx = _FLContext(
        filepath=str(tmp_path / "pkg" / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n"
        },
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    file_src = "x = 1\n"  # no mention of forking path
    key = str(tmp_path / "module.py")
    per_file = {key: {"source": file_src, "msgs": []}}
    list(apply_patch_callgraph([ctx], per_file, None))
    assert per_file[key]["source"] == file_src


def test_apply_patch_callgraph_repo_scan_oserror(tmp_path):
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath=str(tmp_path / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    bad_file = tmp_path / "test_bad.py"
    bad_file.write_text(
        '@patch("pkg.orig.use_fn")\ndef test_f(): helper()\n', encoding="utf-8"
    )
    bad_file.chmod(0o000)
    try:
        msgs = list(apply_patch_callgraph([ctx], {}, str(tmp_path)))
        assert msgs == []
    finally:
        bad_file.chmod(0o644)


def test_apply_patch_callgraph_repo_scan_file_no_change(tmp_path):
    test_file = tmp_path / "helper.py"
    test_file.write_text(
        "# references pkg.orig.use_fn in a comment\nx = 1\n",
        encoding="utf-8",
    )
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath=str(tmp_path / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    msgs = list(apply_patch_callgraph([ctx], {}, str(tmp_path)))
    assert msgs == []
    assert "x = 1" in test_file.read_text(encoding="utf-8")


def test_apply_patch_callgraph_excluded_dirs(tmp_path):
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    excluded_file = venv_dir / "test_something.py"
    excluded_file.write_text(
        '@patch("pkg.orig.use_fn")\ndef test_f(): helper()\n', encoding="utf-8"
    )
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath=str(tmp_path / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    msgs = list(apply_patch_callgraph([ctx], {}, str(tmp_path)))
    assert '@patch("pkg.orig.use_fn")' in excluded_file.read_text(encoding="utf-8")
    assert msgs == []


# ---------------------------------------------------------------------------
# _get_const_votes_from_rewrite
# ---------------------------------------------------------------------------


def test_get_const_votes_empty_refs():
    """No const_refs → empty dict, no parsing needed."""
    assert _get_const_votes_from_rewrite("def test_f(): pass\n", []) == {}


def test_get_const_votes_syntax_error():
    """Unparseable func_text → empty dict (SyntaxError branch)."""
    refs = [_make_ref("TARGET", "pkg.old.X")]
    assert _get_const_votes_from_rewrite("def f(:\n", refs) == {}


def test_get_const_votes_no_function_in_body():
    """Valid Python but no FunctionDef/AsyncFunctionDef → empty dict."""
    refs = [_make_ref("TARGET", "pkg.old.X")]
    result = _get_const_votes_from_rewrite("x = 1\n", refs)
    assert result == {}


def test_get_const_votes_non_call_decorator_skipped():
    """A bare-name decorator (not a Call node) is skipped without error."""
    code = "@pytest.mark.slow\n@patch(TARGET)\ndef test_f(m): pass\n"
    refs = [_make_ref("TARGET", "pkg.old.X")]
    # TARGET still present as Name → no vote entry (const unchanged).
    result = _get_const_votes_from_rewrite(code, refs)
    assert result == {}


def test_get_const_votes_non_patch_call_skipped():
    """A Call decorator whose func is not 'patch' is skipped."""
    code = "@other_decorator('pkg.old.X')\ndef test_f(m): pass\n"
    refs = [_make_ref("TARGET", "pkg.old.X")]
    result = _get_const_votes_from_rewrite(code, refs)
    assert result == {}


def test_get_const_votes_no_args_decorator_skipped():
    """@patch() with no args → skipped (no args branch)."""
    code = "@patch()\ndef test_f(): pass\n"
    refs = [_make_ref("TARGET", "pkg.old.X")]
    result = _get_const_votes_from_rewrite(code, refs)
    assert result == {}


def test_get_const_votes_module_attr_const_name():
    """@patch(module.CONST) style (Attribute node) → const name recorded correctly."""
    # Attribute form used when const is module-aliased after _restore_const_refs.
    code = "@patch(module.TARGET)\ndef test_f(m): pass\n"
    refs = [_make_ref("module.TARGET", "pkg.old.X")]
    result = _get_const_votes_from_rewrite(code, refs)
    # const still present as module.TARGET → no vote entry.
    assert result == {}


def test_get_const_votes_successful_vote():
    """LLM updated the path → new literal collected, vote returned."""
    refs = [_make_ref("TARGET", "pkg.mod.X")]
    code = '@patch("pkg.mod.sub.X")\ndef test_f(m): pass\n'
    result = _get_const_votes_from_rewrite(code, refs)
    assert result == {"pkg.mod.X": "pkg.mod.sub.X"}


def test_get_const_votes_deeply_nested_attr_skipped():
    """@patch(module.sub.CONST) where arg0 is Attribute(Attribute) — falls through
    all elif branches (663->647 coverage: the third elif is False for this form)."""
    # module.sub.CONST: arg0.value is Attribute, not Name → elif at 661 is False;
    # arg0 is not Constant → elif at 663 is False → no match, loop continues.
    code = "@patch(module.sub.CONST)\ndef test_f(m): pass\n"
    refs = [_make_ref("TARGET", "pkg.mod.X")]
    result = _get_const_votes_from_rewrite(code, refs)
    # No string literal collected, TARGET still absent → no vote.
    assert result == {}


# ---------------------------------------------------------------------------
# _update_file_patch_strings — non-participant keep-old vote (lines 3170-3172)
# ---------------------------------------------------------------------------


@mock_patch(_PATCH_CALL_TOOL)
def test_rewrite_non_participant_casts_keep_old_vote(mock_call, tmp_path):
    """A function that fails the rewrite (edit_failure) still casts a keep-old
    vote, preventing a const from being updated when only one of two users
    successfully renamed it.

    Scenario:
      test_a: classify → rename X → after.X; verify OK → string_swap_results.
      test_b: classify → needs_rewrite → rewrite → LLM returns None (failure)
              → edit_failure → NOT in string_swap_results.

    Without the keep-old fix: X proposals = {"after.X"} (single) → const updated.
    With the keep-old fix:    X proposals = {"after.X", "old.X"} → conflicting →
                              test_a inlined, const definition unchanged.
    """
    src = (
        'TARGET = "crispen.before.X"\n'
        "\n"
        "@patch(TARGET)\n"
        "def test_a(mock_x):\n"
        "    pass\n"
        "\n"
        "@patch(TARGET)\n"
        "def test_b(mock_x):\n"
        "    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # test_a: classify → rename → verify OK.
    # test_b: classify → needs_rewrite → rewrite attempt → None response (failure).
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after.X"},
            }
        ),
        _ok(_VERIFY_OK),
        _ok({"needs_rewrite": True}),
        LLMCallResult(tool_input=None, elapsed=0.0, input_tokens=0, output_tokens=0),
    ]
    result, changed, cross = _process_file_source(
        src,
        {"crispen.before.X"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
    )
    # test_b failed → keep-old vote → conflicting → const NOT updated.
    assert 'TARGET = "crispen.before.X"' in result
    # test_a's decorator inlined individually.
    assert '@patch("crispen.after.X")' in result


@mock_patch(_PATCH_CALL_TOOL)
def test_rewrite_non_participant_cross_file_ref_skipped(mock_call, tmp_path):
    """Non-participant with a cross-file const ref: the ref.source_file != scan_file_abs
    branch is False, so no same-file keep-old vote is cast (3171->3170 branch).

    test_a: succeeds (in string_swap_results).
    test_b: fails (not in string_swap_results).  test_b's const is defined in
            helpers.py (cross-file) so the keep-old loop skips it — no
            same_file_proposals entry for that ref.
    """
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "crispen.before.X"\n', encoding="utf-8")
    src = (
        "from .helpers import TARGET\n"
        "\n"
        "@patch(TARGET)\n"
        "def test_a(mock_x):\n"
        "    pass\n"
        "\n"
        "@patch(TARGET)\n"
        "def test_b(mock_x):\n"
        "    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after.X"},
            }
        ),
        _ok(_VERIFY_OK),
        _ok({"needs_rewrite": True}),
        LLMCallResult(tool_input=None, elapsed=0.0, input_tokens=0, output_tokens=0),
    ]
    result, changed, cross = _process_file_source(
        src,
        {"crispen.before.X"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
        repo_root=str(tmp_path),
    )
    # Cross-file ref → no same-file conflict → cross updated (test_a's rename wins).
    helpers_abs = str(helpers.resolve())
    assert helpers_abs in cross
    assert cross[helpers_abs] == {"crispen.before.X": "crispen.after.X"}


# ---------------------------------------------------------------------------
# _callgraph_update_file — BFS-ambiguous const-ref casts keep-old vote (line 3420)
# ---------------------------------------------------------------------------


def test_callgraph_const_ref_no_scan_file_skips_keep_old(tmp_path):
    """scan_file=None → scan_file_abs="" (falsy) → the keep-old block is skipped
    entirely (3420->3426 branch). BFS ambiguous functions don't cast any vote.
    The string literal path is still updated normally.
    """
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
            "conflict.py": "from external import use_fn\ndef resolve(): use_fn()\n",
        },
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
        "    resolve()\n"
    )
    # scan_file=None → scan_file_abs="" → keep-old block skipped; string literal
    # unchanged because BFS is ambiguous (no resolved result).
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx],
        scan_file=None,
        index=None,
    )
    assert not changed


def test_callgraph_const_ref_ambiguous_casts_keep_old_vote(tmp_path):
    """When BFS finds multiple candidates for a const-backed path (ambiguous),
    the function casts a keep-old vote so a shared constant isn't updated to a
    value that is wrong for the ambiguous function.

    test_a: calls helper() → placement (single BFS candidate) → vote "placement".
    test_b: calls helper() + resolve() → placement AND conflict (2 BFS candidates
            for use_fn) → ambiguous → keep-old vote.
    Proposals for _PATCH_USE: {"pkg.placement.use_fn", "pkg.orig.use_fn"} → conflict
    → constant NOT updated; test_a gets its decorator inlined individually.
    """
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
            "conflict.py": "from external import use_fn\ndef resolve(): use_fn()\n",
        },
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_a(m):\n"
        "    helper()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_b(m):\n"
        "    helper()\n"
        "    resolve()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src, {"pkg.orig.use_fn"}, [ctx], scan_file=scan, index=index
    )
    # test_b is ambiguous → keep-old vote → conflict with test_a's rename vote.
    # Constant definition must NOT be updated.
    assert '_PATCH_USE = "pkg.orig.use_fn"' in result
    # test_a's decorator IS inlined (it had a resolved rename).
    assert '@patch("pkg.placement.use_fn")' in result
