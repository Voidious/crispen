"""Tests for patch_rewriter — 100% branch coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch as mock_patch

import libcst as cst

from crispen.config import CrispenConfig
from crispen.llm_client import LLMCallResult
from crispen.patch_rewriter import (
    _FLContext,
    RewriteAccumulator,
    _CgIndex,
    _CG_MAX_DEPTH,
    _CG_MAX_MODULES,
    _apply_cross_file_const_updates,
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
    _cg_build_index,
    _cg_collect_called_names,
    _cg_collect_defined_names,
    _cg_collect_func_body_calls,
    _cg_file_to_module_and_package,
    _cg_parse_imports,
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
    _resolve_forking_path_via_callgraph,
    _resolve_import_to_file,
    _splice_function,
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
    _, _, orig_users, _ = _build_rename_guard_sets([ctx])
    assert orig_users.get("make_client") == ["advise"]


def test_build_rename_guard_sets_no_users_not_in_map():
    # make_client is still imported but not referenced by any top-level def.
    ctx = _make_fl_ctx(
        original_source="from ...llm_client import make_client\ndef advise(): pass\n",
        modified_source="from ...llm_client import make_client\ndef advise(): pass\n",
        new_files={},
    )
    _, _, orig_users, _ = _build_rename_guard_sets([ctx])
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
    _, still_in, orig_users, _ = _build_rename_guard_sets([ctx1, ctx2])
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
    _, _, orig_users, _ = _build_rename_guard_sets([ctx1, ctx2])
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
    assert "via a helper" in ctx_msg


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
    assert "Previous rewrite was invalid" in prompt
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
    # Classify returns tool_input=None → break, no update.
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False


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
def test_process_no_change_verify_rejects_then_accepts(mock_call):
    # No-change verify rejects first; classify+verify accepted on retry.
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT),
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
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
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),  # classify → no change
        _ok(_VERIFY_REJECT),  # verify → reject
        _ok(_CLASSIFY_NO_CHANGE),  # classify (retry) → no change again
        _ok(_VERIFY_REJECT),  # verify → reject (retries exhausted → escalate)
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
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT),
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT),
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
    # the context message.  The still-imported guard must drop these corrections;
    # with empty corrections the retry loop resumes and accepts no-change on verify.
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
    # Correction was filtered (X still imported) — no change applied.
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
    # _acc accumulates calls from both classify and no-change verify.
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
        _ok(_VERIFY_REJECT),
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
        _ok(_VERIFY_REJECT),  # verify → reject
        _ok(_CLASSIFY_RENAME),  # classify (retry) → rename again
        _ok(_VERIFY_REJECT),  # verify → reject (retries exhausted → escalate)
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
        _ok({"correct": False, "issue": "wrong module path"}),
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
# Same-file constant conflict: passthrough + inline substitution
# ---------------------------------------------------------------------------


@mock_patch(_PATCH_CALL_TOOL)
def test_process_passthrough_conflict_inline(mock_call, tmp_path):
    """One test passes through a constant (no rename), another renames it.
    Expected: conflict detected → same-file const stays unchanged; the renaming
    test gets its decorator inlined with the correct literal value.  The
    non-conflicting constant is updated normally via same_file_const_map.

    Covers:
      - line 1736  (same_file_passthrough.add)
      - lines 1762-1774  (conflicting_old_vals loop, continue branch when
                           inline_subs is empty for the passthrough function)
      - lines 1775-1792  (base_text, existing_idx=None → append splice)
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
    # test_a renames Y but NOT X  → X is a passthrough from test_a's perspective.
    # test_b renames X            → conflict (passthrough + rename) for X.
    # Y has a single uncontested rename → const map update.
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
    # Non-conflicting constant updated in place.
    assert 'TARGET2 = "crispen.after.Y"' in result
    # Conflicting constant NOT updated (conflict: passthrough + rename).
    assert 'TARGET = "crispen.before.X"' in result
    # test_b decorator inlined with the correct literal value.
    assert '@patch("crispen.after.X")' in result


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
    result, changed = _callgraph_update_file(
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
    result, changed = _callgraph_update_file(
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
    result, changed = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.placement.use_fn")' in result


def test_callgraph_update_file_no_resolution(tmp_path):
    # Test calls 'unrelated' — not imported → BFS queue empty → no resolution.
    test_src = (
        '@patch("pkg.orig.use_fn")\n' "def test_f(mock_use_fn):\n" "    unrelated()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert not changed


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
    result, changed = _callgraph_update_file(
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
    result, changed = _callgraph_update_file(
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
    result, changed = _callgraph_update_file(
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
    result, changed = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn", "pkg.other.other_fn"},
        [ctx_a, ctx_b],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.placement.use_fn")' in result


def test_callgraph_update_file_const_ref_no_resolution_passthrough(tmp_path):
    test_src = (
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_f(mock_use_fn):\n"
        "    unrelated()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert not changed
    assert '_PATCH_USE = "pkg.orig.use_fn"' in result


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
    result, changed = _callgraph_update_file(
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
    result, changed = _callgraph_update_file(
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
    result, changed = _callgraph_update_file(
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
    result, changed = _callgraph_update_file(
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
    result, changed = _callgraph_update_file(
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
