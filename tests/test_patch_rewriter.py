"""Tests for patch_rewriter — 100% branch coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch as mock_patch

import libcst as cst

from crispen.config import CrispenConfig
from crispen.llm_client import LLMCallResult
from crispen.patch_rewriter import (
    _FLContext,
    RewriteAccumulator,
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
    _compiles,
    _extract_migration_reminder,
    _find_test_functions_to_update,
    _find_with_patch_paths_in_body,
    _import_header,
    _is_patch_call,
    _name_reference_map,
    _matches_any,
    _process_file_source,
    _resolve_import_to_file,
    _splice_function,
    _substitute_consts_in_func_text,
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


def test_find_non_matching_decorator_included_when_triggered(tmp_path):
    """When one decorator matches, ALL resolvable decorators are included.

    A test that patches get_api_key, make_client, and call_with_tool for
    the same migrated function needs all three evaluated — even if only
    call_with_tool's path is in old_paths.
    """
    src = (
        'KEY = "crispen.mod.get_api_key"\n'
        'CALL = "crispen.mod.call_with_tool"\n\n'
        "@patch(KEY)\n"
        "@patch(CALL)\n"
        "def test_f(mock_call, mock_key):\n    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # Only CALL's value is in old_paths; KEY's value is not.
    result = _find_test_functions_to_update(
        src, {"crispen.mod.call_with_tool"}, scan_file=scan
    )
    assert len(result) == 1
    paths = result[0].old_patch_paths
    # Both paths must be present so the LLM can evaluate all decorators.
    assert "crispen.mod.get_api_key" in paths
    assert "crispen.mod.call_with_tool" in paths
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
    assert '"pkg.sub_a.A"' in helpers.read_text(encoding="utf-8")
