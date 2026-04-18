from __future__ import annotations
from crispen.patch_rewriter import (
    _FLContext,
    _build_context_message,
    _build_rename_guard_sets,
    _compiles,
    _extract_patch_lookup,
    _extract_still_imported_names,
    _find_with_patch_paths_in_body,
    _get_external_import_names,
    _is_bad_rename,
    _is_patch_call,
    _matches_any,
    _patch_strings_in_text,
)
import libcst as cst
from .test_process_file_source_basic_flow import _make_fl_ctx


def test_is_patch_call_name_match():
    call_node = cst.parse_expression('patch("foo")')
    assert _is_patch_call(call_node) is True


def test_is_patch_call_attribute_match():
    call_node = cst.parse_expression('mock.patch("foo")')
    assert _is_patch_call(call_node) is True


def test_is_patch_call_other_name():
    call_node = cst.parse_expression('other("foo")')
    assert _is_patch_call(call_node) is False


def test_matches_any_exact():
    assert _matches_any("a.b.C", {"a.b.C"}) is True


def test_matches_any_prefix():
    assert _matches_any("a.b.C.method", {"a.b.C"}) is True


def test_matches_any_near_miss():
    # "a.b.CExtra" should NOT match "a.b.C"
    assert _matches_any("a.b.CExtra", {"a.b.C"}) is False


def test_matches_any_no_match():
    assert _matches_any("x.y.Z", {"a.b.C"}) is False


def test_compiles_valid():
    assert _compiles("x = 1\n") is True


def test_compiles_invalid():
    assert _compiles("def f(:\n    pass\n") is False


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


def test_patch_strings_in_text_context_manager():
    text = 'def test_f():\n    with patch("pkg.mod.C") as m: pass\n'
    assert _patch_strings_in_text(text) == {"pkg.mod.C"}
