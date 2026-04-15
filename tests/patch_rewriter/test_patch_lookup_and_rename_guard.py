from __future__ import annotations
from crispen.patch_rewriter import (
    _FLContext,
    _build_context_message,
    _build_rename_guard_sets,
    _extract_patch_lookup,
    _extract_still_imported_names,
    _is_bad_rename,
    _is_patch_call,
    _matches_any,
)
import libcst as cst
from .helpers import _make_fl_ctx


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
