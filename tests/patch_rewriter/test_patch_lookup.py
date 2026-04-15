from __future__ import annotations
from crispen.patch_rewriter import (
    _build_context_message,
    _extract_patch_lookup,
    _extract_still_imported_names,
    _matches_any,
)
from .patch_lookup import _make_ctx_with_ext_imports
from .context_builders import _make_fl_ctx


def test_matches_any_exact():
    assert _matches_any("a.b.C", {"a.b.C"}) is True


def test_matches_any_prefix():
    assert _matches_any("a.b.C.method", {"a.b.C"}) is True


def test_matches_any_near_miss():
    # "a.b.CExtra" should NOT match "a.b.C"
    assert _matches_any("a.b.CExtra", {"a.b.C"}) is False


def test_matches_any_no_match():
    assert _matches_any("x.y.Z", {"a.b.C"}) is False


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
