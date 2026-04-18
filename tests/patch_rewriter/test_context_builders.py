from __future__ import annotations
from crispen.patch_rewriter import (
    _CG_CANDIDATES_LLM_THRESHOLD,
    _FLContext,
    _build_classify_prompt,
    _build_context_message,
    _build_func_verify_prompt,
    _build_no_change_verify_prompt,
    _build_rewrite_func_prompt,
    _build_rewrite_verify_prompt,
    _extract_migration_reminder,
    _extract_patch_lookup,
    _extract_still_imported_names,
    _get_external_import_names,
    _import_header,
    _name_reference_map,
    _splice_function,
)
from .helpers import _ctx_msg, _make_fl_ctx, _make_fl_ctx_simple


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
