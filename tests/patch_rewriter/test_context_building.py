from __future__ import annotations
from crispen.patch_rewriter import (
    _FLContext,
    _build_context_message,
    _build_rename_guard_sets,
    _extract_migration_reminder,
    _extract_patch_lookup,
    _extract_still_imported_names,
    _get_external_import_names,
    _import_header,
    _name_reference_map,
    _splice_function,
)
from .test_patch_detection import _make_fl_ctx


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
