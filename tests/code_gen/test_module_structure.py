from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    _extract_module_docstring,
    _find_main_block_entity,
    _find_main_direct_callees,
    _strip_module_docstring,
    _strip_orphaned_section_headers,
    _strip_top_level_import_lines,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .test_shared_helpers import _classified, _make_entity, _plan
import textwrap


def test_strip_top_level_import_lines_removes_imports():
    src = "import os\nfrom typing import List\n\n_CONST = 1\n"
    result = _strip_top_level_import_lines(src)
    assert "import os" not in result
    assert "from typing import List" not in result
    assert "_CONST = 1" in result


def test_strip_top_level_import_lines_no_imports():
    src = "_CONST = 1\n"
    assert _strip_top_level_import_lines(src) == src


def test_strip_top_level_import_lines_syntax_error():
    src = "def (\n"
    assert _strip_top_level_import_lines(src) == src


def test_extract_module_docstring_present():
    src = '"""My module."""\n\nimport os\n'
    assert _extract_module_docstring(src) == '"""My module."""'


def test_extract_module_docstring_absent():
    src = "import os\n\ndef foo():\n    pass\n"
    assert _extract_module_docstring(src) is None


def test_extract_module_docstring_syntax_error():
    assert _extract_module_docstring("def (\n") is None


def test_extract_module_docstring_non_string_expr():
    # First statement is an expression but not a string constant.
    src = "1 + 1\n\ndef foo():\n    pass\n"
    assert _extract_module_docstring(src) is None


def test_strip_module_docstring_removes_docstring():
    src = '"""My module."""\n\n_CONST = 1\n'
    result = _strip_module_docstring(src)
    assert '"""My module."""' not in result
    assert "_CONST = 1" in result


def test_strip_module_docstring_no_docstring():
    src = "_CONST = 1\n"
    assert _strip_module_docstring(src) == src


def test_strip_module_docstring_syntax_error():
    src = "def (\n"
    assert _strip_module_docstring(src) == src


def test_generate_subdir_module_docstring_goes_to_init():
    # In subdir-split mode the module docstring belongs in __init__.py, not
    # in the split-off child module.  Migrate the preamble entity (_block_1)
    # along with foo so the docstring is removed from the original source,
    # triggering the restore-to-__init__ logic.
    source = textwrap.dedent(
        """\
        \"\"\"Top-level module doc.\"\"\"

        import os

        def foo():
            return os.sep

        def bar():
            return foo()
        """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["os"])
    e_foo = _make_entity("foo", 5, 6)
    e_bar = _make_entity("bar", 8, 9)
    c = _classified(entities=[e_block, e_foo, e_bar])
    plan = _plan(
        [GroupPlacement(group=["_block_1", "foo"], target_file="pkg/helpers.py")]
    )

    result = generate_file_splits(c, plan, source, "pkg.py", subdir_name="pkg")

    assert not result.abort
    init_src = result.original_source
    helpers_src = result.new_files["pkg/helpers.py"]
    # Docstring belongs in __init__.py.
    assert '"""Top-level module doc."""' in init_src
    # Docstring must NOT appear in the child module.
    assert '"""Top-level module doc."""' not in helpers_src


def test_generate_subdir_docstring_already_in_init_not_duplicated():
    # If the TOP_LEVEL entity stays in the original (not migrated), the
    # docstring remains in the updated source and must not be prepended again.
    source = textwrap.dedent(
        """\
        \"\"\"Top-level module doc.\"\"\"

        _CONST = 1

        def stayed():
            return _CONST

        def migrated():
            pass
        """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["_CONST"])
    e_stayed = _make_entity("stayed", 5, 6)
    e_migrated = _make_entity("migrated", 8, 9)
    c = _classified(entities=[e_block, e_stayed, e_migrated])
    plan = _plan([GroupPlacement(group=["migrated"], target_file="pkg/helpers.py")])

    result = generate_file_splits(c, plan, source, "pkg.py", subdir_name="pkg")

    assert not result.abort
    init_src = result.original_source
    assert init_src.count('"""Top-level module doc."""') == 1


def test_generate_subdir_module_docstring_goes_to_test_init():
    # For test-file subdir splits the module docstring goes into
    # subdir/__init__.py, not into the re-export stub file.
    source = textwrap.dedent(
        """\
        \"\"\"Tests for the runner module.\"\"\"

        import os

        def test_foo():
            return os.sep

        def test_bar():
            return test_foo()
        """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["os"])
    e_foo = _make_entity("test_foo", 5, 6)
    e_bar = _make_entity("test_bar", 8, 9)
    c = _classified(entities=[e_block, e_foo, e_bar])
    plan = _plan(
        [GroupPlacement(group=["_block_1", "test_foo"], target_file="svc/test_foo.py")]
    )

    result = generate_file_splits(
        c, plan, source, "tests/test_svc.py", subdir_name="svc"
    )

    assert not result.abort
    init_src = result.new_files["svc/__init__.py"]
    child_src = result.new_files["svc/test_foo.py"]
    updated_src = result.original_source
    # Docstring belongs in __init__.py.
    assert '"""Tests for the runner module."""' in init_src
    # Docstring must NOT appear in the child test file or the stub file.
    assert '"""Tests for the runner module."""' not in child_src
    assert '"""Tests for the runner module."""' not in updated_src


def test_generate_subdir_docstring_not_stripped_from_non_subdir_split():
    # Outside subdir-split mode, a TOP_LEVEL entity's docstring is preserved
    # in the new file (only imports are stripped, not docstrings).
    source = '"""Module doc."""\n\nimport os\n\ndef foo():\n    return os.sep\n'
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["os"])
    e_foo = _make_entity("foo", 5, 6)
    c = _classified(entities=[e_block, e_foo])
    plan = _plan([GroupPlacement(group=["_block_1", "foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert '"""Module doc."""' in new_src


def test_find_main_block_entity_present():
    from crispen.file_limiter.entity_parser import parse_entities

    source = textwrap.dedent(
        """\
        def run():
            pass

        if __name__ == "__main__":
            run()
        """
    )
    entities = parse_entities(source)
    esmap = {e.name: source.splitlines(keepends=True) for e in entities}
    # Rebuild entity_source_map properly
    lines = source.splitlines(keepends=True)
    esmap = {
        e.name: "".join(lines[e.start_line - 1 : e.end_line]).rstrip() for e in entities
    }
    result = _find_main_block_entity(entities, esmap)
    assert result is not None
    assert result.startswith("_block_")


def test_find_main_block_entity_absent():
    from crispen.file_limiter.entity_parser import parse_entities

    source = "def foo():\n    pass\n"
    entities = parse_entities(source)
    lines = source.splitlines(keepends=True)
    esmap = {
        e.name: "".join(lines[e.start_line - 1 : e.end_line]).rstrip() for e in entities
    }
    assert _find_main_block_entity(entities, esmap) is None


def test_find_main_block_entity_syntax_error_skipped():

    # Entity whose source is invalid Python: should be skipped gracefully.
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, [])
    result = _find_main_block_entity([entity], {"_block_1": "def (invalid"})
    assert result is None


def test_find_main_direct_callees_basic():
    src = 'if __name__ == "__main__":\n    run_tests()\n'
    callees = _find_main_direct_callees(src, {"run_tests", "other"})
    assert callees == {"run_tests"}


def test_find_main_direct_callees_not_in_entity_names():
    src = 'if __name__ == "__main__":\n    unknown()\n'
    callees = _find_main_direct_callees(src, {"run_tests"})
    assert callees == set()


def test_find_main_direct_callees_syntax_error():
    assert _find_main_direct_callees("def (invalid", {"foo"}) == set()


def test_find_main_direct_callees_no_main_block():
    src = "run_tests()\n"
    assert _find_main_direct_callees(src, {"run_tests"}) == set()


def test_generate_shebang_stripped_from_new_file():
    # Shebang on line 1 should NOT appear in generated new files.
    source = "#!/usr/bin/env python3\n\ndef foo():\n    pass\n\ndef bar():\n    foo()\n"
    e_foo = Entity(EntityKind.FUNCTION, "foo", 3, 4, ["foo"])
    e_bar = Entity(EntityKind.FUNCTION, "bar", 6, 7, ["bar"])
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["bar"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "#!/usr/bin/env python3" not in result.new_files["helpers.py"]


def test_generate_shebang_preserved_in_original_when_entity_migrated():
    # When the entity owning line 1 (with shebang comment) is migrated,
    # the shebang must be restored at the top of the original file.
    source = "#!/usr/bin/env python3\ndef foo():\n    pass\n\ndef bar():\n    pass\n"
    e_foo = Entity(EntityKind.FUNCTION, "foo", 1, 3, ["foo"])
    e_bar = Entity(EntityKind.FUNCTION, "bar", 5, 6, ["bar"])
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["foo"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert result.original_source.startswith("#!/usr/bin/env python3\n")
    assert "#!/usr/bin/env python3" not in result.new_files["helpers.py"]


def test_generate_shebang_preserved_when_not_migrated():
    # When the shebang entity stays in the original, shebang remains at top.
    source = "#!/usr/bin/env python3\ndef foo():\n    pass\n\ndef bar():\n    pass\n"
    e_foo = Entity(EntityKind.FUNCTION, "foo", 1, 3, ["foo"])
    e_bar = Entity(EntityKind.FUNCTION, "bar", 5, 6, ["bar"])
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["bar"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert result.original_source.startswith("#!/usr/bin/env python3\n")


def test_generate_main_block_stays_in_original():
    source = textwrap.dedent(
        """\
        def run():
            pass

        def other():
            pass

        if __name__ == "__main__":
            run()
        """
    )
    e_run = Entity(EntityKind.FUNCTION, "run", 1, 2, ["run"])
    e_other = Entity(EntityKind.FUNCTION, "other", 4, 5, ["other"])
    e_main = Entity(EntityKind.TOP_LEVEL, "_block_7", 7, 8, [])
    c = _classified(entities=[e_run, e_other, e_main])
    # Plan tries to migrate run + __main__ block and other.
    plan = _plan(
        [
            GroupPlacement(group=["run", "_block_7"], target_file="helpers.py"),
            GroupPlacement(group=["other"], target_file="helpers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    # __main__ block stays in original.
    assert 'if __name__ == "__main__"' in result.original_source
    assert 'if __name__ == "__main__"' not in result.new_files.get("helpers.py", "")


def test_generate_main_callee_stays_in_original():
    source = textwrap.dedent(
        """\
        def run():
            pass

        def other():
            pass

        if __name__ == "__main__":
            run()
        """
    )
    e_run = Entity(EntityKind.FUNCTION, "run", 1, 2, ["run"])
    e_other = Entity(EntityKind.FUNCTION, "other", 4, 5, ["other"])
    e_main = Entity(EntityKind.TOP_LEVEL, "_block_7", 7, 8, [])
    c = _classified(entities=[e_run, e_other, e_main])
    # Plan tries to migrate run (the direct callee of __main__).
    plan = _plan(
        [
            GroupPlacement(group=["run"], target_file="helpers.py"),
            GroupPlacement(group=["other"], target_file="helpers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    # run() is a direct __main__ callee — must stay in original.
    assert "def run():" in result.original_source
    # other() is not a callee — may be migrated.
    assert "helpers.py" in result.new_files


def test_strip_orphaned_3line_header_at_eof():
    """3-line block with no code after it is removed."""
    div = "# ---\n"
    source = "def foo():\n    pass\n\n\n" + div + "# Old Section\n" + div
    result = _strip_orphaned_section_headers(source)
    assert "# Old Section" not in result
    assert "def foo():" in result


def test_strip_orphaned_single_line_header_at_eof():
    """Single-line header with no code after it is removed."""
    source = "def foo():\n    pass\n\n# --- Removed ---\n"
    result = _strip_orphaned_section_headers(source)
    assert "# --- Removed ---" not in result
    assert "def foo():" in result


def test_strip_not_orphaned_3line_header():
    """3-line block followed by substantive code is kept."""
    div = "# ---\n"
    source = div + "# Helpers\n" + div + "\n\ndef helper():\n    pass\n"
    result = _strip_orphaned_section_headers(source)
    assert "# Helpers" in result
    assert "def helper():" in result


def test_strip_not_orphaned_single_line_header():
    """Single-line header followed by substantive code is kept."""
    source = "# --- Tools ---\n\ndef tool():\n    pass\n"
    result = _strip_orphaned_section_headers(source)
    assert "# --- Tools ---" in result


def test_strip_orphaned_header_followed_only_by_another_header():
    """Header followed only by another header (and then nothing) — both orphaned."""
    source = "def foo():\n" "    pass\n" "\n" "# --- First ---\n" "# --- Second ---\n"
    result = _strip_orphaned_section_headers(source)
    assert "# --- First ---" not in result
    assert "# --- Second ---" not in result
    assert "def foo():" in result


def test_strip_partial_orphan():
    """Only the header with no code after it is removed; the other stays."""
    source = (
        "# --- Active ---\n" "\n" "def foo():\n" "    pass\n" "\n" "# --- Empty ---\n"
    )
    result = _strip_orphaned_section_headers(source)
    assert "# --- Active ---" in result
    assert "# --- Empty ---" not in result


def test_strip_no_headers_returns_unchanged():
    """Source with no section headers is returned unchanged."""
    source = "def foo():\n    pass\n"
    assert _strip_orphaned_section_headers(source) == source


def test_strip_all_headers_have_content():
    """When every header has content below it, source is returned unchanged."""
    source = (
        "# --- A ---\n"
        "\n"
        "def a():\n"
        "    pass\n"
        "\n"
        "# --- B ---\n"
        "\n"
        "def b():\n"
        "    pass\n"
    )
    result = _strip_orphaned_section_headers(source)
    assert "# --- A ---" in result
    assert "# --- B ---" in result


def test_strip_equals_single_line_header_orphaned():
    """=== style orphaned header is also removed."""
    source = "def foo():\n    pass\n\n# === OLD SECTION ===\n"
    result = _strip_orphaned_section_headers(source)
    assert "# === OLD SECTION ===" not in result
