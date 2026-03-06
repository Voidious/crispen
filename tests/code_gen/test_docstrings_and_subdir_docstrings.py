from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import generate_file_splits
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .test_code_gen_utils import _classified, _make_entity, _plan


def test_generate_top_level_entity_imports_not_duplicated():
    # When a TOP_LEVEL entity source contains regular imports (e.g. `import os`)
    # those must NOT appear twice in the generated file: once from
    # _find_needed_imports and again from the entity source itself.
    source = "import os\n\n_CONST = os.sep\n\ndef foo():\n    return os.getcwd()\n"
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["os", "_CONST"])
    e_foo = _make_entity("foo", 5, 6)
    c = _classified(entities=[e_block, e_foo])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1", "foo"], target_file="utils.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert new_src.count("import os") == 1


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
