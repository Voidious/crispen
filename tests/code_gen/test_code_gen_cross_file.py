from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    _add_re_exports,
    _collect_external_imported_names,
    _find_cross_file_imports,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .test_code_gen_entities import _classified, _make_entity, _plan


def test_add_re_exports_top_level_import_derived_names_not_re_exported():
    # A TOP_LEVEL entity that includes import statements: the names introduced
    # by those imports must NOT appear in re-exports because they are preserved
    # in the original file by _remove_entity_lines, not moved to the new file.
    source = "import os\n\nMY_CONST\n"  # MY_CONST still loaded
    entity_src = "from typing import Dict\n\nMY_CONST = 42\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["Dict", "MY_CONST"])
    placement = GroupPlacement(group=["_block_1"], target_file="constants.py")
    result = _add_re_exports(
        source, [placement], {"_block_1": entity}, {"_block_1": entity_src}
    )
    assert "MY_CONST" in result  # assignment-defined name re-exported
    assert "Dict" not in result  # import-derived name suppressed


def test_generate_cross_file_import():
    # fn_a goes to fn_module.py; _block_1 (defining _CONST) goes to constants.py.
    # fn_a references _CONST → fn_module.py must have `from .constants import _CONST`.
    source = "_CONST = 42\n\ndef fn_a():\n    return _CONST\n"
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_fn = _make_entity("fn_a", 3, 4)
    c = _classified(entities=[e_block, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["fn_a"], target_file="fn_module.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    fn_src = result.new_files["fn_module.py"]
    assert "from .constants import _CONST" in fn_src
    # constants.py should NOT have a cross-import (it defines _CONST, not uses it)
    const_src = result.new_files["constants.py"]
    assert "from .fn_module" not in const_src


def test_generate_non_migrated_helper_extracted_to_new_file():
    # _run is non-migrated; test_fn is migrated and references _run.
    # _run is extracted into test_helpers.py to prevent an O→F→O cycle.
    source = textwrap.dedent(
        """\
        import textwrap

        def _run(x):
            return x

        def test_fn():
            return _run(1)
    """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["textwrap"])
    e_run = _make_entity("_run", 3, 4)
    e_test = _make_entity("test_fn", 6, 7)
    c = _classified(entities=[e_block, e_run, e_test])
    plan = _plan([GroupPlacement(group=["test_fn"], target_file="test_helpers.py")])

    result = generate_file_splits(c, plan, source, "original.py")

    new_src = result.new_files["test_helpers.py"]
    # _run is defined in the new file (extracted), not imported from original
    assert "def _run" in new_src
    assert "from .original import _run" not in new_src
    # import textwrap is not referenced by either entity
    assert "from .original import textwrap" not in new_src


def test_generate_self_referential_placement_dropped():
    # LLM names a target file the same as the original → would create a
    # circular import.  The placement must be silently dropped so the entity
    # stays in the original file and no self-import is added.
    source = "class Foo:\n    pass\n\nclass Bar:\n    pass\n"
    e_foo = _make_entity("Foo", 1, 2)
    e_bar = _make_entity("Bar", 4, 5)
    c = _classified(entities=[e_foo, e_bar])
    # "mymodule.py" is also the original filename → self-referential
    plan = _plan(
        [
            GroupPlacement(group=["Foo"], target_file="mymodule.py"),
            GroupPlacement(group=["Bar"], target_file="helpers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "mymodule.py")

    # Foo stays in the original — no circular self-import
    assert "from .mymodule import Foo" not in result.original_source
    assert "mymodule.py" not in result.new_files
    # Bar is still moved normally
    assert "helpers.py" in result.new_files
    assert "class Bar" in result.new_files["helpers.py"]
    # Foo remains in the original source (not removed)
    assert "class Foo" in result.original_source


def test_generate_all_placements_self_referential():
    # All placements target the original file → nothing is moved.
    source = "def foo():\n    pass\n"
    e_foo = _make_entity("foo", 1, 2)
    c = _classified(entities=[e_foo])
    plan = _plan([GroupPlacement(group=["foo"], target_file="original.py")])

    result = generate_file_splits(c, plan, source, "original.py")

    assert result.new_files == {}
    assert "from .original import foo" not in result.original_source
    assert "def foo" in result.original_source


def test_generate_aborts_on_cross_file_import_cycle():
    # fn_a references fn_b (in b.py) and fn_b references fn_a (in a.py).
    # This creates a circular import a.py ↔ b.py that Python cannot load.
    # generate_file_splits must detect the cycle and abort rather than emit
    # broken code.
    source = "def fn_a():\n    return fn_b()\n\ndef fn_b():\n    return fn_a()\n"
    e_a = _make_entity("fn_a", 1, 2)
    e_b = _make_entity("fn_b", 4, 5)
    c = _classified(entities=[e_a, e_b])
    plan = _plan(
        [
            GroupPlacement(group=["fn_a"], target_file="a.py"),
            GroupPlacement(group=["fn_b"], target_file="b.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    assert result.abort is True
    assert result.new_files == {}


def test_generate_aborts_on_cycle_through_original():
    # _CONST is a TOP_LEVEL constant (stays in original).
    # _worker is migrated to helpers.py and references _CONST.
    # main() (non-migrated) calls _worker → original will re-export _worker.
    # Cycle: original → helpers.py (re-export of _worker)
    #              → original (via `from .original import _CONST`).
    source = textwrap.dedent(
        """\
        _CONST = "value"

        def _worker():
            return _CONST

        def main():
            return _worker()
    """
    )
    e_const = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_worker = _make_entity("_worker", 3, 4)
    e_main = _make_entity("main", 6, 7)
    c = _classified(entities=[e_const, e_worker, e_main])
    plan = _plan([GroupPlacement(group=["_worker"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "original.py")

    # helpers.py would need `from .original import _CONST` while original
    # re-exports _worker from helpers.py → circular import → must abort.
    assert result.abort is True
    assert result.new_files == {}


def test_collect_external_imported_names_relative_path():
    # Non-absolute path → empty set (no scan).
    assert _collect_external_imported_names("relative/path.py") == set()


def test_collect_external_imported_names_nonexistent_file(tmp_path):
    # Absolute but non-existent → empty set.
    assert _collect_external_imported_names(str(tmp_path / "ghost.py")) == set()


def test_collect_external_imported_names_no_project_root(tmp_path):
    # File exists but no pyproject.toml/.git above it → empty set.
    f = tmp_path / "module.py"
    f.write_text("x = 1\n")
    # tmp_path is under /tmp which typically has no project markers.
    result = _collect_external_imported_names(str(f))
    # May or may not find a root depending on environment; we just verify no crash.
    assert isinstance(result, set)


def test_collect_external_imported_names_absolute_import(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    mod = pkg / "utils.py"
    mod.write_text("def _helper():\n    pass\n")
    caller = tmp_path / "tests" / "test_utils.py"
    caller.parent.mkdir()
    caller.write_text("from mypkg.utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_relative_import(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    mod = pkg / "utils.py"
    mod.write_text("def _helper():\n    pass\n")
    sibling = pkg / "other.py"
    sibling.write_text("from .utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_self_excluded(tmp_path):
    # The file being scanned is excluded from the search.
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("from module import _x\n")  # self-referential (ignored)
    result = _collect_external_imported_names(str(mod))
    assert "_x" not in result


def test_collect_external_imported_names_syntax_error_skipped(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("def _helper(): pass\n")
    bad = tmp_path / "bad.py"
    bad.write_text("def (invalid\n")
    good = tmp_path / "good.py"
    good.write_text("from module import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_non_matching_import_ignored(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("def _helper(): pass\n")
    other = tmp_path / "other.py"
    other.write_text("from different_module import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" not in result


def test_collect_external_imported_names_non_importfrom_nodes_skipped(tmp_path):
    # Caller file contains a plain `import` statement (not ImportFrom) mixed
    # with a matching `from … import`.  The plain import must be skipped without
    # crashing, and the matching ImportFrom still contributes to the result.
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("def _helper(): pass\n")
    caller = tmp_path / "caller.py"
    caller.write_text("import os\nfrom module import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_deep_relative_import(tmp_path):
    # Two-level relative import: `from ..utils import _helper`
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "utils.py"
    mod.write_text("def _helper(): pass\n")
    sub = tmp_path / "pkg" / "sub" / "caller.py"
    sub.parent.mkdir(parents=True)
    sub.write_text("from ...utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_relative_level_too_deep(tmp_path):
    # Relative import that goes above the project root → skipped without crash.
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "utils.py"
    mod.write_text("def _helper(): pass\n")
    # A file at the top level trying to go up 5 packages (impossible).
    top = tmp_path / "top.py"
    top.write_text("from .....utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    # The over-deep import is silently skipped; no crash.
    assert isinstance(result, set)


def test_find_cross_file_imports_abs_pkg_package_prefix():
    # abs_pkg="tests" → "from tests.block_1 import _MODEL"
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "test_fn.py", abs_pkg="tests"
    )
    assert result == ["from tests.block_1 import _MODEL"]


def test_find_cross_file_imports_abs_pkg_root_level():
    # abs_pkg="" → "from block_1 import _MODEL" (no package prefix)
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "test_fn.py", abs_pkg=""
    )
    assert result == ["from block_1 import _MODEL"]
