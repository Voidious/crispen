from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.code_gen import (
    _find_cross_file_imports,
    _relative_import_prefix,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .test_import_analysis import _classified, _make_entity


def _plan(placements=None) -> FileLimiterPlan:
    return FileLimiterPlan(set3_migrate=[], placements=placements or [], abort=False)


def _abort_plan() -> FileLimiterPlan:
    return FileLimiterPlan(set3_migrate=[], placements=[], abort=True)


def test_generate_single_entity_migration():
    source = "import os\n\ndef foo():\n    os.getcwd()\n"
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert result.abort is False
    assert "utils.py" in result.new_files
    new_src = result.new_files["utils.py"]
    assert "import os" in new_src
    assert "def foo():" in new_src
    # Original should not have foo's def anymore
    assert "def foo():" not in result.original_source
    # But should have a re-export
    assert "from .utils import foo" in result.original_source


def test_generate_private_entity_no_reexport():
    source = "def _helper():\n    pass\n"
    entity = _make_entity("_helper", 1, 2)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["_helper"], target_file="private.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "from .private import" not in result.original_source


def test_generate_entity_not_in_source_map():
    # Group has entity name not in classified.entities → entity skipped in new file.
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    c = _classified(entities=[entity])
    # "ghost" is in the group but has no matching entity
    plan = _plan([GroupPlacement(group=["foo", "ghost"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "utils.py" in result.new_files
    # "ghost" produces no source so only "foo" appears
    new_src = result.new_files["utils.py"]
    assert "def foo():" in new_src


def test_generate_no_imports_needed():
    # Entity uses no imports → no import section in new file.
    source = "def add(a, b):\n    return a + b\n"
    entity = _make_entity("add", 1, 2)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["add"], target_file="math_utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["math_utils.py"]
    # No "import" prefix expected
    assert not new_src.startswith("import")
    assert "def add" in new_src


def test_generate_multiple_groups_same_file():
    source = textwrap.dedent(
        """\
        import os

        def foo():
            pass

        def bar():
            pass
        """
    )
    e_foo = _make_entity("foo", 3, 4)
    e_bar = _make_entity("bar", 6, 7)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan(
        [
            GroupPlacement(group=["foo"], target_file="utils.py"),
            GroupPlacement(group=["bar"], target_file="utils.py"),
        ]
    )
    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert "def foo():" in new_src
    assert "def bar():" in new_src


def test_generate_multiple_different_target_files():
    source = "def foo():\n    pass\n\ndef bar():\n    pass\n"
    e_foo = _make_entity("foo", 1, 2)
    e_bar = _make_entity("bar", 4, 5)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan(
        [
            GroupPlacement(group=["foo"], target_file="foo_module.py"),
            GroupPlacement(group=["bar"], target_file="bar_module.py"),
        ]
    )
    result = generate_file_splits(c, plan, source, "big.py")

    assert "foo_module.py" in result.new_files
    assert "bar_module.py" in result.new_files
    assert "def foo():" in result.new_files["foo_module.py"]
    assert "def bar():" in result.new_files["bar_module.py"]
    assert "from .bar_module import bar" in result.original_source
    assert "from .foo_module import foo" in result.original_source


def test_generate_future_import_not_duplicated_when_in_entity_source():
    # Entity source itself contains `from __future__ import annotations`
    # (e.g. the _block_1 TOP_LEVEL entity which IS the file's import block).
    # It must appear only once at the top of the new file, not again inside
    # the entity source, which would cause a SyntaxError.
    source = textwrap.dedent(
        """\
        from __future__ import annotations

        \"\"\"Module docstring.\"\"\"

        from __future__ import annotations

        import os

        _CONST = 42
    """
    )
    # _block_1 spans the whole file and contains the future import + constants.
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 9, ["_CONST"])
    c = _classified(entities=[e_block])
    plan = _plan([GroupPlacement(group=["_block_1"], target_file="constants.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["constants.py"]
    assert new_src.count("from __future__ import annotations") == 1
    # Must be at the very start of the file (before any other code).
    first_non_blank = next(line for line in new_src.splitlines() if line.strip())
    assert first_non_blank == "from __future__ import annotations"


def test_generate_future_import_always_included():
    source = "from __future__ import annotations\n\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert "from __future__ import annotations" in new_src


def test_find_cross_file_imports_basic():
    # fn_a references _MODEL which is defined in block_1.py
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "llm_extract.py"
    )
    assert result == ["from .block_1 import _MODEL"]


def test_find_cross_file_imports_same_file_excluded():
    # _MODEL goes to the same file as fn_a → no cross-file import needed
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "llm_extract.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "llm_extract.py"
    )
    assert result == []


def test_find_cross_file_imports_no_match():
    # Referenced name not in name_to_target_file → no cross-file import
    entity_source_map = {"fn_a": "def fn_a():\n    return os.getcwd()\n"}
    result = _find_cross_file_imports(["fn_a"], entity_source_map, {}, "utils.py")
    assert result == []


def test_find_cross_file_imports_entity_not_in_map():
    # Entity name not in entity_source_map → treated as empty source, no imports
    result = _find_cross_file_imports(["ghost"], {}, {"x": "other.py"}, "utils.py")
    assert result == []


def test_relative_import_prefix_same_directory():
    # Both files at the root level → single dot.
    assert _relative_import_prefix("a.py", "b.py") == ".b"


def test_relative_import_prefix_sibling_subdir():
    # from_file is in sub/, to_file is in helpers/ → go up one, then down.
    assert _relative_import_prefix("sub/a.py", "helpers/b.py") == "..helpers.b"


def test_relative_import_prefix_same_subdir():
    # Both in the same subdirectory → single dot.
    assert _relative_import_prefix("sub/a.py", "sub/b.py") == ".b"


def test_relative_import_prefix_to_nested():
    # to_file is in a subdirectory of root while from_file is at root.
    assert _relative_import_prefix("a.py", "helpers/b.py") == ".helpers.b"


def test_find_cross_file_imports_cross_directory():
    # fn_a is in tests/test.py; helper is in helpers/entities.py.
    # Cross-directory import needs ".." to go up from tests/ to root.
    entity_source_map = {"fn_a": "def fn_a():\n    return _helper()\n"}
    name_to_target_file = {"_helper": "helpers/entities.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "tests/test.py"
    )
    assert result == ["from ..helpers.entities import _helper"]


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


def test_generate_no_circular_import_when_helper_referenced_by_migrated():
    # Integration test: _run stays in original and is used by test_fn (migrated).
    # Without the fix: original → helpers.py (re-export) and helpers.py → original.
    # With the fix: _run is moved into helpers.py; original imports _run from helpers.
    source = textwrap.dedent(
        """\
        def _run(x):
            return x

        def test_fn(tmp_path):
            return _run(tmp_path)
    """
    )
    e_run = _make_entity("_run", 1, 2)
    e_test = _make_entity("test_fn", 4, 5)
    c = _classified(entities=[e_run, e_test])
    plan = _plan([GroupPlacement(group=["test_fn"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "original.py")

    helpers_src = result.new_files["helpers.py"]
    # _run is defined in helpers.py (extracted), not imported from original
    assert "def _run" in helpers_src
    assert "from .original import _run" not in helpers_src
    # original re-imports _run from helpers.py (since it's still used there via
    # non-migrated code — but in this minimal example there's nothing left)
    # At minimum, no circular self-import exists
    assert "from .original import" not in helpers_src


def test_generate_prunes_unused_names_from_multiname_import():
    # foo uses only List, not Dict; the new file's import should be narrowed.
    source = "from typing import Dict, List\n\ndef foo(x: List):\n    return x\n"
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert "from typing import List" in new_src
    assert "Dict" not in new_src


def test_generate_prunes_fully_unused_import_from_original():
    # import os is only used by foo; after foo migrates the original no longer
    # needs os, so the import should be removed.
    source = "import os\n\ndef foo():\n    os.getcwd()\n\ndef bar():\n    return 1\n"
    e_foo = _make_entity("foo", 3, 4)
    e_bar = _make_entity("bar", 6, 7)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "from .utils import foo" in result.original_source
    assert "import os" not in result.original_source
    assert "def bar():" in result.original_source


def test_generate_narrows_partial_unused_import_in_original():
    # foo uses Dict; bar uses List.  After foo migrates, Dict should be
    # stripped from the original's import while List is kept.
    source = (
        "from typing import Dict, List\n\n"
        "def foo(x: Dict):\n    return x\n\n"
        "def bar(x: List):\n    return x\n"
    )
    e_foo = _make_entity("foo", 3, 4)
    e_bar = _make_entity("bar", 6, 7)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "from typing import List" in result.original_source
    assert "Dict" not in result.original_source


def test_generate_migrated_top_level_import_names_not_in_cross_file_imports():
    # Regression: when a TOP_LEVEL entity containing "from dataclasses import
    # dataclass" is migrated, the name "dataclass" must NOT be added to the
    # name→target-file map.  A FUNCTION entity in a separate new file that also
    # uses dataclass should get "from dataclasses import dataclass" (via
    # _find_needed_imports) rather than "from .constants import dataclass" (a
    # spurious cross-file import that would fail at runtime because constants.py
    # never exports dataclass).
    source = (
        "from dataclasses import dataclass\n\n"
        "_CONST = 42\n\n"
        "def make():\n    return dataclass\n"
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["dataclass", "_CONST"])
    e_make = _make_entity("make", 5, 6)
    c = _classified(entities=[e_block, e_make])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["make"], target_file="utils.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    utils_src = result.new_files["utils.py"]
    # Must import dataclass from the stdlib, not from constants.py
    assert "from dataclasses import dataclass" in utils_src
    assert "from .constants import dataclass" not in utils_src


def test_generate_private_entity_reexported_when_external_caller(tmp_path):
    # Private entity is re-exported when an external file imports it.
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    mod = pkg / "big.py"
    mod.write_text("def _helper():\n    pass\n")
    caller = tmp_path / "tests" / "test_big.py"
    caller.parent.mkdir()
    caller.write_text("from mypkg.big import _helper\n")

    source = "def _helper():\n    pass\n"
    entity = _make_entity("_helper", 1, 2)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["_helper"], target_file="private.py")])

    result = generate_file_splits(c, plan, source, str(mod))

    assert "from .private import _helper" in result.original_source


def test_generate_file_splits_removes_inline_redundant_imports():
    # When a split new file has both a top-level import and an inline re-import
    # of the same name, the inline one should be removed.
    source = textwrap.dedent(
        """\
        from mymod import Helper

        def test_uses_helper():
            from mymod import Helper
            assert Helper()
        """
    )
    entity = _make_entity("test_uses_helper", 3, 5)
    c = _classified(entities=[entity])
    plan = _plan(
        [GroupPlacement(group=["test_uses_helper"], target_file="test_split.py")]
    )
    result = generate_file_splits(c, plan, source, "big.py")
    new_src = result.new_files["test_split.py"]
    # The inline re-import should be removed; the module-level one covers it.
    assert new_src.count("from mymod import Helper") == 1


def test_generate_test_file_reexports_use_absolute_imports(tmp_path):
    # When the original is a test file, re-exports in the updated original
    # must use absolute imports so pytest can load the file.
    (tmp_path / "pyproject.toml").touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_engine.py"
    test_file.write_text("")

    source = "import os\n\ndef foo():\n    os.getcwd()\n"
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="test_helpers.py")])

    result = generate_file_splits(c, plan, source, str(test_file))

    assert "from tests.test_helpers import foo" in result.original_source
    assert "from .test_helpers import foo" not in result.original_source


def test_generate_test_file_cross_imports_use_absolute_imports(tmp_path):
    # Cross-file imports in generated test split files must also be absolute.
    (tmp_path / "pyproject.toml").touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_engine.py"
    test_file.write_text("")

    source = "_CONST = 42\n\ndef test_fn():\n    return _CONST\n"
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_fn = _make_entity("test_fn", 3, 4)
    c = _classified(entities=[e_block, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="test_constants.py"),
            GroupPlacement(group=["test_fn"], target_file="test_fns.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, str(test_file))

    fn_src = result.new_files["test_fns.py"]
    assert "from tests.test_constants import _CONST" in fn_src
    assert "from .test_constants import _CONST" not in fn_src


def test_generate_file_splits_subdir_name_uses_init_as_original_basename():
    # When subdir_name="service", the dependency graph treats "service/__init__.py"
    # as the original file node.  Because main (public) is re-exported from
    # __init__, _extract_shared_helpers pulls helper into service/main.py to
    # break the __init__ → main → __init__ cycle.  The split must not abort.
    source = "def helper():\n    return 1\n\ndef main():\n    return helper()\n"
    e_helper = _make_entity("helper", 1, 2)
    e_main = _make_entity("main", 4, 5)
    c = _classified(entities=[e_helper, e_main])
    # Only main is migrated; helper stays in "original" (→ service/__init__.py).
    plan = _plan([GroupPlacement(group=["main"], target_file="service/main.py")])

    result = generate_file_splits(c, plan, source, "service.py", subdir_name="service")

    assert not result.abort
    # helper is extracted into service/main.py to break the re-export cycle.
    main_src = result.new_files["service/main.py"]
    assert "def helper" in main_src
    assert "def main" in main_src
    # Re-exports use the short relative prefix ".main", not ".service.main".
    assert "from .main import" in result.original_source
    assert "from .service.main" not in result.original_source


def test_generate_file_splits_subdir_name_re_exports_use_relative_prefix():
    # With subdir_name set (non-test), re-exports in the "original" source
    # (which becomes __init__.py) use ".utils" not ".service.utils".
    # target_file already has the "service/" prefix (added by runner.py).
    source = "def foo():\n    pass\n"
    e_foo = _make_entity("foo", 1, 2)
    c = _classified(entities=[e_foo])
    plan = _plan([GroupPlacement(group=["foo"], target_file="service/utils.py")])

    result = generate_file_splits(c, plan, source, "service.py", subdir_name="service")

    assert not result.abort
    assert "from .utils import foo" in result.original_source
    assert "from .service.utils" not in result.original_source


def test_generate_file_splits_subdir_name_cross_file_uses_relative():
    # In subdir mode, cross-file imports between new files use relative imports
    # even when the original file is a test (abs_pkg would normally apply).
    # NOTE: runner.py prefixes target_file with subdir_name before this call,
    # so target_files already include "svc/" here.
    source = "def helper():\n    return 1\n\ndef test_fn():\n    return helper()\n"
    e_helper = _make_entity("helper", 1, 2)
    e_test = _make_entity("test_fn", 4, 5)
    c = _classified(entities=[e_helper, e_test])
    plan = _plan(
        [
            GroupPlacement(group=["helper"], target_file="svc/helpers.py"),
            GroupPlacement(group=["test_fn"], target_file="svc/test_fns.py"),
        ]
    )

    # Use a path that looks like a test file so abs_pkg would normally be set.
    result = generate_file_splits(
        c, plan, source, "tests/test_svc.py", subdir_name="svc"
    )

    assert not result.abort
    # Cross-file import from test_fns.py to helpers.py should be relative.
    test_src = result.new_files["svc/test_fns.py"]
    assert "from .helpers import helper" in test_src


def test_generate_file_splits_subdir_bumps_needed_imports():
    # In subdir-split mode, relative imports from the original file that appear
    # in new sub-files must be incremented by one level so they still resolve
    # correctly from inside the subdirectory package.
    source = "from .sibling import CONST\n\ndef foo():\n    return CONST\n"
    e_foo = _make_entity("foo", 3, 4)
    c = _classified(entities=[e_foo])
    plan = _plan([GroupPlacement(group=["foo"], target_file="service/utils.py")])

    result = generate_file_splits(c, plan, source, "service.py", subdir_name="service")

    assert not result.abort
    utils_src = result.new_files["service/utils.py"]
    assert "from ..sibling import CONST" in utils_src
    assert "from .sibling import CONST" not in utils_src


def test_generate_file_splits_subdir_bumps_init_imports():
    # In subdir-split mode, relative imports in the non-migrated original source
    # (which becomes subdir/__init__.py) must also be bumped by one level so
    # they still point at the correct modules from inside the package.
    source2 = (
        "from .. import llm_client\n"
        "from .base import Base\n\n"
        "def stayed():\n    return llm_client, Base\n\n"
        "def migrated():\n    pass\n"
    )
    e_stayed2 = _make_entity("stayed", 4, 5)
    e_migrated2 = _make_entity("migrated", 7, 8)
    c = _classified(entities=[e_stayed2, e_migrated2])
    plan = _plan([GroupPlacement(group=["migrated"], target_file="pkg/helpers.py")])

    result = generate_file_splits(c, plan, source2, "pkg.py", subdir_name="pkg")

    assert not result.abort
    init_src = result.original_source
    assert "from ... import llm_client" in init_src
    assert "from ..base import Base" in init_src
    assert "from .. import llm_client" not in init_src
    assert "from .base import Base" not in init_src


def test_generate_file_splits_subdir_bumps_two_levels_deep():
    # When the LLM places a new file two directories deep (e.g.
    # "pkg/pkg/core.py"), relative imports must be bumped by 2 dots, not 1.
    # This matches the real-world scenario where subdir_name="pkg" but the
    # advisor proposes "pkg/pkg/core.py" as a target.
    source = "from .. import llm_client\n\ndef func():\n    return llm_client\n"
    e_func = _make_entity("func", 3, 4)
    c = _classified(entities=[e_func])
    plan = _plan([GroupPlacement(group=["func"], target_file="pkg/pkg/core.py")])

    result = generate_file_splits(c, plan, source, "pkg.py", subdir_name="pkg")

    assert not result.abort
    core_src = result.new_files["pkg/pkg/core.py"]
    # 2 levels deep → original ".." becomes "...." (4 dots)
    assert "from .... import llm_client" in core_src
    assert "from .. import llm_client" not in core_src
    assert "from ... import llm_client" not in core_src


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
