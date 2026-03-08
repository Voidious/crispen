from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    _find_cross_file_imports,
    _merge_from_imports,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .test_shared_helpers import _abort_plan, _classified, _make_entity, _plan
import textwrap


def test_generate_abort_plan():
    plan = _abort_plan()
    c = _classified()
    result = generate_file_splits(c, plan, "def foo():\n    pass\n", "big.py")
    assert result.abort is True
    assert result.new_files == {}
    assert result.original_source == "def foo():\n    pass\n"


def test_generate_empty_placements():
    plan = _plan()  # placements=[]
    c = _classified()
    source = "def foo():\n    pass\n"
    result = generate_file_splits(c, plan, source, "big.py")
    assert result.abort is False
    assert result.new_files == {}
    assert result.original_source == source


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


def test_merge_from_imports_no_overlap():
    imports = ["from .a import x", "from .b import y"]
    assert _merge_from_imports(imports) == ["from .a import x", "from .b import y"]


def test_merge_from_imports_overlapping():
    imports = ["from .conv import A, C", "from .conv import B, C"]
    result = _merge_from_imports(imports)
    assert result == ["from .conv import A, B, C"]


def test_merge_from_imports_deduplicates_names():
    imports = ["from .m import foo, bar", "from .m import bar, baz"]
    result = _merge_from_imports(imports)
    assert result == ["from .m import bar, baz, foo"]


def test_merge_from_imports_preserves_plain_imports():
    imports = ["import os", "from .m import x", "import sys"]
    result = _merge_from_imports(imports)
    assert result == ["from .m import x", "import os", "import sys"]


def test_merge_from_imports_empty():
    assert _merge_from_imports([]) == []


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


def test_generate_cross_file_import_no_duplicate_names():
    # Two entities (fn_a and fn_b) migrate to the same new file.
    # fn_a uses X and Z from helpers; fn_b uses Y and Z from helpers.
    # The new file must get exactly one merged import with X, Y, Z —
    # not two overlapping imports that both include Z (F811 redefinition).
    source = textwrap.dedent(
        """\
        X = 1
        Y = 2
        Z = 3

        def fn_a():
            return X + Z

        def fn_b():
            return Y + Z
        """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["X", "Y", "Z"])
    e_a = _make_entity("fn_a", 5, 6)
    e_b = _make_entity("fn_b", 8, 9)
    c = _classified(entities=[e_block, e_a, e_b])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["fn_a", "fn_b"], target_file="funcs.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    funcs_src = result.new_files["funcs.py"]
    # Both fn_a and fn_b are present
    assert "def fn_a" in funcs_src
    assert "def fn_b" in funcs_src
    # Only ONE import from constants, with X, Y, Z each appearing exactly once
    import_lines = [
        ln for ln in funcs_src.splitlines() if "from .constants import" in ln
    ]
    assert len(import_lines) == 1, f"Expected 1 import line, got: {import_lines}"
    assert import_lines[0].count("X") == 1
    assert import_lines[0].count("Y") == 1
    assert import_lines[0].count("Z") == 1


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


def test_generate_aborts_on_cycle_through_original_test_subdir():
    # In a test-file subdir split non_migrated_home ("test_svc.py") differs
    # from original_basename ("svc/__init__.py").  The cycle detection must
    # treat the original test file as its own graph node:
    #
    # _CONFIG stays in test_svc.py (TOP_LEVEL, non-migrated).
    # _helper is migrated to svc/test_helpers.py and references _CONFIG.
    # test_fn (non-migrated) calls _helper → test_svc.py re-exports _helper.
    # Cycle: test_svc.py → svc/test_helpers.py (re-export of _helper)
    #              → test_svc.py (via `from ..test_svc import _CONFIG`).
    source = textwrap.dedent(
        """\
        _CONFIG = "value"

        def _helper():
            return _CONFIG

        def test_fn():
            return _helper()
    """
    )
    e_config = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONFIG"])
    e_helper = _make_entity("_helper", 3, 4)
    e_test = _make_entity("test_fn", 6, 7)
    c = _classified(entities=[e_config, e_helper, e_test])
    plan = _plan([GroupPlacement(group=["_helper"], target_file="svc/test_helpers.py")])

    result = generate_file_splits(
        c, plan, source, "tests/test_svc.py", subdir_name="svc"
    )

    # svc/test_helpers.py imports _CONFIG from test_svc.py, and test_svc.py
    # re-exports _helper from svc/test_helpers.py → circular import → abort.
    assert result.abort is True
    assert result.new_files == {}


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
