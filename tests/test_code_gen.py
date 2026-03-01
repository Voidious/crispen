"""Tests for file_limiter.code_gen — 100% branch coverage."""

from __future__ import annotations

import textwrap

from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import (
    ImportInfo,
    _add_re_exports,
    _collect_name_loads,
    _extract_import_info,
    _extract_shared_helpers,
    _find_cross_file_imports,
    _find_needed_imports,
    _remove_entity_lines,
    _target_module_name,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(name: str, start: int, end: int, defines=None) -> Entity:
    return Entity(EntityKind.FUNCTION, name, start, end, defines or [name])


def _classified(
    *, entities=None, set_2_groups=None, set_3_groups=None
) -> ClassifiedEntities:
    return ClassifiedEntities(
        entities=entities or [],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=set_2_groups or [],
        set_3_groups=set_3_groups or [],
        abort=False,
    )


def _plan(placements=None) -> FileLimiterPlan:
    return FileLimiterPlan(set3_migrate=[], placements=placements or [], abort=False)


def _abort_plan() -> FileLimiterPlan:
    return FileLimiterPlan(set3_migrate=[], placements=[], abort=True)


# ---------------------------------------------------------------------------
# _collect_name_loads
# ---------------------------------------------------------------------------


def test_collect_name_loads_basic():
    source = "x = foo + bar"
    names = _collect_name_loads(source)
    assert "foo" in names
    assert "bar" in names


def test_collect_name_loads_store_not_included():
    source = "x = 1"
    names = _collect_name_loads(source)
    # x is a Store, not a Load
    assert "x" not in names


def test_collect_name_loads_syntax_error():
    assert _collect_name_loads("def (invalid") == set()


# ---------------------------------------------------------------------------
# _extract_import_info
# ---------------------------------------------------------------------------


def test_extract_import_info_syntax_error():
    assert _extract_import_info("def (invalid") == []


def test_extract_import_info_plain_import():
    infos = _extract_import_info("import os\n")
    assert len(infos) == 1
    assert "os" in infos[0].names
    assert infos[0].is_future is False


def test_extract_import_info_import_with_asname():
    infos = _extract_import_info("import os as operating_system\n")
    assert infos[0].names == ["operating_system"]


def test_extract_import_info_dotted_import():
    infos = _extract_import_info("import os.path\n")
    assert infos[0].names == ["os"]


def test_extract_import_info_from_import():
    infos = _extract_import_info("from pathlib import Path\n")
    assert "Path" in infos[0].names
    assert infos[0].is_future is False


def test_extract_import_info_from_import_with_asname():
    infos = _extract_import_info("from pathlib import Path as P\n")
    assert infos[0].names == ["P"]


def test_extract_import_info_future_import():
    infos = _extract_import_info("from __future__ import annotations\n")
    assert infos[0].is_future is True
    assert "annotations" in infos[0].names


def test_extract_import_info_skips_non_imports():
    infos = _extract_import_info("def foo():\n    pass\n")
    assert infos == []


def test_extract_import_info_multiple():
    source = "import os\nfrom pathlib import Path\n"
    infos = _extract_import_info(source)
    assert len(infos) == 2


# ---------------------------------------------------------------------------
# _find_needed_imports
# ---------------------------------------------------------------------------


def test_find_needed_imports_referenced_name():
    # Entity references "os"; import for "os" should be included.
    entity_src_map = {"foo": "def foo():\n    os.getcwd()\n"}
    infos = [ImportInfo(names=["os"], source="import os", is_future=False)]
    result = _find_needed_imports(["foo"], entity_src_map, infos, {"foo"})
    assert "import os" in result


def test_find_needed_imports_unreferenced_name():
    # Entity doesn't reference "sys"; import should be excluded.
    entity_src_map = {"foo": "def foo():\n    pass\n"}
    infos = [ImportInfo(names=["sys"], source="import sys", is_future=False)]
    result = _find_needed_imports(["foo"], entity_src_map, infos, {"foo"})
    assert result == []


def test_find_needed_imports_future_always_included():
    # __future__ import is always included regardless of entity references.
    entity_src_map = {"foo": "def foo():\n    pass\n"}
    infos = [
        ImportInfo(
            names=["annotations"],
            source="from __future__ import annotations",
            is_future=True,
        )
    ]
    result = _find_needed_imports(["foo"], entity_src_map, infos, {"foo"})
    assert "from __future__ import annotations" in result


def test_find_needed_imports_deduplicates():
    # Two ImportInfo entries with the same source string → only one included.
    entity_src_map = {"foo": "def foo():\n    os.getcwd()\n"}
    infos = [
        ImportInfo(names=["os"], source="import os", is_future=False),
        ImportInfo(names=["os"], source="import os", is_future=False),  # duplicate
    ]
    result = _find_needed_imports(["foo"], entity_src_map, infos, {"foo"})
    assert result.count("import os") == 1


def test_find_needed_imports_entity_not_in_map():
    # Entity name not in entity_source_map → treated as empty source.
    infos = [ImportInfo(names=["os"], source="import os", is_future=False)]
    result = _find_needed_imports(["ghost"], {}, infos, set())
    assert result == []


# ---------------------------------------------------------------------------
# _target_module_name
# ---------------------------------------------------------------------------


def test_target_module_name_simple():
    assert _target_module_name("utils.py") == "utils"


def test_target_module_name_nested():
    assert _target_module_name("helpers/io.py") == "helpers.io"


# ---------------------------------------------------------------------------
# _remove_entity_lines
# ---------------------------------------------------------------------------


def test_remove_entity_lines_removes_range():
    source = "line1\nline2\nline3\nline4\n"
    entity = _make_entity("foo", 2, 3)
    entity_map = {"foo": entity}
    result = _remove_entity_lines(source, {"foo"}, entity_map)
    assert "line1" in result
    assert "line2" not in result
    assert "line3" not in result
    assert "line4" in result


def test_remove_entity_lines_name_not_in_map():
    # Name not in entity_map → nothing removed.
    source = "line1\nline2\n"
    result = _remove_entity_lines(source, {"ghost"}, {})
    assert result == source


# ---------------------------------------------------------------------------
# _add_re_exports
# ---------------------------------------------------------------------------


def test_add_re_exports_all_private_no_change():
    # Private name not called anywhere in remaining source → no import added.
    source = "import os\n\ndef _helper():\n    pass\n"
    entity = _make_entity("_helper", 3, 4)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"_helper": entity})
    assert result == source


def test_add_re_exports_private_referenced_in_source():
    # Private name still called in remaining source → import is added.
    source = "import os\n\n_helper()\n"
    entity = _make_entity("_helper", 3, 3)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"_helper": entity})
    assert "from .utils import _helper" in result


def test_add_re_exports_public_inserted_after_imports():
    source = "import os\n\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 3, 4)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity})
    assert "from .utils import foo" in result
    # Re-export line should come after "import os"
    lines = result.splitlines()
    import_idx = next(i for i, l in enumerate(lines) if "import os" in l)
    reexport_idx = next(i for i, l in enumerate(lines) if "from .utils import foo" in l)
    assert reexport_idx > import_idx


def test_add_re_exports_no_import_in_source():
    # No imports in source → re-export inserted at beginning.
    source = "\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 2, 3)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity})
    assert "from .utils import foo" in result


def test_add_re_exports_from_import_line():
    # "from pathlib import Path" should be detected as an import line.
    source = "from pathlib import Path\n\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 3, 4)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity})
    lines = result.splitlines()
    from_import_idx = next(
        i for i, l in enumerate(lines) if "from pathlib import Path" in l
    )
    reexport_idx = next(i for i, l in enumerate(lines) if "from .utils import foo" in l)
    assert reexport_idx > from_import_idx


def test_add_re_exports_multiple_targets_sorted():
    source = "import os\n"
    e1 = _make_entity("foo", 1, 2)
    e2 = _make_entity("bar", 3, 4)
    placements = [
        GroupPlacement(group=["foo"], target_file="b_module.py"),
        GroupPlacement(group=["bar"], target_file="a_module.py"),
    ]
    result = _add_re_exports(source, placements, {"foo": e1, "bar": e2})
    # a_module comes before b_module (sorted)
    a_idx = result.index("a_module")
    b_idx = result.index("b_module")
    assert a_idx < b_idx


def test_add_re_exports_mixed_public_private():
    source = "import os\n"
    entity_map = {
        "pub": _make_entity("pub", 1, 2),
        "_priv": _make_entity("_priv", 3, 4),
    }
    placement = GroupPlacement(group=["pub", "_priv"], target_file="utils.py")
    result = _add_re_exports(source, [placement], entity_map)
    # Only "pub" in re-export, not "_priv"
    assert "pub" in result
    assert "_priv" not in result


# ---------------------------------------------------------------------------
# generate_file_splits
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _find_cross_file_imports
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# generate_file_splits — cross-file import integration
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _extract_shared_helpers
# ---------------------------------------------------------------------------


def _make_classified(entities, migrated_names=None):
    migrated = set(migrated_names or [])
    return (
        ClassifiedEntities(
            entities=entities,
            entity_class={},
            graph={},
            set_1=[],
            set_2_groups=[],
            set_3_groups=[],
            abort=False,
        ),
        migrated,
    )


def test_extract_shared_helpers_extracts_referenced_function():
    # _helper is non-migrated, test_fn (migrated to helpers.py) references it.
    e_helper = Entity(EntityKind.FUNCTION, "_helper", 1, 2, ["_helper"])
    e_test = Entity(EntityKind.FUNCTION, "test_fn", 4, 6, ["test_fn"])
    classified, migrated_names = _make_classified([e_helper, e_test], ["test_fn"])
    entity_map = {"_helper": e_helper, "test_fn": e_test}
    entity_source_map = {
        "_helper": "def _helper():\n    pass",
        "test_fn": "def test_fn():\n    return _helper()",
    }
    file_entity_names = {"helpers.py": ["test_fn"]}
    name_to_target_file = {"_helper": "original.py", "test_fn": "helpers.py"}

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    # _helper extracted into helpers.py (prepended before test_fn)
    assert file_entity_names["helpers.py"] == ["_helper", "test_fn"]
    assert "_helper" in migrated_names
    assert name_to_target_file["_helper"] == "helpers.py"
    assert len(synthetic) == 1
    assert synthetic[0].group == ["_helper"]
    assert synthetic[0].target_file == "helpers.py"


def test_extract_shared_helpers_skips_top_level_entities():
    # TOP_LEVEL entities are not extracted (only FUNCTION/CLASS).
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_test = Entity(EntityKind.FUNCTION, "test_fn", 3, 4, ["test_fn"])
    classified, migrated_names = _make_classified([e_block, e_test], ["test_fn"])
    entity_map = {"_block_1": e_block, "test_fn": e_test}
    entity_source_map = {
        "_block_1": "_CONST = 42",
        "test_fn": "def test_fn():\n    return _CONST",
    }
    file_entity_names = {"helpers.py": ["test_fn"]}
    name_to_target_file = {"_CONST": "original.py", "test_fn": "helpers.py"}

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    assert "_block_1" not in migrated_names
    assert file_entity_names["helpers.py"] == ["test_fn"]
    assert synthetic == []


def test_extract_shared_helpers_extracts_only_once_for_multiple_refs():
    # _helper referenced twice in the same migrated entity → extracted once.
    e_helper = Entity(EntityKind.FUNCTION, "_helper", 1, 2, ["_helper"])
    e_test = Entity(EntityKind.FUNCTION, "test_fn", 4, 6, ["test_fn"])
    classified, migrated_names = _make_classified([e_helper, e_test], ["test_fn"])
    entity_map = {"_helper": e_helper, "test_fn": e_test}
    entity_source_map = {
        "_helper": "def _helper():\n    pass",
        "test_fn": "def test_fn():\n    _helper()\n    _helper()",
    }
    file_entity_names = {"helpers.py": ["test_fn"]}
    name_to_target_file = {"_helper": "original.py", "test_fn": "helpers.py"}

    _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    assert file_entity_names["helpers.py"].count("_helper") == 1


def test_extract_shared_helpers_skips_name_already_pointing_to_other_target():
    # A non-migrated FUNCTION entity whose defined name already points to a
    # non-original target in name_to_target_file (e.g. a migrated entity also
    # defines it) should not be added to defined_to_entity.
    e_helper = Entity(EntityKind.FUNCTION, "_helper", 1, 2, ["_helper"])
    e_test = Entity(EntityKind.FUNCTION, "test_fn", 4, 5, ["test_fn"])
    classified, migrated_names = _make_classified([e_helper, e_test], ["test_fn"])
    entity_map = {"_helper": e_helper, "test_fn": e_test}
    entity_source_map = {
        "_helper": "def _helper(): pass",
        "test_fn": "def test_fn(): return _helper()",
    }
    file_entity_names = {"helpers.py": ["test_fn"]}
    # _helper already points to helpers.py (not original) — skip it
    name_to_target_file = {"_helper": "helpers.py", "test_fn": "helpers.py"}

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    assert "_helper" not in migrated_names
    assert synthetic == []


def test_extract_shared_helpers_no_extraction_when_no_original_dep():
    # test_fn references other_fn which is also migrated → no extraction needed.
    e_other = Entity(EntityKind.FUNCTION, "other_fn", 1, 2, ["other_fn"])
    e_test = Entity(EntityKind.FUNCTION, "test_fn", 4, 5, ["test_fn"])
    classified, migrated_names = _make_classified(
        [e_other, e_test], ["test_fn", "other_fn"]
    )
    entity_map = {"other_fn": e_other, "test_fn": e_test}
    entity_source_map = {
        "other_fn": "def other_fn():\n    pass",
        "test_fn": "def test_fn():\n    return other_fn()",
    }
    file_entity_names = {"helpers.py": ["test_fn", "other_fn"]}
    name_to_target_file = {"other_fn": "helpers.py", "test_fn": "helpers.py"}

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    assert synthetic == []
    assert file_entity_names["helpers.py"] == ["test_fn", "other_fn"]


def test_extract_shared_helpers_transitive_pull_in():
    # _helper_a is directly wanted by fn_a (in f1.py).
    # _helper_a's source calls _helper_b (non-migrated, in original).
    # _helper_b must be transitively extracted into f1.py to prevent an
    # O→f1.py cycle (f1.py imports _helper_a which calls _helper_b in original;
    # original re-exports _helper_a from f1.py → cycle).
    e_a = Entity(EntityKind.FUNCTION, "_helper_a", 1, 2, ["_helper_a"])
    e_b = Entity(EntityKind.FUNCTION, "_helper_b", 3, 4, ["_helper_b"])
    e_fn = Entity(EntityKind.FUNCTION, "fn_a", 6, 7, ["fn_a"])
    classified, migrated_names = _make_classified([e_a, e_b, e_fn], ["fn_a"])
    entity_map = {"_helper_a": e_a, "_helper_b": e_b, "fn_a": e_fn}
    entity_source_map = {
        "_helper_a": "def _helper_a():\n    _helper_b()",
        "_helper_b": "def _helper_b():\n    pass",
        "fn_a": "def fn_a():\n    _helper_a()",
    }
    file_entity_names = {"f1.py": ["fn_a"]}
    name_to_target_file = {
        "_helper_a": "original.py",
        "_helper_b": "original.py",
        "fn_a": "f1.py",
    }

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    # Both helpers extracted into f1.py.
    assert "_helper_a" in file_entity_names["f1.py"]
    assert "_helper_b" in file_entity_names["f1.py"]
    assert "_helper_a" in migrated_names
    assert "_helper_b" in migrated_names
    assert name_to_target_file["_helper_a"] == "f1.py"
    assert name_to_target_file["_helper_b"] == "f1.py"
    assert len(synthetic) == 2


def test_extract_shared_helpers_scc_prevents_new_to_new_cycle():
    # helper_a is wanted by f1.py; helper_b is wanted by f2.py.
    # They mutually reference each other → one SCC → must go to the same file
    # to prevent the F1→F2→F1 import cycle.
    e_a = Entity(EntityKind.FUNCTION, "helper_a", 1, 2, ["helper_a"])
    e_b = Entity(EntityKind.FUNCTION, "helper_b", 3, 4, ["helper_b"])
    e_fn1 = Entity(EntityKind.FUNCTION, "fn_1", 6, 7, ["fn_1"])
    e_fn2 = Entity(EntityKind.FUNCTION, "fn_2", 9, 10, ["fn_2"])
    classified = ClassifiedEntities(
        entities=[e_a, e_b, e_fn1, e_fn2],
        entity_class={},
        graph={
            "helper_a": {"helper_b"},
            "helper_b": {"helper_a"},
            "fn_1": set(),
            "fn_2": set(),
        },
        set_1=[],
        set_2_groups=[],
        set_3_groups=[],
        abort=False,
    )
    migrated_names = {"fn_1", "fn_2"}
    entity_map = {"helper_a": e_a, "helper_b": e_b, "fn_1": e_fn1, "fn_2": e_fn2}
    entity_source_map = {
        "helper_a": "def helper_a():\n    helper_b()",
        "helper_b": "def helper_b():\n    helper_a()",
        "fn_1": "def fn_1():\n    helper_a()",
        "fn_2": "def fn_2():\n    helper_b()",
    }
    file_entity_names = {"f1.py": ["fn_1"], "f2.py": ["fn_2"]}
    name_to_target_file = {
        "helper_a": "original.py",
        "helper_b": "original.py",
        "fn_1": "f1.py",
        "fn_2": "f2.py",
    }

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    # Both helpers must land in the same file (f1.py is first in plan order).
    assert name_to_target_file["helper_a"] == name_to_target_file["helper_b"]
    chosen = name_to_target_file["helper_a"]
    assert "helper_a" in file_entity_names[chosen]
    assert "helper_b" in file_entity_names[chosen]
    assert "helper_a" in migrated_names
    assert "helper_b" in migrated_names
    # One synthetic placement covering both (single SCC).
    assert len(synthetic) == 1
    assert set(synthetic[0].group) == {"helper_a", "helper_b"}


def test_extract_shared_helpers_transitive_dep_already_wanted():
    # helper_a is directly wanted by f1.py; helper_b is directly wanted by f2.py.
    # helper_a's source also references helper_b (transitive), but helper_b is
    # already in wanted → the "dep_name in wanted" branch prevents re-adding it.
    e_a = Entity(EntityKind.FUNCTION, "helper_a", 1, 2, ["helper_a"])
    e_b = Entity(EntityKind.FUNCTION, "helper_b", 3, 4, ["helper_b"])
    e_fn1 = Entity(EntityKind.FUNCTION, "fn_1", 6, 7, ["fn_1"])
    e_fn2 = Entity(EntityKind.FUNCTION, "fn_2", 9, 10, ["fn_2"])
    classified, migrated_names = _make_classified(
        [e_a, e_b, e_fn1, e_fn2], ["fn_1", "fn_2"]
    )
    entity_map = {"helper_a": e_a, "helper_b": e_b, "fn_1": e_fn1, "fn_2": e_fn2}
    entity_source_map = {
        "helper_a": "def helper_a():\n    helper_b()",
        "helper_b": "def helper_b():\n    pass",
        "fn_1": "def fn_1():\n    helper_a()",
        "fn_2": "def fn_2():\n    helper_b()",
    }
    file_entity_names = {"f1.py": ["fn_1"], "f2.py": ["fn_2"]}
    name_to_target_file = {
        "helper_a": "original.py",
        "helper_b": "original.py",
        "fn_1": "f1.py",
        "fn_2": "f2.py",
    }

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    # Both helpers are extracted (as separate SCCs since no mutual cycle in graph).
    assert "helper_a" in migrated_names
    assert "helper_b" in migrated_names
    # Two synthetic placements — one for each singleton SCC.
    assert len(synthetic) == 2


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
