from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import (
    _extract_shared_helpers,
    _find_cross_file_imports,
    _merge_from_imports,
    _topo_depth,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .test_basic_generate import _classified, _make_classified, _make_entity, _plan
import textwrap


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


def test_find_cross_file_imports_cross_directory():
    # fn_a is in tests/test.py; helper is in helpers/entities.py.
    # Cross-directory import needs ".." to go up from tests/ to root.
    entity_source_map = {"fn_a": "def fn_a():\n    return _helper()\n"}
    name_to_target_file = {"_helper": "helpers/entities.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "tests/test.py"
    )
    assert result == ["from ..helpers.entities import _helper"]


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


def test_topo_depth_empty():
    assert _topo_depth({}) == {}


def test_topo_depth_dag():
    # Linear chain: a → b → c.  c is the leaf (depth 0), b has depth 1, a depth 2.
    # The outer loop visits a first, which recurses into b then c, memoising both.
    # When the outer loop reaches b and c they are already in depths (True branch).
    graph = {"a": {"b"}, "b": {"c"}, "c": set()}
    assert _topo_depth(graph) == {"a": 2, "b": 1, "c": 0}


def test_topo_depth_cycle():
    graph = {"a": {"b"}, "b": {"a"}}
    assert _topo_depth(graph) == {"a": 0, "b": 0}


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
    # helper_a's source also references helper_b (transitive), so helper_b's
    # wanting-set grows from {f2.py} to {f1.py, f2.py} — True branch of the
    # transitive update condition.
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


def test_extract_shared_helpers_transitive_dep_no_new_targets():
    # fn_1 directly references both helper_a and helper_b.
    # helper_a's source also references helper_b (transitive dep).
    # When the transitive loop processes helper_a, helper_b already has the same
    # wanting-set {f1.py} → new_targets is empty → False branch of update condition.
    e_a = Entity(EntityKind.FUNCTION, "helper_a", 1, 2, ["helper_a"])
    e_b = Entity(EntityKind.FUNCTION, "helper_b", 3, 4, ["helper_b"])
    e_fn = Entity(EntityKind.FUNCTION, "fn_1", 6, 7, ["fn_1"])
    classified, migrated_names = _make_classified([e_a, e_b, e_fn], ["fn_1"])
    entity_map = {"helper_a": e_a, "helper_b": e_b, "fn_1": e_fn}
    entity_source_map = {
        "helper_a": "def helper_a():\n    helper_b()",
        "helper_b": "def helper_b():\n    pass",
        "fn_1": "def fn_1():\n    helper_a()\n    helper_b()",
    }
    file_entity_names = {"f1.py": ["fn_1"]}
    name_to_target_file = {
        "helper_a": "original.py",
        "helper_b": "original.py",
        "fn_1": "f1.py",
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

    # Both helpers are still extracted; the transitive dep on helper_b is a no-op
    # because helper_b already has {f1.py} in its wanting-set (direct want).
    assert "helper_a" in migrated_names
    assert "helper_b" in migrated_names
    assert len(synthetic) == 2


def test_extract_shared_helpers_avoids_cycle_by_choosing_downstream_file():
    # _run is wanted by both test_skip.py and test_transformers.py.
    # test_skip.py already imports from test_transformers.py (_RaisingTransformer).
    # Placing _run in test_skip.py would force test_transformers.py to import from
    # test_skip.py → cycle.  The cycle-aware logic must pick test_transformers.py
    # (the downstream file) instead.
    e_raise = Entity(
        EntityKind.FUNCTION, "_RaisingTransformer", 1, 3, ["_RaisingTransformer"]
    )
    e_run = Entity(EntityKind.FUNCTION, "_run", 4, 5, ["_run"])
    e_skip = Entity(EntityKind.FUNCTION, "fn_skip", 7, 9, ["fn_skip"])
    e_transform = Entity(EntityKind.FUNCTION, "fn_transform", 11, 13, ["fn_transform"])
    classified, migrated_names = _make_classified(
        [e_raise, e_run, e_skip, e_transform],
        ["fn_skip", "fn_transform", "_RaisingTransformer"],
    )
    entity_map = {
        "_RaisingTransformer": e_raise,
        "_run": e_run,
        "fn_skip": e_skip,
        "fn_transform": e_transform,
    }
    entity_source_map = {
        "_RaisingTransformer": "def _RaisingTransformer():\n    pass",
        "_run": "def _run(x):\n    return x",
        # fn_skip refs _RaisingTransformer (migrated to test_transformers.py) AND
        # _run (non-migrated) → _run is wanted by test_skip.py.
        "fn_skip": "def fn_skip():\n    _RaisingTransformer()\n    _run(1)",
        # fn_transform also refs _run → _run is wanted by test_transformers.py too.
        "fn_transform": "def fn_transform():\n    _run(2)",
    }
    file_entity_names = {
        "test_skip.py": ["fn_skip"],
        "test_transformers.py": ["fn_transform", "_RaisingTransformer"],
    }
    name_to_target_file = {
        "_RaisingTransformer": "test_transformers.py",
        "_run": "original.py",
        "fn_skip": "test_skip.py",
        "fn_transform": "test_transformers.py",
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

    # _run must go to test_transformers.py, not test_skip.py.
    assert name_to_target_file["_run"] == "test_transformers.py"
    assert "_run" in file_entity_names["test_transformers.py"]
    assert "_run" not in file_entity_names["test_skip.py"]
    assert "_run" in migrated_names
    assert len(synthetic) == 1
    assert synthetic[0].group == ["_run"]
    assert synthetic[0].target_file == "test_transformers.py"


def test_extract_shared_helpers_skips_scc_when_no_cycle_free_placement():
    # fn_1 (in f1.py) refs fn_2 (in f2.py) and fn_2 refs fn_1 → pre-existing
    # cycle in file_deps.  fn_1 also refs helper_h (non-migrated), which itself
    # refs fn_2.  The only candidate for helper_h is f1.py; placing it there
    # would still result in a cycle (f1.py→f2.py→f1.py already exists).
    # Since no cycle-free placement exists, the SCC is skipped entirely.
    e_fn1 = Entity(EntityKind.FUNCTION, "fn_1", 1, 2, ["fn_1"])
    e_fn2 = Entity(EntityKind.FUNCTION, "fn_2", 4, 5, ["fn_2"])
    e_h = Entity(EntityKind.FUNCTION, "helper_h", 7, 8, ["helper_h"])
    classified, migrated_names = _make_classified([e_fn1, e_fn2, e_h], ["fn_1", "fn_2"])
    entity_map = {"fn_1": e_fn1, "fn_2": e_fn2, "helper_h": e_h}
    entity_source_map = {
        "fn_1": "def fn_1():\n    fn_2()\n    helper_h()",
        "fn_2": "def fn_2():\n    fn_1()",
        "helper_h": "def helper_h():\n    fn_2()",
    }
    file_entity_names = {"f1.py": ["fn_1"], "f2.py": ["fn_2"]}
    name_to_target_file = {
        "fn_1": "f1.py",
        "fn_2": "f2.py",
        "helper_h": "original.py",
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

    # helper_h is skipped — no placement avoids the pre-existing cycle.
    assert "helper_h" not in migrated_names
    assert synthetic == []


def test_extract_shared_helpers_helper_refs_migrated_entity_in_other_file():
    # helper_a (non-migrated) references fn_2 (migrated to f2.py).
    # When placed in f1.py the trial and apply phases must account for the
    # resulting f1.py → f2.py dependency edge.
    e_fn1 = Entity(EntityKind.FUNCTION, "fn_1", 1, 2, ["fn_1"])
    e_fn2 = Entity(EntityKind.FUNCTION, "fn_2", 4, 5, ["fn_2"])
    e_helper = Entity(EntityKind.FUNCTION, "helper_a", 7, 8, ["helper_a"])
    classified, migrated_names = _make_classified(
        [e_fn1, e_fn2, e_helper], ["fn_1", "fn_2"]
    )
    entity_map = {"fn_1": e_fn1, "fn_2": e_fn2, "helper_a": e_helper}
    entity_source_map = {
        "fn_1": "def fn_1():\n    helper_a()",
        "fn_2": "def fn_2():\n    pass",
        "helper_a": "def helper_a():\n    fn_2()",
    }
    file_entity_names = {"f1.py": ["fn_1"], "f2.py": ["fn_2"]}
    name_to_target_file = {
        "fn_1": "f1.py",
        "fn_2": "f2.py",
        "helper_a": "original.py",
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

    # helper_a is extracted to f1.py; its dep on fn_2 (f2.py) is tracked in
    # both the trial and apply dep-file branches.
    assert "helper_a" in migrated_names
    assert name_to_target_file["helper_a"] == "f1.py"
    assert len(synthetic) == 1
    assert synthetic[0].target_file == "f1.py"


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


def test_generate_cross_import_dedup_across_entities():
    # helper goes to helpers.py; foo and bar both go to workers.py and both
    # reference helper — the cross-file import should appear once (dedup).
    source = textwrap.dedent(
        """\
        def helper():
            pass

        def foo():
            helper()

        def bar():
            helper()
        """
    )
    e_h = Entity(EntityKind.FUNCTION, "helper", 1, 2, ["helper"])
    e_foo = Entity(EntityKind.FUNCTION, "foo", 4, 5, ["foo"])
    e_bar = Entity(EntityKind.FUNCTION, "bar", 7, 8, ["bar"])
    c = _classified(entities=[e_h, e_foo, e_bar])
    plan = _plan(
        [
            GroupPlacement(group=["helper"], target_file="helpers.py"),
            GroupPlacement(group=["foo", "bar"], target_file="workers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    workers = result.new_files["workers.py"]
    # "from .helpers import helper" should appear exactly once.
    assert workers.count("import helper") == 1
