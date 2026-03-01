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
    _prune_unused_imports,
    _relative_import_prefix,
    _remove_entity_lines,
    _target_module_name,
    _topo_depth,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .test_add_re_exports import (
    test_add_re_exports_all_private_no_change,
    test_add_re_exports_from_import_line,
    test_add_re_exports_mixed_public_private,
    test_add_re_exports_multiple_targets_sorted,
    test_add_re_exports_no_import_in_source,
    test_add_re_exports_private_referenced_in_source,
    test_add_re_exports_public_inserted_after_imports,
)
from .test_collect_name_loads import (
    test_collect_name_loads_basic,
    test_collect_name_loads_store_not_included,
    test_collect_name_loads_syntax_error,
)
from .test_extract_import_info import (
    test_extract_import_info_dotted_import,
    test_extract_import_info_from_import,
    test_extract_import_info_from_import_with_asname,
    test_extract_import_info_future_import,
    test_extract_import_info_import_with_asname,
    test_extract_import_info_multiple,
    test_extract_import_info_plain_import,
    test_extract_import_info_skips_non_imports,
    test_extract_import_info_syntax_error,
)
from .test_find_cross_file_imports import (
    test_find_cross_file_imports_basic,
    test_find_cross_file_imports_cross_directory,
    test_find_cross_file_imports_entity_not_in_map,
    test_find_cross_file_imports_no_match,
    test_find_cross_file_imports_same_file_excluded,
)
from .test_find_needed_imports import (
    test_find_needed_imports_deduplicates,
    test_find_needed_imports_entity_not_in_map,
    test_find_needed_imports_future_always_included,
    test_find_needed_imports_referenced_name,
    test_find_needed_imports_unreferenced_name,
)
from .test_generate import (
    test_generate_abort_plan,
    test_generate_empty_placements,
    test_generate_entity_not_in_source_map,
    test_generate_future_import_always_included,
    test_generate_future_import_not_duplicated_when_in_entity_source,
    test_generate_multiple_different_target_files,
    test_generate_multiple_groups_same_file,
    test_generate_no_imports_needed,
    test_generate_private_entity_no_reexport,
    test_generate_single_entity_migration,
)
from .test_generate_cross_file_import import (
    _classified,
    _make_entity,
    _plan,
    test_generate_aborts_on_cross_file_import_cycle,
    test_generate_all_placements_self_referential,
    test_generate_cross_file_import,
    test_generate_non_migrated_helper_extracted_to_new_file,
    test_generate_self_referential_placement_dropped,
)
from .test_prune_unused_imports import (
    test_generate_prunes_unused_names_from_multiname_import,
    test_prune_unused_imports_multiline_import_collapsed,
    test_prune_unused_imports_narrows_partial_from_import,
    test_prune_unused_imports_narrows_plain_import,
    test_prune_unused_imports_no_replacements_needed,
    test_prune_unused_imports_preserves_future_import,
    test_prune_unused_imports_preserves_star_import,
    test_prune_unused_imports_relative_import_narrowed,
    test_prune_unused_imports_removes_fully_unused_from_import,
    test_prune_unused_imports_removes_fully_unused_plain_import,
    test_prune_unused_imports_syntax_error,
)
from .test_relative_import_prefix import (
    test_relative_import_prefix_same_directory,
    test_relative_import_prefix_same_subdir,
    test_relative_import_prefix_sibling_subdir,
    test_relative_import_prefix_to_nested,
)
from .test_remove_entity_lines import (
    test_remove_entity_lines_name_not_in_map,
    test_remove_entity_lines_removes_range,
)
from .test_target_module_name import (
    test_target_module_name_nested,
    test_target_module_name_simple,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _collect_name_loads
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _extract_import_info
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _find_needed_imports
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _target_module_name
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _remove_entity_lines
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _add_re_exports
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# generate_file_splits
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _find_cross_file_imports
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _relative_import_prefix
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# generate_file_splits — cross-file import integration
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _topo_depth
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _extract_shared_helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _prune_unused_imports
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# generate_file_splits — import pruning integration
# ---------------------------------------------------------------------------
