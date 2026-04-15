from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import (
    _bump_relative_imports,
    _extract_shared_helpers,
    _find_main_block_entity,
    _find_main_direct_callees,
    _find_project_root,
    _is_test_name,
    _merge_conftest_sources,
    _module_path_from_file,
    _multiline_string_ranges,
    _normalize_blank_lines,
    _remove_entity_lines,
    _source_is_only_docstring,
    _split_cross_imports_by_test,
    _strip_orphaned_indented_comments,
    _strip_orphaned_section_headers,
    _sub_skip_strings,
    _topo_depth,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .helpers import _classified, _make_classified, _make_entity, _plan


def test_remove_entity_lines_removes_range():
    source = "line1\nline2\nline3\nline4\n"
    entity = _make_entity("foo", 2, 3)
    entity_map = {"foo": entity}
    result = _remove_entity_lines(source, {"foo"}, entity_map, {})
    assert "line1" in result
    assert "line2" not in result
    assert "line3" not in result
    assert "line4" in result


def test_remove_entity_lines_name_not_in_map():
    # Name not in entity_map → nothing removed.
    source = "line1\nline2\n"
    result = _remove_entity_lines(source, {"ghost"}, {}, {})
    assert result == source


def test_remove_entity_lines_top_level_preserves_import_lines():
    # When a TOP_LEVEL entity containing both imports and assignments is
    # migrated, the import lines must be kept in the original file so that
    # the remaining functions still have access to those names.
    source = "import os\n_CONST = 1\n\ndef foo():\n    return os.getcwd()\n"
    entity_src = "import os\n_CONST = 1\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 2, ["os", "_CONST"])
    entity_map = {"_block_1": entity}
    entity_source_map = {"_block_1": entity_src}
    result = _remove_entity_lines(source, {"_block_1"}, entity_map, entity_source_map)
    assert "import os" in result  # import line preserved
    assert "_CONST" not in result  # assignment line removed
    assert "def foo():" in result  # function untouched


def test_remove_entity_lines_top_level_no_source_map_removes_all():
    # Empty entity_source_map → no imports can be identified, all lines removed.
    source = "import os\n_CONST = 1\n\ndef foo():\n    pass\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 2, ["os", "_CONST"])
    entity_map = {"_block_1": entity}
    result = _remove_entity_lines(source, {"_block_1"}, entity_map, {})
    assert "import os" not in result
    assert "_CONST" not in result


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


def test_find_project_root_finds_pyproject_toml(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    sub = tmp_path / "pkg" / "module.py"
    sub.parent.mkdir()
    sub.write_text("x = 1\n")
    assert _find_project_root(sub) == tmp_path


def test_find_project_root_finds_git(tmp_path):
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "module.py"
    sub.write_text("x = 1\n")
    assert _find_project_root(sub) == tmp_path


def test_find_project_root_called_with_directory(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    assert _find_project_root(tmp_path) == tmp_path


def test_find_project_root_not_found(tmp_path):
    # tmp_path is under /tmp which has no project markers → None.
    sub = tmp_path / "module.py"
    sub.write_text("x = 1\n")
    result = _find_project_root(sub)
    # If the test runner is inside a project that happens to include tmp_path
    # (unlikely but possible with in-tree pytest), just ensure the function
    # returns without crashing.  The important coverage is the happy path above.
    assert result is None or result.exists()


def test_module_path_from_file_success(tmp_path):
    f = tmp_path / "pkg" / "utils.py"
    f.parent.mkdir()
    f.write_text("")
    assert _module_path_from_file(tmp_path, f) == "pkg.utils"


def test_module_path_from_file_top_level(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("")
    assert _module_path_from_file(tmp_path, f) == "module"


def test_module_path_from_file_not_under_root(tmp_path):
    other = tmp_path.parent / "other.py"
    assert _module_path_from_file(tmp_path, other) is None


def test_bump_relative_imports_single_dot():
    assert _bump_relative_imports("from .foo import bar") == "from ..foo import bar"


def test_bump_relative_imports_two_dots():
    assert _bump_relative_imports("from .. import baz") == "from ... import baz"


def test_bump_relative_imports_leaves_absolute():
    src = "import os\nfrom typing import List"
    assert _bump_relative_imports(src) == src


def test_bump_relative_imports_multiline():
    src = "from .a import x\nimport sys\nfrom ..b import y\n"
    result = _bump_relative_imports(src)
    assert "from ..a import x" in result
    assert "from ...b import y" in result
    assert "import sys" in result


def test_bump_relative_imports_n_two():
    assert _bump_relative_imports("from .. import foo", n=2) == "from .... import foo"


def test_bump_relative_imports_n_zero():
    src = "from .foo import bar"
    assert _bump_relative_imports(src, n=0) == src


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


def test_generate_file_splits_subdir_injects_tc_import_for_nonmigrated_entity():
    # When a _block_N TOP_LEVEL entity that holds the `if TYPE_CHECKING:` block
    # is migrated to a sub-file, any non-migrated entity that references the
    # guarded name in a quoted annotation must receive the TYPE_CHECKING import
    # in the updated original (__init__.py).
    #
    # The original file has three entities:
    #   _block_1 — the TYPE_CHECKING block (migrated to sub.py)
    #   helper   — migrated to sub.py
    #   entry    — stays in __init__.py, references "MyConfig" in annotation
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .config import MyConfig\n"
        "\n"
        "def helper():\n"
        "    pass\n"
        "\n"
        "def entry(cfg: 'MyConfig') -> None:\n"
        "    helper()\n"
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, [])
    e_helper = _make_entity("helper", 5, 6)
    e_entry = _make_entity("entry", 8, 9)
    c = _classified(entities=[e_block, e_helper, e_entry])
    plan = _plan(
        [GroupPlacement(group=["_block_1", "helper"], target_file="pkg/sub.py")]
    )

    result = generate_file_splits(c, plan, source, "pkg.py", subdir_name="pkg")

    assert not result.abort
    init_src = result.original_source
    # The TYPE_CHECKING import must be injected and bumped for the new depth.
    assert "if TYPE_CHECKING:" in init_src
    assert "from ..config import MyConfig" in init_src


def test_source_is_only_docstring_true():
    assert _source_is_only_docstring('"""Just a docstring."""\n') is True


def test_source_is_only_docstring_with_other_content():
    assert _source_is_only_docstring('"""Doc."""\n\nimport os\n') is False


def test_source_is_only_docstring_no_docstring():
    assert _source_is_only_docstring("import os\n") is False


def test_source_is_only_docstring_syntax_error():
    assert _source_is_only_docstring("def (\n") is False


def test_is_test_name_test_class():
    assert _is_test_name("TestFoo") is True


def test_is_test_name_test_function():
    assert _is_test_name("test_bar") is True


def test_is_test_name_non_test():
    assert _is_test_name("helper") is False
    assert _is_test_name("Foo") is False
    assert _is_test_name("_test_private") is False


def test_split_cross_imports_by_test_pure_non_test():
    non_test, test_named = _split_cross_imports_by_test(["from .foo import helper"])
    assert non_test == ["from .foo import helper"]
    assert test_named == []


def test_split_cross_imports_by_test_pure_test():
    non_test, test_named = _split_cross_imports_by_test(
        ["from .foo import TestFoo, test_bar"]
    )
    assert non_test == []
    assert test_named == ["from .foo import TestFoo, test_bar"]


def test_split_cross_imports_by_test_mixed():
    non_test, test_named = _split_cross_imports_by_test(
        ["from .foo import TestFoo, helper, test_bar"]
    )
    assert non_test == ["from .foo import helper"]
    assert test_named == ["from .foo import TestFoo, test_bar"]


def test_split_cross_imports_by_test_plain_import_passthrough():
    # Plain "import x" lines (no "from") pass through to non_test unchanged.
    non_test, test_named = _split_cross_imports_by_test(["import os"])
    assert non_test == ["import os"]
    assert test_named == []


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


def test_generate_pytest_conftest_disabled_no_conftest():
    # Default (pytest_conftest=False): fixture goes to assigned file, re-exported.
    src = "@pytest.fixture\ndef client():\n    pass\n"
    entity = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["client"], target_file="fixtures.py")])

    result = generate_file_splits(c, plan, src, "test_big.py")

    assert "fixtures.py" in result.new_files
    assert "conftest.py" not in result.new_files
    assert "client" in result.new_files["fixtures.py"]


def test_generate_pytest_conftest_subdir_routes_to_subdir_conftest():
    # With pytest_conftest=True AND subdir_name set, fixtures go to
    # <subdir>/conftest.py (not the parent conftest.py).  This prevents
    # multiple test files in the same directory from conflicting when they
    # each have a fixture of the same name.
    src = "@pytest.fixture\ndef client():\n    pass\n"
    entity = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["client"], target_file="expr/fixtures.py")])

    result = generate_file_splits(
        c, plan, src, "test_big.py", subdir_name="expr", pytest_conftest=True
    )

    assert "expr/conftest.py" in result.new_files
    assert "def client():" in result.new_files["expr/conftest.py"]
    assert "conftest.py" not in result.new_files  # parent conftest untouched
    assert "import client" not in result.original_source


def test_generate_pytest_conftest_subdir_fixture_referenced_in_remaining_goes_to_parent():  # noqa: E501
    # When a fixture is migrated from a subdir split but its name still appears
    # in entities that remain in the original file, route it to the parent
    # conftest.py (not the subdir conftest) so those tests can find it.
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        def test_big(client):
            pass
        """
    )
    e_client = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    e_test = Entity(EntityKind.FUNCTION, "test_big", 5, 6, ["test_big"])
    # Only the fixture is migrated; the test stays in the original.
    c = _classified(entities=[e_client, e_test])
    plan = _plan([GroupPlacement(group=["client"], target_file="expr/fixtures.py")])

    result = generate_file_splits(
        c, plan, src, "test_big.py", subdir_name="expr", pytest_conftest=True
    )

    # Fixture goes to parent conftest.py, not the subdir one.
    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]
    assert "expr/conftest.py" not in result.new_files
    # No import of client back into the original.
    assert "import client" not in result.original_source


def test_generate_pytest_conftest_subdir_fixture_overrides_parent_conftest(tmp_path):
    # When the fixture is referenced in remaining source AND the parent conftest
    # already has a fixture with the same name (the module was overriding it),
    # the fixture is *copied* (not moved) to the subdir conftest so migrated
    # tests get the override; the entity also stays in the original file so
    # the original test discovers it from its own module.
    parent_conftest = tmp_path / "conftest.py"
    parent_conftest.write_text(
        "@pytest.fixture\ndef client():\n    return 'base'\n", encoding="utf-8"
    )
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            return 'override'

        def test_big(client):
            pass
        """
    )
    e_client = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    e_test = Entity(EntityKind.FUNCTION, "test_big", 5, 6, ["test_big"])
    c = _classified(entities=[e_client, e_test])
    plan = _plan([GroupPlacement(group=["client"], target_file="expr/fixtures.py")])
    original_path = str(tmp_path / "test_big.py")

    result = generate_file_splits(
        c, plan, src, original_path, subdir_name="expr", pytest_conftest=True
    )

    # Fixture goes to subdir conftest for migrated tests.
    assert "expr/conftest.py" in result.new_files
    assert "def client():" in result.new_files["expr/conftest.py"]
    assert "return 'override'" in result.new_files["expr/conftest.py"]
    # Parent conftest is NOT modified (would drop the override via merge).
    assert "conftest.py" not in result.new_files
    # Fixture stays in original file so the original test finds the override.
    assert "def client():" in result.original_source
    assert "return 'override'" in result.original_source
    # No re-export import injected.
    assert "import client" not in result.original_source


def test_merge_conftest_sources_deduplicates_imports():
    # Imports that already exist are not repeated.
    existing = "import pytest\n\n\n@pytest.fixture\ndef prior():\n    pass\n"
    new = "import pytest\n\n\n@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert result.count("import pytest") == 1


def test_merge_conftest_sources_deduplicates_functions():
    # A function already in existing is not appended again.
    existing = "@pytest.fixture\ndef client():\n    return 1\n"
    new = "@pytest.fixture\ndef client():\n    return 2\n"
    result = _merge_conftest_sources(existing, new)
    assert result.count("def client():") == 1
    assert "return 1" in result
    assert "return 2" not in result


def test_merge_conftest_sources_appends_new_fixture():
    # A new fixture not in existing is appended.
    existing = "@pytest.fixture\ndef prior():\n    pass\n"
    new = "@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert "def prior():" in result
    assert "def client():" in result
    assert result.index("prior") < result.index("client")


def test_merge_conftest_sources_no_changes_returns_existing():
    # When nothing new to add, return existing unchanged.
    existing = "import pytest\n\n\n@pytest.fixture\ndef client():\n    pass\n"
    new = "import pytest\n\n\n@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert result == existing


def test_merge_conftest_sources_inserts_new_imports_before_functions():
    # New imports are inserted after existing imports but before functions — no E402.
    existing = "import pytest\n\n\n@pytest.fixture\ndef prior():\n    pass\n"
    new = "import asyncio\n\n\n@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert "import asyncio" in result
    assert "def client():" in result
    # Imports must come before the first function definition.
    assert result.index("import asyncio") < result.index("def prior():")


def test_merge_conftest_sources_syntax_error_fallback():
    # Falls back to simple concatenation when existing cannot be parsed.
    existing = "def (broken"
    new = "import pytest\n"
    result = _merge_conftest_sources(existing, new)
    assert "def (broken" in result
    assert "import pytest" in result


def test_merge_conftest_sources_preserves_comments():
    # Comments in the existing conftest are preserved.
    existing = "# shared fixtures\nimport pytest\n\n\ndef prior():\n    pass\n"
    new = "@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert "# shared fixtures" in result
    assert "def client():" in result


def test_merge_conftest_sources_from_import_dedup():
    # from-style imports are also deduplicated via the _import_key F: path.
    existing = "from conftest import setup\n\n\ndef prior():\n    pass\n"
    new = "from conftest import setup\n\n\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert result.count("from conftest import setup") == 1
    assert "def client():" in result


def test_merge_conftest_sources_only_new_imports_no_defs():
    # When only new imports are added but no new functions, ends with newline.
    existing = "import pytest\n\n\ndef prior():\n    pass\n"
    new = "import asyncio\n"
    result = _merge_conftest_sources(existing, new)
    assert "import asyncio" in result
    assert result.endswith("\n")
    # No duplicate function definition appended.
    assert result.count("def prior():") == 1


def test_merge_conftest_sources_non_import_non_def_in_new():
    # Bare statements (assignments, expressions) in new_content are silently ignored.
    existing = "def prior():\n    pass\n"
    new = "X = 42\n"
    result = _merge_conftest_sources(existing, new)
    # Nothing to import or define → returns existing unchanged.
    assert result == existing


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


def test_normalize_blank_lines_strips_leading_blanks():
    """Leading blank lines are removed (prevents E303 at top of file)."""
    source = "\n\n\ndef foo():\n    pass\n"
    result = _normalize_blank_lines(source)
    assert result.startswith("def foo():")


def test_normalize_blank_lines_collapses_excess_top_level():
    """4+ consecutive newlines between top-level defs collapse to 3."""
    source = "def foo():\n    pass\n\n\n\n\ndef bar():\n    pass\n"
    result = _normalize_blank_lines(source)
    assert "\n\n\n\n" not in result
    assert "def foo():" in result
    assert "def bar():" in result


def test_normalize_blank_lines_collapses_body_blanks():
    """2+ blank lines inside an indented body collapse to 1 (prevents E303 in body)."""
    source = "def foo():\n    x = 1\n\n\n    y = 2\n"
    result = _normalize_blank_lines(source)
    assert "\n\n\n    y" not in result
    assert "\n\n    y" in result


def test_normalize_blank_lines_empty_source():
    """Whitespace-only source returns empty string."""
    assert _normalize_blank_lines("\n\n\n") == ""


def test_normalize_blank_lines_trailing_newline():
    """Result always ends with exactly one newline."""
    source = "x = 1\n\n\n"
    result = _normalize_blank_lines(source)
    assert result.endswith("\n")
    assert not result.endswith("\n\n")


def test_normalize_blank_lines_preserves_multiline_string_body_blanks():
    """Blank lines inside a multi-line string literal are never collapsed.

    Regression: _EXCESS_BLANK_BODY_RE matched \\n{3,}(?=[ \\t]) inside
    triple-quoted strings, collapsing 2 blank lines before an indented line
    to 1 (e.g. stored source-code fixtures in tests).
    """
    # The triple-quoted string contains 2 blank lines before an indented `def`.
    # That produces the sequence \\n\\n\\n        def inside the raw source,
    # which _EXCESS_BLANK_BODY_RE would collapse to \\n\\n        def.
    source = textwrap.dedent(
        """\
        import textwrap
        def foo():
            src = textwrap.dedent(
                \"\"\"\\
                @dataclass
                class _SplitTask:
                    pass


                def _find_free_vars():
                    x = 1
                \"\"\"
            )
        """
    )
    result = _normalize_blank_lines(source)
    # Two blank lines before the indented `def` inside the string must survive.
    # After outer textwrap.dedent the string content has 8-space indentation.
    assert "\n\n\n        def _find_free_vars" in result


def test_normalize_blank_lines_still_collapses_excess_outside_strings():
    """Blank-line collapsing still fires for code outside string literals."""
    source = "def foo():\n    x = 1\n\n\n    y = 2\n"
    result = _normalize_blank_lines(source)
    assert "\n\n\n    y" not in result
    assert "\n\n    y" in result


def test_multiline_string_ranges_triple_quoted():
    """Detects a triple-quoted string spanning multiple lines."""
    source = 'x = """\nhello\n"""\n'
    ranges = _multiline_string_ranges(source)
    assert len(ranges) == 1
    start, end = ranges[0]
    assert source[start:end] == '"""\nhello\n"""'


def test_multiline_string_ranges_single_line_string_ignored():
    """Single-line strings (no literal newline) are not returned."""
    source = 'x = "hello\\n"\n'
    ranges = _multiline_string_ranges(source)
    assert ranges == []


def test_multiline_string_ranges_no_strings():
    """Returns empty list when there are no string literals."""
    source = "x = 1 + 2\n"
    ranges = _multiline_string_ranges(source)
    assert ranges == []


def test_multiline_string_ranges_invalid_source():
    """Falls back to empty list on tokenization error."""
    # Unterminated string triggers TokenError.
    source = 'x = """\nhello\n'
    ranges = _multiline_string_ranges(source)
    assert ranges == []


def test_sub_skip_strings_does_not_touch_string_content():
    """Pattern match inside a multi-line string is not substituted."""
    import re

    pattern = re.compile(r"\n{3,}(?=[ \t])")
    source = 'def f():\n    s = """\n    a\n\n\n    b\n    """\n'
    result = _sub_skip_strings(pattern, "\n\n", source)
    # The sequence inside the string must survive unchanged.
    assert "\n\n\n    b" in result


def test_sub_skip_strings_applies_outside_strings():
    """Pattern match outside string literals is substituted normally."""
    import re

    pattern = re.compile(r"\n{3,}(?=[ \t])")
    source = "def f():\n    x = 1\n\n\n    y = 2\n"
    result = _sub_skip_strings(pattern, "\n\n", source)
    assert "\n\n\n    y" not in result
    assert "\n\n    y" in result


def test_sub_skip_strings_no_strings_falls_through():
    """When there are no multi-line strings the plain .sub() path is taken."""
    import re

    pattern = re.compile(r"x")
    source = "x = 1\n"
    result = _sub_skip_strings(pattern, "y", source)
    assert result == "y = 1\n"


def test_strip_orphaned_indented_comments_removes_orphan():
    """Indented comment at module level (outside any AST node) is removed."""
    source = "\n\n    # This comment was left behind after function removal\n"
    result = _strip_orphaned_indented_comments(source)
    assert "# This comment was left behind" not in result


def test_strip_orphaned_indented_comments_keeps_inside_function():
    """Indented comment inside a function body is preserved."""
    source = "def foo():\n    # normal comment\n    pass\n"
    result = _strip_orphaned_indented_comments(source)
    assert "# normal comment" in result


def test_strip_orphaned_indented_comments_keeps_module_level_comment():
    """Non-indented module-level comment is preserved."""
    source = "# module comment\ndef foo():\n    pass\n"
    result = _strip_orphaned_indented_comments(source)
    assert "# module comment" in result


def test_strip_orphaned_indented_comments_syntax_error():
    """SyntaxError in source returns source unchanged."""
    source = "    # orphaned\ndef f(: pass\n"
    result = _strip_orphaned_indented_comments(source)
    assert result == source
