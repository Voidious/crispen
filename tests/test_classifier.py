"""Tests for file_limiter.classifier — 100% branch coverage."""

from __future__ import annotations

import textwrap

from crispen.file_limiter.classifier import (
    EntityClass,
    _assign_sccs,
    _entity_overlaps_diff,
    _is_import_only_entity,
    classify_entities,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _func_entity(name: str, start: int, end: int) -> Entity:
    return Entity(EntityKind.FUNCTION, name, start, end, [name])


# ---------------------------------------------------------------------------
# _entity_overlaps_diff
# ---------------------------------------------------------------------------


def test_overlaps_diff_empty_ranges():
    entity = _func_entity("foo", 5, 10)
    assert _entity_overlaps_diff(entity, []) is False


def test_overlaps_diff_true():
    entity = _func_entity("foo", 5, 10)
    assert _entity_overlaps_diff(entity, [(8, 12)]) is True


def test_overlaps_diff_entity_before_range():
    # entity.start_line > end → first condition False (short-circuit)
    entity = _func_entity("foo", 10, 20)
    assert _entity_overlaps_diff(entity, [(1, 5)]) is False


def test_overlaps_diff_entity_after_range_start():
    # entity.start_line <= end but entity.end_line < start
    entity = _func_entity("foo", 1, 5)
    assert _entity_overlaps_diff(entity, [(10, 20)]) is False


def test_overlaps_diff_second_range_matches():
    # First range misses, second range overlaps → True
    entity = _func_entity("foo", 10, 20)
    assert _entity_overlaps_diff(entity, [(1, 5), (15, 25)]) is True


def test_overlaps_diff_adjacent_exactly():
    entity = _func_entity("foo", 5, 10)
    assert _entity_overlaps_diff(entity, [(10, 15)]) is True
    assert _entity_overlaps_diff(entity, [(1, 5)]) is True


# ---------------------------------------------------------------------------
# _assign_sccs
# ---------------------------------------------------------------------------


def test_assign_sccs_empty():
    set_1, set_2, set_3 = _assign_sccs([], {})
    assert set_1 == []
    assert set_2 == []
    assert set_3 == []


def test_assign_sccs_unmodified_to_set1():
    ec = {"a": EntityClass.UNMODIFIED}
    set_1, set_2, set_3 = _assign_sccs([["a"]], ec)
    assert "a" in set_1
    assert set_2 == []
    assert set_3 == []


def test_assign_sccs_new_to_set2():
    ec = {"a": EntityClass.NEW}
    set_1, set_2, set_3 = _assign_sccs([["a"]], ec)
    assert set_1 == []
    assert ["a"] in set_2
    assert set_3 == []


def test_assign_sccs_modified_to_set3():
    ec = {"a": EntityClass.MODIFIED}
    set_1, set_2, set_3 = _assign_sccs([["a"]], ec)
    assert set_1 == []
    assert set_2 == []
    assert ["a"] in set_3


def test_assign_sccs_new_plus_modified_to_set2():
    # SCC with NEW + MODIFIED: NEW takes over → set_2
    ec = {"new": EntityClass.NEW, "mod": EntityClass.MODIFIED}
    set_1, set_2, set_3 = _assign_sccs([["new", "mod"]], ec)
    assert set_1 == []
    scc_set = set(set_2[0])
    assert scc_set == {"new", "mod"}
    assert set_3 == []


def test_assign_sccs_unmodified_plus_new_to_set1():
    # UNMODIFIED takes priority over NEW → entire SCC stays in set_1
    ec = {"old": EntityClass.UNMODIFIED, "new": EntityClass.NEW}
    set_1, set_2, set_3 = _assign_sccs([["old", "new"]], ec)
    assert set(set_1) == {"old", "new"}
    assert set_2 == []
    assert set_3 == []


def test_assign_sccs_multiple_sccs_all_types():
    ec = {
        "unmod": EntityClass.UNMODIFIED,
        "new_fn": EntityClass.NEW,
        "mod_fn": EntityClass.MODIFIED,
    }
    sccs = [["unmod"], ["new_fn"], ["mod_fn"]]
    set_1, set_2, set_3 = _assign_sccs(sccs, ec)
    assert "unmod" in set_1
    assert ["new_fn"] in set_2
    assert ["mod_fn"] in set_3


# ---------------------------------------------------------------------------
# classify_entities
# ---------------------------------------------------------------------------


def test_classify_empty_sources():
    result = classify_entities("", "", [])
    assert result.entities == []
    assert result.entity_class == {}
    assert result.set_1 == []
    assert result.set_2_groups == []
    assert result.set_3_groups == []
    assert result.abort is False


def test_classify_single_entity_no_abort():
    source = "def foo():\n    pass\n"
    result = classify_entities(source, source, [])
    assert result.abort is False  # single entity never triggers abort


def test_classify_abort_all_one_scc():
    # a calls b, b calls a → mutual cycle → one SCC → abort
    source = "def a():\n    b()\n\ndef b():\n    a()\n"
    # Both lines in diff so both are MODIFIED
    result = classify_entities("", source, [(1, 5)])
    assert len(result.entities) == 2
    assert result.abort is True


def test_classify_no_abort_multiple_sccs():
    # a calls b but b does not call a → two separate SCCs
    source = "def a():\n    b()\n\ndef b():\n    pass\n"
    result = classify_entities("", source, [(1, 5)])
    assert result.abort is False
    assert len(result.set_2_groups) >= 1  # new entities go to set_2


def test_classify_all_unmodified():
    source = "def foo():\n    pass\ndef bar():\n    pass\n"
    result = classify_entities(source, source, [])
    assert result.entity_class["foo"] == EntityClass.UNMODIFIED
    assert result.entity_class["bar"] == EntityClass.UNMODIFIED
    assert set(result.set_1) == {"foo", "bar"}
    assert result.set_2_groups == []
    assert result.set_3_groups == []


def test_classify_all_new():
    post = "def foo():\n    pass\n"
    result = classify_entities("", post, [(1, 2)])
    assert result.entity_class["foo"] == EntityClass.NEW
    assert result.set_1 == []
    # foo is a singleton SCC → goes to set_2_groups
    assert len(result.set_2_groups) == 1
    assert result.set_2_groups[0] == ["foo"]


def test_classify_modified():
    original = "def foo():\n    pass\n"
    post = "def foo():\n    return 1\n"
    result = classify_entities(original, post, [(2, 2)])
    assert result.entity_class["foo"] == EntityClass.MODIFIED
    # Modified singleton SCC → set_3_groups
    assert len(result.set_3_groups) == 1
    assert result.set_3_groups[0] == ["foo"]


def test_classify_unmodified_entity_not_in_diff():
    original = "def foo():\n    pass\n\ndef bar():\n    pass\n"
    # post is same; diff only touches bar's lines
    post = original
    result = classify_entities(original, post, [(4, 5)])
    # foo's lines (1-2) don't overlap [(4,5)] → UNMODIFIED
    assert result.entity_class["foo"] == EntityClass.UNMODIFIED
    # bar's lines (4-5) overlap [(4,5)] → MODIFIED
    assert result.entity_class["bar"] == EntityClass.MODIFIED


def test_classify_mixed_new_modified_unmodified():
    original = textwrap.dedent(
        """\
        def unchanged():
            pass

        def changed():
            pass
    """
    )
    post = textwrap.dedent(
        """\
        def unchanged():
            pass

        def changed():
            return 42

        def brand_new():
            pass
    """
    )
    # changed is at lines 4-5 in post; brand_new at lines 7-8
    diff_ranges = [(5, 5), (7, 8)]
    result = classify_entities(original, post, diff_ranges)

    assert result.entity_class["unchanged"] == EntityClass.UNMODIFIED
    assert result.entity_class["changed"] == EntityClass.MODIFIED
    assert result.entity_class["brand_new"] == EntityClass.NEW

    # Each entity is a singleton SCC: unchanged→set_1, changed→set_3, brand_new→set_2.
    assert "unchanged" in result.set_1
    assert any("changed" in g for g in result.set_3_groups)
    assert any("brand_new" in g for g in result.set_2_groups)


def test_classify_new_cycle_both_go_to_set2():
    # Two brand-new functions calling each other — one SCC → set_2
    post = "def a():\n    b()\n\ndef b():\n    a()\n"
    result = classify_entities("", post, [(1, 5)])
    assert result.entity_class["a"] == EntityClass.NEW
    assert result.entity_class["b"] == EntityClass.NEW
    assert result.abort is True  # 2 entities, 1 SCC
    # abort=True but the SCC still gets assigned to set_2 (abort is for the caller)
    all_set2 = {name for g in result.set_2_groups for name in g}
    assert all_set2 == {"a", "b"}


def test_classify_modified_cycle_goes_to_set3():
    original = "def a():\n    b()\n\ndef b():\n    a()\n"
    post = "def a():\n    b()\n\ndef b():\n    a()\n"
    # Both overlap the diff → both MODIFIED; mutual cycle → 1 SCC
    result = classify_entities(original, post, [(1, 5)])
    assert result.entity_class["a"] == EntityClass.MODIFIED
    assert result.entity_class["b"] == EntityClass.MODIFIED
    assert result.abort is True
    all_set3 = {name for g in result.set_3_groups for name in g}
    assert all_set3 == {"a", "b"}


def test_classify_graph_exposed():
    post = "def foo():\n    bar()\n\ndef bar():\n    pass\n"
    result = classify_entities("", post, [(1, 5)])
    # foo depends on bar
    assert "bar" in result.graph.get("foo", set())
    assert result.graph.get("bar") == set()


# ---------------------------------------------------------------------------
# _is_import_only_entity
# ---------------------------------------------------------------------------


def _top_level_entity(name: str, start: int, end: int) -> Entity:
    return Entity(EntityKind.TOP_LEVEL, name, start, end, [])


def test_is_import_only_imports_only():
    src = "import os\nimport sys\n"
    lines = src.splitlines(keepends=True)
    entity = _top_level_entity("_block_1", 1, 2)
    assert _is_import_only_entity(entity, lines) is True


def test_is_import_only_docstring_and_imports():
    src = '"""Module docstring."""\nimport os\nfrom sys import argv\n'
    lines = src.splitlines(keepends=True)
    entity = _top_level_entity("_block_1", 1, 3)
    assert _is_import_only_entity(entity, lines) is True


def test_is_import_only_has_assignment():
    src = "import os\n_X = 1\n"
    lines = src.splitlines(keepends=True)
    entity = _top_level_entity("_block_1", 1, 2)
    assert _is_import_only_entity(entity, lines) is False


def test_is_import_only_syntax_error():
    # Unparseable source → returns False
    src = "def (\n"
    lines = src.splitlines(keepends=True)
    entity = _top_level_entity("_block_1", 1, 1)
    assert _is_import_only_entity(entity, lines) is False


def test_is_import_only_empty():
    # Empty body → True (no non-import statements)
    src = "\n"
    lines = src.splitlines(keepends=True)
    entity = _top_level_entity("_block_1", 1, 1)
    assert _is_import_only_entity(entity, lines) is True


# ---------------------------------------------------------------------------
# classify_entities: import-only block forced to UNMODIFIED
# ---------------------------------------------------------------------------


def test_classify_import_only_block_stays_unmodified_even_when_diff_overlaps():
    # The import block overlaps the diff range but must stay UNMODIFIED.
    post = textwrap.dedent(
        """\
        import os
        import sys

        def foo():
            return os.getcwd()
    """
    )
    # diff covers line 1-2 (the import block)
    result = classify_entities(post, post, [(1, 2)])
    assert result.entity_class["_block_1"] == EntityClass.UNMODIFIED
    assert "_block_1" in result.set_1


def test_classify_import_only_block_with_docstring_stays_unmodified():
    post = textwrap.dedent(
        """\
        \"\"\"Docstring.\"\"\"
        import os

        def foo():
            return os.getcwd()
    """
    )
    result = classify_entities(post, post, [(1, 2)])
    assert result.entity_class["_block_1"] == EntityClass.UNMODIFIED
    assert "_block_1" in result.set_1


def test_classify_mixed_block_not_forced_unmodified():
    # Block with imports AND an assignment is not import-only → stays MODIFIED.
    post = textwrap.dedent(
        """\
        import os
        _X = 1

        def foo():
            return _X
    """
    )
    result = classify_entities(post, post, [(1, 2)])
    assert result.entity_class["_block_1"] == EntityClass.MODIFIED
