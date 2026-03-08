from __future__ import annotations
from crispen.file_limiter.advisor import (
    GroupPlacement,
    _build_group_mermaid,
    _compute_projected_lines,
    _find_conflicting_placement_indices,
    _group_summary,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .test_plan_core import _classified, _make_entity


def test_find_conflicting_idx_plan_vs_plan():
    """Flat file + subdir with same stem both appear → both indices returned."""
    placements = [
        GroupPlacement(group=["foo"], target_file="utils.py"),
        GroupPlacement(group=["bar"], target_file="utils/io.py"),
        GroupPlacement(group=["baz"], target_file="helpers.py"),
    ]
    idxs = _find_conflicting_placement_indices(placements, frozenset(), frozenset())
    assert idxs == [0, 1]


def test_find_conflicting_idx_file_vs_existing_dir():
    """Flat .py target whose stem matches an existing directory → index returned."""
    placements = [GroupPlacement(group=["foo"], target_file="models.py")]
    idxs = _find_conflicting_placement_indices(
        placements, frozenset(), frozenset({"models"})
    )
    assert idxs == [0]


def test_find_conflicting_idx_subdir_vs_existing_file():
    """Subdir target whose top matches an existing .py file → index returned."""
    placements = [GroupPlacement(group=["bar"], target_file="helpers/io.py")]
    idxs = _find_conflicting_placement_indices(
        placements, frozenset({"helpers.py"}), frozenset()
    )
    assert idxs == [0]


def test_find_conflicting_idx_no_conflict():
    """Clean plan with no conflicts → empty list."""
    placements = [
        GroupPlacement(group=["foo"], target_file="utils.py"),
        GroupPlacement(group=["bar"], target_file="helpers.py"),
    ]
    assert (
        _find_conflicting_placement_indices(placements, frozenset(), frozenset()) == []
    )


def test_group_summary_with_docstring_and_params():
    """Entity with docstring and params → both appear in summary."""
    ent = _make_entity(
        "foo",
        1,
        10,
        docstring="Parse the config file. More details here.",
        params=["path: str", "strict: bool"],
    )
    summary = _group_summary(["foo"], {"foo": ent})
    assert "foo (10 lines)" in summary
    assert '"Parse the config file."' in summary
    assert "params: path: str, strict: bool" in summary


def test_group_summary_with_params_only():
    """Entity with params but no docstring → params appear, no docstring quote."""
    ent = _make_entity("bar", 1, 5, params=["x: int", "y"])
    summary = _group_summary(["bar"], {"bar": ent})
    assert "params: x: int, y" in summary
    assert '"' not in summary


def test_group_summary_docstring_no_period():
    """Docstring with no period → full text used as first sentence."""
    ent = _make_entity("baz", 1, 3, docstring="No period here")
    summary = _group_summary(["baz"], {"baz": ent})
    assert '"No period here"' in summary


def test_group_summary_with_section_header():
    """Entity with section_header → section appears first in extras."""

    ent = Entity(
        EntityKind.FUNCTION,
        "foo",
        1,
        5,
        ["foo"],
        section_header="Helpers",
    )
    summary = _group_summary(["foo"], {"foo": ent})
    assert 'section: "Helpers"' in summary


def test_group_summary_no_section_header():
    """Entity without section_header → no 'section:' in summary."""
    ent = _make_entity("bar", 1, 5)
    summary = _group_summary(["bar"], {"bar": ent})
    assert "section:" not in summary


def test_build_group_mermaid_no_edges():
    """Empty graph → no inter-group deps → returns empty string."""
    c = _classified(entities=[], set_2_groups=[["foo"], ["bar"]])
    result = _build_group_mermaid([["foo"], ["bar"]], c)
    assert result == ""


def test_build_group_mermaid_with_inter_group_dep():
    """G0 depends on G1 → Mermaid text with that edge is returned."""
    c = _classified(graph={"foo": {"bar"}, "bar": set()})
    result = _build_group_mermaid([["foo"], ["bar"]], c)
    assert "```mermaid" in result
    assert "G0 --> G1" in result


def test_build_group_mermaid_dep_outside_chunk():
    """Dep to entity outside chunk → dep_gid is None → not added → empty."""
    c = _classified(graph={"foo": {"external"}, "bar": set()})
    result = _build_group_mermaid([["foo"], ["bar"]], c)
    assert result == ""


def test_build_group_mermaid_intra_group_dep():
    """Dep within same SCC group → dep_gid == gid → not added as edge."""
    # foo and baz are in the same group; foo depends on baz (intra-SCC edge)
    c = _classified(graph={"foo": {"baz"}, "baz": {"foo"}, "bar": set()})
    result = _build_group_mermaid([["foo", "baz"], ["bar"]], c)
    assert result == ""


def test_compute_projected_lines_basic():
    """Entities found in map → lines counted per target file."""
    entity_a = _make_entity("func_a", 1, 50)  # 50 lines
    entity_b = _make_entity("func_b", 51, 100)  # 50 lines
    entity_map = {"func_a": entity_a, "func_b": entity_b}
    placements = [
        GroupPlacement(group=["func_a"], target_file="utils.py"),
        GroupPlacement(group=["func_b"], target_file="utils.py"),
    ]
    projected = _compute_projected_lines(placements, entity_map)
    assert projected == {"utils.py": 100}


def test_compute_projected_lines_unknown_entity():
    """Entity name not in map → no lines added for that entity (skipped)."""
    entity_map = {}  # nothing in the map
    placements = [GroupPlacement(group=["ghost"], target_file="utils.py")]
    projected = _compute_projected_lines(placements, entity_map)
    assert projected == {}


def test_compute_projected_lines_multiple_files():
    """Entities across multiple target files → separate line counts."""
    entity_a = _make_entity("func_a", 1, 100)  # 100 lines
    entity_b = _make_entity("func_b", 101, 200)  # 100 lines
    entity_map = {"func_a": entity_a, "func_b": entity_b}
    placements = [
        GroupPlacement(group=["func_a"], target_file="module_a.py"),
        GroupPlacement(group=["func_b"], target_file="module_b.py"),
    ]
    projected = _compute_projected_lines(placements, entity_map)
    assert projected == {"module_a.py": 100, "module_b.py": 100}
