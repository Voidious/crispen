from __future__ import annotations
from crispen.file_limiter.advisor import _build_group_mermaid
from .helpers import _classified


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
