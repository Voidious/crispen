from __future__ import annotations
from crispen.file_limiter.code_gen import _topo_depth


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
