from __future__ import annotations
from crispen.file_limiter.code_gen import _topo_depth


def test_topo_depth_cycle():
    graph = {"a": {"b"}, "b": {"a"}}
    assert _topo_depth(graph) == {"a": 0, "b": 0}
