from __future__ import annotations
from typing import Dict, Set
from ..dep_graph import find_sccs


def _topo_depth(graph: Dict[str, Set[str]]) -> Dict[str, int]:
    """Return topological depth for each node in a DAG.

    Depth 0 = leaf (no outgoing edges).  A node's depth is 1 + the maximum
    depth of its dependencies.  All dependency nodes must be keys in *graph*.
    On non-DAG inputs (cycles detected), returns 0 for every node as a safe
    fallback so that callers degrade to arbitrary candidate ordering.
    """
    if any(len(s) > 1 for s in find_sccs(graph)):
        return {node: 0 for node in graph}
    depths: Dict[str, int] = {}

    def dfs(node: str) -> int:
        if node in depths:
            return depths[node]
        depths[node] = 1 + max((dfs(dep) for dep in graph[node]), default=-1)
        return depths[node]

    for node in graph:
        dfs(node)
    return depths
