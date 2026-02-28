"""Build entity dependency graphs and find strongly connected components."""

from __future__ import annotations

import ast
from typing import Dict, List, Set

from .entity_parser import Entity


def build_dep_graph(entities: List[Entity], source: str) -> Dict[str, Set[str]]:
    """Return a dependency graph over *entities*.

    The returned dict maps each ``entity.name`` to the set of entity names
    it *depends on*.  An edge A → B is added when A's source text references
    a name that is defined by B (detected by walking the full module AST and
    matching ``Name`` nodes with ``Load`` context against each entity's line
    range).  Self-loops (A → A) are excluded.

    If *source* cannot be parsed the graph is returned with empty edge sets.
    """
    # name → entity that defines it (last writer wins for duplicate names)
    name_to_definer: Dict[str, str] = {}
    for entity in entities:
        for name in entity.names_defined:
            name_to_definer[name] = entity.name

    # source line → entity that owns it
    line_to_entity: Dict[int, str] = {}
    for entity in entities:
        for line in range(entity.start_line, entity.end_line + 1):
            line_to_entity[line] = entity.name

    graph: Dict[str, Set[str]] = {e.name: set() for e in entities}

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return graph

    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        owner = line_to_entity.get(node.lineno)
        if owner is None:  # pragma: no cover – blank/comment lines have no AST nodes
            continue
        definer = name_to_definer.get(node.id)
        if definer is not None and definer != owner:
            graph[owner].add(definer)

    return graph


def find_sccs(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Return strongly connected components in reverse topological order.

    Uses Tarjan's algorithm.  Each inner list is one SCC (a group of entity
    names that are mutually reachable).  Singleton nodes appear as one-element
    lists.  Edges to nodes that are not keys in *graph* are silently ignored.

    Reverse topological order means: if SCC A depends on SCC B, B is returned
    before A.
    """
    index: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {}
    stack: List[str] = []
    counter: List[int] = [0]
    sccs: List[List[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = counter[0]
        lowlink[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in graph.get(v, set()):
            if w not in graph:
                continue
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w, False):
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc: List[str] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in graph:
        if v not in index:
            strongconnect(v)

    return sccs
