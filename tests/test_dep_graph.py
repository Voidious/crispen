"""Tests for file_limiter.dep_graph — 100% branch coverage."""

from __future__ import annotations

from crispen.file_limiter.dep_graph import build_dep_graph, find_sccs
from crispen.file_limiter.entity_parser import Entity, EntityKind


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _func(name: str, start: int, end: int, defines: list[str] | None = None) -> Entity:
    return Entity(
        kind=EntityKind.FUNCTION,
        name=name,
        start_line=start,
        end_line=end,
        names_defined=defines if defines is not None else [name],
    )


# ---------------------------------------------------------------------------
# build_dep_graph
# ---------------------------------------------------------------------------


def test_build_dep_graph_empty_entities():
    graph = build_dep_graph([], "")
    assert graph == {}


def test_build_dep_graph_no_dependencies():
    source = "def foo():\n    pass\n\ndef bar():\n    pass\n"
    entities = [_func("foo", 1, 2), _func("bar", 4, 5)]
    graph = build_dep_graph(entities, source)
    assert graph == {"foo": set(), "bar": set()}


def test_build_dep_graph_simple_dependency():
    source = "def foo():\n    bar()\n\ndef bar():\n    pass\n"
    entities = [_func("foo", 1, 2), _func("bar", 4, 5)]
    graph = build_dep_graph(entities, source)
    assert "bar" in graph["foo"]
    assert graph["bar"] == set()


def test_build_dep_graph_mutual_dependency():
    source = "def a():\n    b()\n\ndef b():\n    a()\n"
    entities = [_func("a", 1, 2), _func("b", 4, 5)]
    graph = build_dep_graph(entities, source)
    assert "b" in graph["a"]
    assert "a" in graph["b"]


def test_build_dep_graph_self_reference_excluded():
    # Recursive function: foo references itself — no self-loop.
    source = "def foo(n):\n    return foo(n - 1)\n"
    entities = [_func("foo", 1, 2)]
    graph = build_dep_graph(entities, source)
    assert graph == {"foo": set()}


def test_build_dep_graph_external_name_no_edge():
    # Reference to a name not defined by any entity → no edge.
    source = "def foo():\n    os.getcwd()\n"
    entities = [_func("foo", 1, 2)]
    graph = build_dep_graph(entities, source)
    assert graph == {"foo": set()}


def test_build_dep_graph_syntax_error_returns_empty_edge_sets():
    # Invalid source: ast.parse raises SyntaxError; graph keys remain but edges empty.
    entities = [_func("foo", 1, 2)]
    graph = build_dep_graph(entities, "def (invalid")
    assert graph == {"foo": set()}


def test_build_dep_graph_owner_is_none_branch():
    # Entity covers only line 2; Name node on line 1 has no owner → skipped.
    source = "foo()\nx = 1\n"
    entities = [_func("_block_2", 2, 2, ["x"])]
    graph = build_dep_graph(entities, source)
    # foo() is on line 1 which is not covered by any entity — owner is None
    assert graph == {"_block_2": set()}


def test_build_dep_graph_class_depends_on_function():
    source = (
        "def helper():\n    pass\n\nclass Foo:\n    def m(self):\n        helper()\n"
    )
    entities = [_func("helper", 1, 2), _func("Foo", 4, 6)]
    graph = build_dep_graph(entities, source)
    assert "helper" in graph["Foo"]


# ---------------------------------------------------------------------------
# find_sccs
# ---------------------------------------------------------------------------


def test_find_sccs_empty_graph():
    assert find_sccs({}) == []


def test_find_sccs_single_node_no_cycle():
    sccs = find_sccs({"a": set()})
    assert len(sccs) == 1
    assert sccs[0] == ["a"]


def test_find_sccs_two_nodes_no_cycle():
    sccs = find_sccs({"a": {"b"}, "b": set()})
    flat = [node for scc in sccs for node in scc]
    assert set(flat) == {"a", "b"}
    assert all(len(scc) == 1 for scc in sccs)


def test_find_sccs_simple_two_cycle():
    sccs = find_sccs({"a": {"b"}, "b": {"a"}})
    assert len(sccs) == 1
    assert set(sccs[0]) == {"a", "b"}


def test_find_sccs_three_node_cycle():
    graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
    sccs = find_sccs(graph)
    assert len(sccs) == 1
    assert set(sccs[0]) == {"a", "b", "c"}


def test_find_sccs_cycle_plus_dependent():
    # a↔b (cycle), c→a (c depends on cycle but is not part of it)
    graph = {"a": {"b"}, "b": {"a"}, "c": {"a"}}
    sccs = find_sccs(graph)
    assert len(sccs) == 2
    sizes = sorted(len(scc) for scc in sccs)
    assert sizes == [1, 2]


def test_find_sccs_cross_edge_on_stack_false():
    # Covers the `elif on_stack.get(w, False)` → False branch.
    # a→b and a→c; c→b.  b completes first; when c visits b, b is off the stack.
    graph = {"a": {"b", "c"}, "b": set(), "c": {"b"}}
    sccs = find_sccs(graph)
    flat = {node for scc in sccs for node in scc}
    assert flat == {"a", "b", "c"}
    assert all(len(scc) == 1 for scc in sccs)


def test_find_sccs_external_edge_ignored():
    # Edge to a node that is not a key in graph → silently skipped.
    graph = {"a": {"not_in_graph"}}
    sccs = find_sccs(graph)
    assert len(sccs) == 1
    assert sccs[0] == ["a"]


def test_find_sccs_already_visited_skips_main_loop():
    # b is visited recursively from a; the main loop skips b (already in index).
    # This also ensures the `if v not in index` → False branch in the main loop.
    graph = {"a": {"b"}, "b": set()}
    sccs = find_sccs(graph)
    all_names = [node for scc in sccs for node in scc]
    assert set(all_names) == {"a", "b"}
    assert len(all_names) == 2  # no duplicates


def test_find_sccs_reverse_topological_order():
    # b has no dependencies; a depends on b.
    # In reverse topo order: b (dependency) comes before a (dependent).
    graph = {"a": {"b"}, "b": set()}
    sccs = find_sccs(graph)
    names = [scc[0] for scc in sccs]
    assert names.index("b") < names.index("a")


def test_find_sccs_two_independent_cycles():
    # (a↔b) and (c↔d) are independent; two SCCs of size 2.
    graph = {"a": {"b"}, "b": {"a"}, "c": {"d"}, "d": {"c"}}
    sccs = find_sccs(graph)
    assert len(sccs) == 2
    assert all(len(scc) == 2 for scc in sccs)
    all_names = {node for scc in sccs for node in scc}
    assert all_names == {"a", "b", "c", "d"}
