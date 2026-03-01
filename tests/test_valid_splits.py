from __future__ import annotations
import textwrap
import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider
from crispen.refactors.function_splitter import _find_valid_splits


def _parse_func(source: str):
    """Return (body_stmts, positions, source_lines) for the first function.

    Uses a CSTVisitor to capture body_stmts from the wrapper's internal copy,
    ensuring they match the keys in the positions dict.
    """
    tree = cst.parse_module(source)
    wrapper = MetadataWrapper(tree)
    positions = wrapper.resolve(PositionProvider)

    class _Getter(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self):
            self.stmts: list = []

        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            if not self.stmts:  # first function only
                self.stmts = list(node.body.body)

    getter = _Getter()
    wrapper.visit(getter)
    source_lines = source.splitlines(keepends=True)
    return getter.stmts, positions, source_lines


def test_find_valid_splits_all_valid():
    src = "def foo():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n"
    stmts, positions, lines = _parse_func(src)
    # With a very loose limit, all splits should be valid
    result = _find_valid_splits(stmts, positions, max_lines=1000)
    assert len(result) > 0
    # Ordered latest first
    assert result == sorted(result, reverse=True)


def test_find_valid_splits_none_valid():
    # max_lines=1 means even a 1-stmt head (+ return call = 2 lines) is invalid
    src = "def foo():\n    a = 1\n    b = 2\n    c = 3\n"
    stmts, positions, lines = _parse_func(src)
    result = _find_valid_splits(stmts, positions, max_lines=1)
    assert result == []


def test_find_valid_splits_stops_at_max_candidates():
    # 7 statements → iterates from 6 down, stops after 5 valid candidates
    src = "def foo():\n" + "".join(f"    a{i} = {i}\n" for i in range(7))
    stmts, positions, lines = _parse_func(src)
    result = _find_valid_splits(stmts, positions, max_lines=1000)
    assert len(result) == 5


def test_find_valid_splits_fewer_than_max():
    # 4 statements → at most 3 valid splits (indices 3, 2, 1)
    src = "def foo():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n"
    stmts, positions, lines = _parse_func(src)
    result = _find_valid_splits(stmts, positions, max_lines=1000)
    assert 1 <= len(result) <= 3


def test_find_valid_splits_empty_body():
    # Should not crash with an empty list (though normally not called)
    result = _find_valid_splits([], {}, max_lines=1000)
    assert result == []


def test_find_valid_splits_nested_funcdef_restricts_upper():
    # First nested funcdef at index 2 → valid splits only at indices ≤ 2.
    src = textwrap.dedent(
        """\
        def outer():
            a = 1
            b = 2
            def inner():
                pass
            c = 3
            d = 4
    """
    )
    stmts, positions, lines = _parse_func(src)
    # body_stmts: [a=1, b=2, def inner, c=3, d=4]
    # First nested funcdef at index 2 → upper=2 → range(2, 0, -1) = [2, 1]
    result = _find_valid_splits(stmts, positions, max_lines=1000)
    assert all(i <= 2 for i in result)
    assert 3 not in result
    assert 4 not in result
