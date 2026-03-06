from __future__ import annotations
from unittest.mock import MagicMock
import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider
from crispen.refactors.function_splitter import (
    _count_body_lines,
    _head_effective_lines,
    _stmts_source,
    FunctionSplitter,
)


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


def _make_mock_response(names_list):
    """Build a mock Anthropic message response for the name_helper_functions tool."""
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "name_helper_functions"
    mock_block.input = {
        "names": [{"id": str(i), "name": n} for i, n in enumerate(names_list)]
    }
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    return mock_response


def test_count_body_lines_no_docstring():
    src = "def foo():\n    x = 1\n    y = 2\n    z = 3\n"
    assert _count_body_lines(src) == 3


def test_count_body_lines_with_docstring():
    src = 'def foo():\n    """doc"""\n    x = 1\n    y = 2\n'
    # docstring skipped; body is lines 2 (x=1) and 3 (y=2)
    assert _count_body_lines(src) == 2


def test_count_body_lines_multiline_docstring():
    src = 'def foo():\n    """line1\n    line2\n    """\n    x = 1\n'
    # docstring spans lines 2-4; body starts at x=1 (line 5)
    result = _count_body_lines(src)
    assert result == 1


def test_count_body_lines_only_docstring():
    # Body has only a docstring → effectively empty
    src = 'def foo():\n    """doc"""\n'
    assert _count_body_lines(src) == 0


def test_count_body_lines_parse_error():
    assert _count_body_lines("def f(\n  !!invalid") == 0


def test_count_body_lines_no_funcdef():
    # Module-level code, no function
    assert _count_body_lines("x = 1\n") == 0


def test_stmts_source_basic():
    src = "def foo():\n    x = 1\n    y = 2\n    z = 3\n"
    stmts, positions, lines = _parse_func(src)
    result = _stmts_source(stmts[:2], lines, positions)
    assert "x = 1" in result
    assert "y = 2" in result
    assert "z = 3" not in result


def test_stmts_source_empty():
    src = "def foo():\n    x = 1\n"
    _, positions, lines = _parse_func(src)
    assert _stmts_source([], lines, positions) == ""


def test_stmts_source_dedented():
    src = "def foo():\n    x = 1\n    y = 2\n"
    stmts, positions, lines = _parse_func(src)
    result = _stmts_source(stmts, lines, positions)
    # Should be dedented (no leading 4-space indent)
    assert result.startswith("x = 1") or result.startswith("x = 1\n")


def test_head_effective_lines_no_docstring():
    src = "def foo():\n    x = 1\n    y = 2\n    z = 3\n"
    stmts, positions, lines = _parse_func(src)
    # split_idx=2: head=[x,y], last=y at line 3, first=x at line 2 → 3-2+2=3
    result = _head_effective_lines(stmts, 2, positions, False)
    assert result == 3


def test_head_effective_lines_with_docstring_normal():
    src = 'def foo():\n    """doc"""\n    x = 1\n    y = 2\n    z = 3\n'
    stmts, positions, lines = _parse_func(src)
    # split_idx=3: head=[doc, x, y], first_non_doc=x at line 3, last=y at line 4
    # 4-3+2=3
    result = _head_effective_lines(stmts, 3, positions, True)
    assert result == 3


def test_head_effective_lines_only_docstring_in_head():
    # split_idx=1 with docstring: first_non_doc_idx=1 >= split_idx=1 → returns 1
    src = 'def foo():\n    """doc"""\n    x = 1\n    y = 2\n'
    stmts, positions, lines = _parse_func(src)
    result = _head_effective_lines(stmts, 1, positions, True)
    assert result == 1


def _make_long_func(n_stmts: int, func_name: str = "long_func") -> str:
    """Build a function with n_stmts independent assignments."""
    lines = [f"def {func_name}():\n"]
    for i in range(n_stmts):
        lines.append(f"    a{i} = {i}\n")
    lines.append("    return 0\n")
    return "".join(lines)


def test_function_splitter_under_limits_no_op():
    # A small function should not be split
    src = "def small():\n    x = 1\n    return x\n"
    splitter = FunctionSplitter([(1, 10)], source=src, verbose=False)
    assert splitter.get_rewritten_source() is None


def test_function_splitter_parse_error_no_crash():
    # Invalid source should not crash
    splitter = FunctionSplitter([(1, 10)], source="def f(\n  !!invalid", verbose=False)
    assert splitter.get_rewritten_source() is None


def test_function_splitter_out_of_range_no_op():
    # Function exists but is outside changed ranges
    src = _make_long_func(80)
    splitter = FunctionSplitter([(200, 300)], source=src, verbose=False, max_lines=10)
    assert splitter.get_rewritten_source() is None
