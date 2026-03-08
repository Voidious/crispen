from __future__ import annotations
from libcst.metadata import MetadataWrapper
from crispen.refactors.function_splitter import (
    _FunctionCollector,
    _count_body_lines,
    _extract_func_source,
    _head_effective_lines,
    _is_docstring_stmt,
    _run_with_timeout,
    _stmts_source,
)
from .utils import _make_func_info, _parse_func, _parse_stmt
import textwrap
import libcst as cst
import pytest


def test_is_docstring_triple_quoted():
    stmt = _parse_stmt('def f():\n    """doc"""\n').body.body[0]
    assert _is_docstring_stmt(stmt) is True


def test_is_docstring_single_quoted():
    stmt = _parse_stmt("def f():\n    'doc'\n").body.body[0]
    assert _is_docstring_stmt(stmt) is True


def test_is_docstring_concatenated():
    stmt = _parse_stmt('def f():\n    "foo" "bar"\n').body.body[0]
    assert _is_docstring_stmt(stmt) is True


def test_is_docstring_non_docstring_expr():
    # A numeric literal is not a docstring
    stmt = _parse_stmt("def f():\n    42\n").body.body[0]
    assert _is_docstring_stmt(stmt) is False


def test_is_docstring_import():
    stmt = _parse_stmt("import os\n")
    assert _is_docstring_stmt(stmt) is False


def test_is_docstring_assignment():
    stmt = _parse_stmt("x = 1\n")
    assert _is_docstring_stmt(stmt) is False


def test_is_docstring_two_stmts_on_line():
    # Two statements on one line — len(body) != 1
    stmt = _parse_stmt("x = 1; y = 2\n")
    assert _is_docstring_stmt(stmt) is False


def test_is_docstring_compound_stmt():
    # A compound statement (If) is not a SimpleStatementLine
    src = "def f():\n    if True:\n        pass\n"
    stmt = cst.parse_module(src).body[0].body.body[0]
    assert _is_docstring_stmt(stmt) is False


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


def test_run_with_timeout_propagates_exception():
    def _raise():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        _run_with_timeout(_raise, 5)


def test_extract_func_source():
    lines = ["line1\n", "line2\n", "line3\n", "line4\n"]
    fi = _make_func_info(2, 3)
    result = _extract_func_source(fi, lines)
    assert result == "line2\nline3\n"


def test_function_collector_module_level():
    src = "def foo():\n    x = 1\n"
    tree = cst.parse_module(src)
    wrapper = MetadataWrapper(tree)
    collector = _FunctionCollector()
    wrapper.visit(collector)
    assert len(collector.functions) == 1
    assert collector.functions[0].node.name.value == "foo"
    assert collector.functions[0].class_name is None
    assert collector.functions[0].indent == ""


def test_function_collector_class_method():
    src = textwrap.dedent(
        """\
        class Foo:
            def bar(self):
                pass
    """
    )
    tree = cst.parse_module(src)
    wrapper = MetadataWrapper(tree)
    collector = _FunctionCollector()
    wrapper.visit(collector)
    assert len(collector.functions) == 1
    assert collector.functions[0].class_name == "Foo"
    assert collector.functions[0].indent == "    "


def test_function_collector_skips_async():
    src = "async def foo():\n    pass\n"
    tree = cst.parse_module(src)
    wrapper = MetadataWrapper(tree)
    collector = _FunctionCollector()
    wrapper.visit(collector)
    assert len(collector.functions) == 0


def test_function_collector_skips_generator():
    src = "def gen():\n    yield 1\n"
    tree = cst.parse_module(src)
    wrapper = MetadataWrapper(tree)
    collector = _FunctionCollector()
    wrapper.visit(collector)
    assert len(collector.functions) == 0


def test_function_collector_skips_nested_functions():
    # Functions with nested funcdefs are skipped entirely; inner functions
    # (inside a function scope) are also skipped by the scope-kind guard.
    src = textwrap.dedent(
        """\
        def outer():
            def inner():
                pass
            return inner
    """
    )
    tree = cst.parse_module(src)
    wrapper = MetadataWrapper(tree)
    collector = _FunctionCollector()
    wrapper.visit(collector)
    # outer has a nested funcdef → skipped; inner is in a function scope → skipped
    assert len(collector.functions) == 0


def test_function_collector_captures_params():
    src = "def foo(a, b, c):\n    pass\n"
    tree = cst.parse_module(src)
    wrapper = MetadataWrapper(tree)
    collector = _FunctionCollector()
    wrapper.visit(collector)
    assert collector.functions[0].original_params == ["a", "b", "c"]
