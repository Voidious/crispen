from __future__ import annotations
import textwrap
from unittest.mock import MagicMock
import libcst as cst
from libcst.metadata import MetadataWrapper
from crispen.refactors.function_splitter import (
    _FuncInfo,
    _FunctionCollector,
    _extract_func_source,
    _func_in_changed_range,
)


def _make_func_info(start, end):
    """Create a minimal _FuncInfo for range tests."""
    mock_node = MagicMock()
    return _FuncInfo(
        node=mock_node,
        start_line=start,
        end_line=end,
        class_name=None,
        indent="",
        original_params=[],
    )


def test_func_in_changed_range_overlaps():
    fi = _make_func_info(5, 15)
    assert _func_in_changed_range(fi, [(1, 10)]) is True


def test_func_in_changed_range_no_overlap():
    fi = _make_func_info(5, 10)
    assert _func_in_changed_range(fi, [(20, 30)]) is False


def test_func_in_changed_range_adjacent():
    fi = _make_func_info(5, 10)
    assert _func_in_changed_range(fi, [(10, 20)]) is True


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
