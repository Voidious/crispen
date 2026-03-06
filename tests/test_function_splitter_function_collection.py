from __future__ import annotations
import textwrap
import libcst as cst
from libcst.metadata import MetadataWrapper
from crispen.refactors.function_splitter import _FunctionCollector


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
