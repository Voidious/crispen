from __future__ import annotations
import textwrap
import libcst as cst
from crispen.refactors.function_splitter import _has_yield


def test_has_yield_simple():
    src = "def gen():\n    yield 1\n"
    func = cst.parse_module(src).body[0]
    assert _has_yield(func) is True


def test_has_yield_from():
    src = "def gen():\n    yield from [1, 2]\n"
    func = cst.parse_module(src).body[0]
    assert _has_yield(func) is True


def test_has_yield_none():
    src = "def foo():\n    return 1\n"
    func = cst.parse_module(src).body[0]
    assert _has_yield(func) is False


def test_has_yield_nested_not_counted():
    src = textwrap.dedent(
        """\
        def foo():
            def inner():
                yield 1
            return inner
    """
    )
    func = cst.parse_module(src).body[0]
    # yield is inside nested function, should not count
    assert _has_yield(func) is False
