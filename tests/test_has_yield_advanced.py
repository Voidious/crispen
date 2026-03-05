from __future__ import annotations
import textwrap
import libcst as cst
from crispen.refactors.function_splitter import _has_yield


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
