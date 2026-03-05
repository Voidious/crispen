from __future__ import annotations
import textwrap
import libcst as cst
from crispen.refactors.function_splitter import _has_nested_funcdef


def test_has_nested_funcdef_with_nested():
    src = textwrap.dedent(
        """\
        def outer():
            x = 1
            def inner():
                return x
            return inner
    """
    )
    func = cst.parse_module(src).body[0]
    assert _has_nested_funcdef(func) is True


def test_has_nested_funcdef_without_nested():
    src = "def foo():\n    x = 1\n    return x\n"
    func = cst.parse_module(src).body[0]
    assert _has_nested_funcdef(func) is False


def test_has_nested_funcdef_first_stmt():
    # Nested funcdef is the very first statement in the body
    src = textwrap.dedent(
        """\
        def outer():
            def inner():
                pass
            return inner()
    """
    )
    func = cst.parse_module(src).body[0]
    assert _has_nested_funcdef(func) is True
