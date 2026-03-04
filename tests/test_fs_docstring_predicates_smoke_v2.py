from __future__ import annotations
from crispen.refactors.function_splitter import _is_docstring_stmt
from tests.fs_ast_parse_helpers_v2 import _parse_stmt


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
