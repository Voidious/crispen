from __future__ import annotations
import libcst as cst
from crispen.refactors.function_splitter import _is_docstring_stmt
from tests.fs_ast_parse_helpers_v2 import _parse_stmt


def test_is_docstring_two_stmts_on_line():
    # Two statements on one line — len(body) != 1
    stmt = _parse_stmt("x = 1; y = 2\n")
    assert _is_docstring_stmt(stmt) is False


def test_is_docstring_compound_stmt():
    # A compound statement (If) is not a SimpleStatementLine
    src = "def f():\n    if True:\n        pass\n"
    stmt = cst.parse_module(src).body[0].body.body[0]
    assert _is_docstring_stmt(stmt) is False
