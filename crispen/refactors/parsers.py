from __future__ import annotations
import ast
import textwrap


def _try_parse_ast(source: str):
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _parse_block_source(source: str) -> ast.AST | None:
    try:
        return ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
