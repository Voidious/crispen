from __future__ import annotations
import ast
import textwrap


def _try_parse_ast_source(source: str):
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _try_parse_dedent_block(src: str) -> ast.AST | None:
    try:
        return ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return None
