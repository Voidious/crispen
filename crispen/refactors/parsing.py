from __future__ import annotations
import ast
import textwrap


def _parse_block_source(source: str) -> ast.AST | None:
    try:
        return ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
