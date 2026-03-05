from __future__ import annotations
import ast
import textwrap


def _parse_source_and_init_set(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    return tree


def _parse_source_with_init_set(source: str, set_name: str):
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None, set()
    init_set = set()
    return tree, init_set
