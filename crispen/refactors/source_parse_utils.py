from __future__ import annotations
import ast


def _parse_source_and_init_set(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    return tree
