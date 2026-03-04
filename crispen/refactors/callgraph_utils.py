from __future__ import annotations
import ast
from .ast_parsing import _try_parse_ast_source


def _collect_called_names(source: str) -> set:
    """Return a set of all names called (as functions) in *source*.

    Uses ast.parse + ast.walk to find all ast.Call nodes.  Returns the
    called name: func.id for ast.Name callees, func.attr for ast.Attribute
    callees.  On SyntaxError, returns an empty set.
    """
    tree = _try_parse_ast_source(source)
    if tree is None:
        return set()
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names
