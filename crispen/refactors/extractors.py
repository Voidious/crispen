from __future__ import annotations
import ast


def _extract_import_names(node, target_set):
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split(".")[0]
            target_set.add(name)
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            target_set.add(name)


def _extract_defined_names(source: str) -> set:
    """Return all function and class names defined anywhere in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
