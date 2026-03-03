from __future__ import annotations
import ast


def _collect_attribute_names(source: str) -> set:
    """Return all attribute names (dot-access names) anywhere in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def _collect_called_attr_names(source: str) -> set:
    """Return attribute names used as method calls in *source*.

    Unlike :func:`_collect_attribute_names`, this only returns names that
    appear as the attribute of a call expression (i.e. ``obj.method(...)``).
    Plain attribute reads and type annotations like ``ast.AST`` are ignored,
    so the new-method-call check does not produce false positives for
    standard-library type references.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
