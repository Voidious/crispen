from __future__ import annotations
import ast
from typing import Optional, Set


def _strip_top_level_import_lines(src: str) -> str:
    """Return *src* with all top-level import statements removed.

    Uses AST to locate the exact line range of each import node, correctly
    handling multi-line imports.  Returns *src* unchanged when it cannot be
    parsed as Python.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    remove: Set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for ln in range(node.lineno, node.end_lineno + 1):
                remove.add(ln)
    if not remove:
        return src
    lines = src.splitlines(keepends=True)
    return "".join(line for i, line in enumerate(lines, 1) if i not in remove)


def _extract_module_docstring(source: str) -> Optional[str]:
    """Return the module-level docstring source text, or None if absent."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if not (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        return None
    node = tree.body[0]
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno]).rstrip()


def _strip_module_docstring(src: str) -> str:
    """Return *src* with the leading module-level docstring removed."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    if not (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        return src
    node = tree.body[0]
    remove = set(range(node.lineno, node.end_lineno + 1))
    lines = src.splitlines(keepends=True)
    return "".join(line for i, line in enumerate(lines, 1) if i not in remove)
