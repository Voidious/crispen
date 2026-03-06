from __future__ import annotations
import ast
from typing import List, Set
from .models_core import ImportInfo


def _import_derived_names(source: str) -> Set[str]:
    """Return names introduced solely by import statements in *source*.

    These names live in the original file's namespace via its import
    statements and cannot be re-exported from a new module the way
    assignment-defined names can.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name)
    return names


def _collect_name_loads(source: str) -> Set[str]:
    """Return all Name loads referenced in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
    return names


def _extract_import_info(source: str) -> List[ImportInfo]:
    """Return :class:`ImportInfo` for each top-level import in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines(keepends=True)
    result: List[ImportInfo] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [
                alias.asname if alias.asname else alias.name.split(".")[0]
                for alias in node.names
            ]
            src = "".join(lines[node.lineno - 1 : node.end_lineno]).rstrip()
            result.append(ImportInfo(names=names, source=src, is_future=False))
        elif isinstance(node, ast.ImportFrom):
            names = [
                alias.asname if alias.asname else alias.name for alias in node.names
            ]
            src = "".join(lines[node.lineno - 1 : node.end_lineno]).rstrip()
            is_future = node.module == "__future__"
            result.append(ImportInfo(names=names, source=src, is_future=is_future))

    return result
