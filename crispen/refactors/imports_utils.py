from __future__ import annotations
import ast


def _add_imported_local_names(node: ast.AST, out: set) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split(".")[0]
            out.add(name)
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            out.add(name)
