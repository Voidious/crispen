from __future__ import annotations
import ast
from typing import Dict, List, Optional, Set
from .entity_parser import Entity


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


def _import_line_numbers(entity: Entity, entity_src: str) -> Set[int]:
    """Return absolute 1-based line numbers of import statements in *entity*.

    Used to preserve import lines in the original file when a TOP_LEVEL
    entity that mixes imports and assignments is migrated.
    """
    try:
        tree = ast.parse(entity_src)
    except SyntaxError:
        return set()
    result: Set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for rel_ln in range(node.lineno, node.end_lineno + 1):
                result.add(entity.start_line + rel_ln - 1)
    return result


def _prune_inline_redundant_imports(source: str) -> str:
    """Remove function-body imports that duplicate module-level imports.

    When a function-local ``from x import y`` re-imports a name that is
    already provided by a top-level import, flake8 reports an F811
    redefinition warning.  This function removes such redundant inner imports
    (or narrows them when only some names are redundant).

    Returns *source* unchanged when it cannot be parsed or nothing needs
    pruning.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    # Names already available from top-level (module-level) imports.
    top_level_names: Set[str] = set()
    top_level_node_ids: Set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_node_ids.add(id(node))
            for alias in node.names:
                top_level_names.add(
                    alias.asname if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            top_level_node_ids.add(id(node))
            for alias in node.names:
                top_level_names.add(alias.asname if alias.asname else alias.name)

    if not top_level_names:
        return source

    # Find all import nodes that are NOT at module level.
    inner_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and id(node) not in top_level_node_ids
    ]

    if not inner_imports:
        return source

    lines = source.splitlines(keepends=True)
    # Maps 1-based line number → replacement line (None = remove that line).
    line_ops: Dict[int, Optional[str]] = {}

    for stmt in inner_imports:
        if isinstance(stmt, ast.Import):
            kept = [
                a
                for a in stmt.names
                if (a.asname if a.asname else a.name.split(".")[0])
                not in top_level_names
            ]
        else:
            kept = [
                a
                for a in stmt.names
                if (a.asname if a.asname else a.name) not in top_level_names
            ]

        if len(kept) == len(stmt.names):
            continue  # no redundancy — nothing to remove

        # Mark every line of this import for removal.
        for ln in range(stmt.lineno, stmt.end_lineno + 1):
            line_ops[ln] = None

        if kept:
            # Rebuild a narrowed import preserving original indentation.
            alias_strs = [
                f"{a.name} as {a.asname}" if a.asname else a.name for a in kept
            ]
            orig_line = lines[stmt.lineno - 1]
            indent = orig_line[: len(orig_line) - len(orig_line.lstrip())]
            if isinstance(stmt, ast.ImportFrom):
                dots = "." * (stmt.level or 0)
                mod = stmt.module or ""
                new_line = f"{indent}from {dots}{mod} import {', '.join(alias_strs)}\n"
            else:
                new_line = f"{indent}import {', '.join(alias_strs)}\n"
            line_ops[stmt.lineno] = new_line

    if not line_ops:
        return source

    result: List[str] = []
    for i, line in enumerate(lines, 1):
        if i in line_ops:
            repl = line_ops[i]
            if repl is not None:
                result.append(repl)
            # else: None → line is removed
        else:
            result.append(line)
    return "".join(result)
