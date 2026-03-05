from __future__ import annotations
import ast
from typing import Dict, List, Optional, Set
from .entity_parser import Entity, EntityKind
from .import_analysis import _collect_name_loads, _parse_source_for_names


def _import_line_numbers(entity: Entity, entity_src: str) -> Set[int]:
    """Return absolute 1-based line numbers of import statements in *entity*.

    Used to preserve import lines in the original file when a TOP_LEVEL
    entity that mixes imports and assignments is migrated.
    """
    tree = _parse_source_for_names(entity_src)
    if tree is None:
        return set()
    result: Set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for rel_ln in range(node.lineno, node.end_lineno + 1):
                result.add(entity.start_line + rel_ln - 1)
    return result


def _remove_entity_lines(
    source: str,
    migrated_names: Set[str],
    entity_map: Dict[str, Entity],
    entity_source_map: Dict[str, str],
) -> str:
    """Return *source* with lines belonging to migrated entities removed.

    For TOP_LEVEL entities, import statement lines are preserved in the
    original file even when the entity is migrated: the remaining code may
    still reference those imported names, and stdlib/third-party names
    cannot be safely re-exported from a new module.
    """
    remove: Set[int] = set()
    preserve: Set[int] = set()
    for name in migrated_names:
        entity = entity_map.get(name)
        if entity is None:
            continue
        for ln in range(entity.start_line, entity.end_line + 1):
            remove.add(ln)
        if entity.kind == EntityKind.TOP_LEVEL:
            preserve |= _import_line_numbers(entity, entity_source_map.get(name, ""))

    lines = source.splitlines(keepends=True)
    return "".join(
        line for i, line in enumerate(lines, 1) if i not in remove or i in preserve
    )


def _mark_import_for_removal(line_mapping, import_node):
    for ln in range(import_node.lineno, import_node.end_lineno + 1):
        line_mapping[ln] = None


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

        _mark_import_for_removal(line_ops, stmt)

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


def _prune_unused_imports(source: str) -> str:
    """Remove or narrow unused imports in a generated file.

    ``from __future__`` and star imports are always preserved.  Multi-name
    imports are narrowed to only the names actually referenced in *source*
    rather than dropped wholesale.  Fully-unused imports are removed entirely.

    Returns *source* unchanged when it cannot be parsed or nothing needs
    pruning.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    used = _collect_name_loads(source)
    lines = source.splitlines(keepends=True)
    # Maps 1-based line number → replacement line (None = remove that line).
    replacements: Dict[int, Optional[str]] = {}

    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue

        # Always preserve __future__ imports.
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue

        # Always preserve star imports.
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            continue

        kept = [
            a
            for a in node.names
            if (a.asname if a.asname else a.name.split(".")[0]) in used
        ]

        if len(kept) == len(node.names):
            continue  # nothing to prune

        _mark_import_for_removal(replacements, node)

        if not kept:
            continue  # fully unused — all lines already removed

        # Rebuild a single-line import with only the kept aliases.
        alias_strs = [f"{a.name} as {a.asname}" if a.asname else a.name for a in kept]
        if isinstance(node, ast.ImportFrom):
            level_dots = "." * (node.level or 0)
            module = node.module or ""
            new_line = f"from {level_dots}{module} import {', '.join(alias_strs)}\n"
        else:
            new_line = f"import {', '.join(alias_strs)}\n"
        replacements[node.lineno] = new_line

    if not replacements:
        return source

    result: List[str] = []
    for i, line in enumerate(lines, 1):
        if i not in replacements:
            result.append(line)
        elif replacements[i] is not None:
            result.append(replacements[i])
        # else: line is removed — skip it
    return "".join(result)
