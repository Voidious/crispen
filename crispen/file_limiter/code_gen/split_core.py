from __future__ import annotations
import ast
import re
from typing import Dict, Optional, Set
from ..entity_parser import Entity, EntityKind

# Matches any line that is an import statement (plain or from-import).
_IMPORT_LINE_RE = re.compile(r"^(import\s+|from\s+\S.*\s+import\s+)")

# Matches a `from __future__ import …` line (with optional trailing newline).
_FUTURE_IMPORT_LINE_RE = re.compile(r"^from __future__ import .*\n?", re.MULTILINE)

# Matches the leading dots of a relative import (``from .foo`` or ``from ..``).
_REL_IMPORT_RE = re.compile(r"^from (\.+)", re.MULTILINE)


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
