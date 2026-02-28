"""Parse top-level entities from a Python source file."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class EntityKind(Enum):
    """Kind of top-level entity in a Python source file."""

    FUNCTION = "function"
    CLASS = "class"
    TOP_LEVEL = "top_level"


@dataclass
class Entity:
    """A top-level entity in a Python source file.

    Covers functions, classes, and contiguous blocks of other top-level
    statements (imports, assignments, etc.) together with any comment lines
    directly preceding them.
    """

    kind: EntityKind
    name: str  # function/class name; "_block_N" (start line) for TOP_LEVEL
    start_line: int  # 1-indexed, includes preceding attached comments
    end_line: int  # 1-indexed, inclusive
    names_defined: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers used by parse_entities
# ---------------------------------------------------------------------------


def _target_names(target: ast.expr) -> List[str]:
    """Recursively collect plain Name identifiers from an assignment target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: List[str] = []
        for elt in target.elts:
            names.extend(_target_names(elt))
        return names
    return []


def _collect_defined_names(node: ast.AST) -> List[str]:
    """Return the module-level name(s) that *node* makes available."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        names: List[str] = []
        for target in node.targets:
            names.extend(_target_names(target))
        return names
    if isinstance(node, ast.AnnAssign):
        if node.value is not None and isinstance(node.target, ast.Name):
            return [node.target.id]
        return []
    if isinstance(node, ast.Import):
        return [
            alias.asname if alias.asname else alias.name.split(".")[0]
            for alias in node.names
        ]
    if isinstance(node, ast.ImportFrom):
        return [alias.asname if alias.asname else alias.name for alias in node.names]
    return []


def _find_attached_comment_start(lines: List[str], stmt_start: int) -> int:
    """Return the 1-indexed first line of comments attached to *stmt_start*.

    Scans backward from the line just before *stmt_start*.  A comment line
    (stripped text starts with ``#``) with no intervening blank line is
    considered attached.  Scanning stops at any non-comment line.
    """
    first_comment = stmt_start
    i = stmt_start - 2  # 0-indexed line just before stmt_start
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("#"):
            first_comment = i + 1  # convert to 1-indexed
            i -= 1
        else:
            break
    return first_comment


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_entities(source: str) -> List[Entity]:
    """Parse *source* into a flat list of top-level Entities in source order.

    Three kinds of entities are produced:

    * ``FUNCTION`` — every top-level ``def`` or ``async def``, including its
      decorators and any directly-preceding comment lines.
    * ``CLASS`` — every top-level ``class``, including decorators and attached
      comments.
    * ``TOP_LEVEL`` — one entity per contiguous run of all other top-level
      statements (imports, assignments, bare expressions, …), together with
      any comment lines directly before the first statement in the run.

    Returns an empty list if *source* cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines(keepends=True)
    entities: List[Entity] = []
    pending_block: List[ast.stmt] = []

    def _flush_block() -> None:
        """Flush *pending_block* as a single TOP_LEVEL entity."""
        if not pending_block:
            return
        first_stmt_start = pending_block[0].lineno
        block_start = _find_attached_comment_start(lines, first_stmt_start)
        block_end = pending_block[-1].end_lineno
        names: List[str] = []
        for stmt in pending_block:
            names.extend(_collect_defined_names(stmt))
        entities.append(
            Entity(
                kind=EntityKind.TOP_LEVEL,
                name=f"_block_{block_start}",
                start_line=block_start,
                end_line=block_end,
                names_defined=names,
            )
        )
        pending_block.clear()

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _flush_block()
            # Use the first decorator line when decorators are present.
            if stmt.decorator_list:
                stmt_first_line = stmt.decorator_list[0].lineno
            else:
                stmt_first_line = stmt.lineno
            comment_start = _find_attached_comment_start(lines, stmt_first_line)
            kind = (
                EntityKind.CLASS
                if isinstance(stmt, ast.ClassDef)
                else EntityKind.FUNCTION
            )
            entities.append(
                Entity(
                    kind=kind,
                    name=stmt.name,
                    start_line=comment_start,
                    end_line=stmt.end_lineno,
                    names_defined=[stmt.name],
                )
            )
        else:
            pending_block.append(stmt)

    _flush_block()
    return entities
