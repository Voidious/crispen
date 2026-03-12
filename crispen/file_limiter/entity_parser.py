"""Parse top-level entities from a Python source file."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


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
    docstring: Optional[str] = None  # first docstring if present
    params: List[str] = field(default_factory=list)  # up to 3 param descriptions
    section_header: Optional[str] = (
        None  # label of the nearest preceding section header
    )


# ---------------------------------------------------------------------------
# Section header detection
# ---------------------------------------------------------------------------

# Matches a "divider" line: # followed by 3+ copies of the same non-alphanumeric,
# non-space character (e.g. ---..., ===..., ###..., ***..., ~~~...).
_SECTION_DIVIDER_RE = re.compile(r"^# ([^a-zA-Z0-9\s])\1{2,}\s*$")

# Matches a single-line section header: # --- label --- or # === LABEL === etc.
# Group 1 = delimiter char, group 2 = label text.
_SECTION_SINGLE_RE = re.compile(r"^# ([^a-zA-Z0-9\s])\1{2,} (.+?) \1{2,}\s*$")


def _parse_section_headers(lines: List[str]) -> List[Tuple[int, int, str]]:
    """Return ``(start_line, end_line, label)`` triples for section header blocks.

    Both line numbers are 1-indexed and inclusive.  Two patterns are
    recognised:

    * **3-line block** — a divider line, a label line, another divider line::

          # ---------------------------------------------------------------------------
          # Helpers used by parse_entities
          # ---------------------------------------------------------------------------

    * **Single-line** — ``# --- Label ---`` or ``# === LABEL ===``

    *lines* may contain trailing newlines; they are stripped before matching.
    """
    result: List[Tuple[int, int, str]] = []
    n = len(lines)
    i = 0
    while i < n:
        stripped = lines[i].rstrip("\n").rstrip()
        # Try 3-line block first (needs at least 3 remaining lines).
        if i + 2 < n and _SECTION_DIVIDER_RE.match(stripped):
            middle = lines[i + 1].rstrip("\n").rstrip()
            bottom = lines[i + 2].rstrip("\n").rstrip()
            if (
                middle.startswith("# ")
                and middle[2:].strip()
                and not _SECTION_DIVIDER_RE.match(middle)
                and _SECTION_DIVIDER_RE.match(bottom)
            ):
                label = middle[2:].strip()
                result.append((i + 1, i + 3, label))  # 1-indexed, inclusive
                i += 3
                continue
        # Try single-line header.
        m = _SECTION_SINGLE_RE.match(stripped)
        if m:
            result.append((i + 1, i + 1, m.group(2).strip()))  # 1-indexed
            i += 1
            continue
        i += 1
    return result


# ---------------------------------------------------------------------------
# Helpers used by parse_entities
# ---------------------------------------------------------------------------


def _format_params(args: ast.arguments, max_params: int = 3) -> List[str]:
    """Return up to *max_params* formatted parameter strings, excluding self/cls.

    Appends ``"..."`` when there are more parameters than *max_params*.
    """
    all_args = args.args
    if all_args and all_args[0].arg in ("self", "cls"):
        all_args = all_args[1:]
    result = []
    for arg in all_args[:max_params]:
        if arg.annotation is not None:
            result.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
        else:
            result.append(arg.arg)
    if len(all_args) > max_params:
        result.append("...")
    return result


def _class_init_params(node: ast.ClassDef) -> List[str]:
    """Return the first 3 formatted params of __init__ (excluding self), or []."""
    for item in node.body:
        if (
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "__init__"
        ):
            return _format_params(item.args)
    return []


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
            if isinstance(stmt, ast.ClassDef):
                kind = EntityKind.CLASS
                params = _class_init_params(stmt)
            else:
                kind = EntityKind.FUNCTION
                params = _format_params(stmt.args)
            entities.append(
                Entity(
                    kind=kind,
                    name=stmt.name,
                    start_line=comment_start,
                    end_line=stmt.end_lineno,
                    names_defined=[stmt.name],
                    docstring=ast.get_docstring(stmt),
                    params=params,
                )
            )
        else:
            pending_block.append(stmt)

    _flush_block()

    # Assign section_header: the label of the nearest preceding section header
    # block whose end_line is before the entity's start_line.
    headers = _parse_section_headers(lines)
    if headers:
        h_idx = 0
        current_label: Optional[str] = None
        for entity in entities:
            while h_idx < len(headers) and headers[h_idx][1] < entity.start_line:
                current_label = headers[h_idx][2]
                h_idx += 1
            entity.section_header = current_label

    return entities
