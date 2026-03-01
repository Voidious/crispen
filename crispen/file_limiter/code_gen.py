"""Code generation for FileLimiter: build new files and update original source."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from .advisor import FileLimiterPlan, GroupPlacement
from .classifier import ClassifiedEntities
from .entity_parser import Entity


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class ImportInfo:
    """A top-level import statement and the names it introduces."""

    names: List[str]  # names made available by this import
    source: str  # the import statement text (no trailing newline)
    is_future: bool  # True if `from __future__ import ...`


@dataclass
class SplitResult:
    """Output of :func:`generate_file_splits`."""

    new_files: Dict[str, str]  # {target_file: source_code}
    original_source: str  # updated original file source
    abort: bool  # True if generation failed / nothing to split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches any line that is an import statement (plain or from-import).
_IMPORT_LINE_RE = re.compile(r"^[ \t]*(import\s+|from\s+\S.*\s+import\s+)")


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


def _find_needed_imports(
    entity_names: List[str],
    entity_source_map: Dict[str, str],
    import_infos: List[ImportInfo],
    all_entity_names: Set[str],
) -> List[str]:
    """Return import statements needed by the given entities.

    Always includes ``from __future__`` imports.  Other imports are included
    when any of the names they introduce appear in the entities' source.
    Duplicate import source strings are deduplicated.
    """
    referenced: Set[str] = set()
    for name in entity_names:
        src = entity_source_map.get(name, "")
        referenced |= _collect_name_loads(src)

    needed: List[str] = []
    seen: Set[str] = set()
    for info in import_infos:
        if info.source in seen:
            continue
        if info.is_future or any(n in referenced for n in info.names):
            needed.append(info.source)
            seen.add(info.source)

    return needed


def _target_module_name(target_file: str) -> str:
    """Convert a relative target filename to a dotted module name.

    ``"utils.py"`` → ``"utils"``, ``"helpers/io.py"`` → ``"helpers.io"``.
    """
    path = Path(target_file)
    parts = list(path.with_suffix("").parts)
    return ".".join(parts)


def _remove_entity_lines(
    source: str, migrated_names: Set[str], entity_map: Dict[str, Entity]
) -> str:
    """Return *source* with lines belonging to migrated entities removed."""
    remove: Set[int] = set()
    for name in migrated_names:
        entity = entity_map.get(name)
        if entity:
            for ln in range(entity.start_line, entity.end_line + 1):
                remove.add(ln)

    lines = source.splitlines(keepends=True)
    return "".join(line for i, line in enumerate(lines, 1) if i not in remove)


def _add_re_exports(
    source: str, placements: List[GroupPlacement], entity_map: Dict[str, Entity]
) -> str:
    """Add ``from .module import name`` imports for migrated entities.

    Public names are always re-exported so external callers can still import
    them from the original module.  Private names (starting with ``_``) are
    re-imported only when the remaining *source* still references them.

    Inserts after the last import line in *source*.  Returns *source* unchanged
    when there are no names to import.
    """
    still_loaded = _collect_name_loads(source)
    re_exports: Dict[str, List[str]] = {}
    for placement in placements:
        module = _target_module_name(placement.target_file)
        to_import = [
            name
            for name in placement.group
            if not name.startswith("_") or name in still_loaded
        ]
        if to_import:
            re_exports.setdefault(module, []).extend(to_import)

    if not re_exports:
        return source

    export_stmts = [
        f"from .{module} import {', '.join(sorted(names))}\n"
        for module, names in sorted(re_exports.items())
    ]

    lines = source.splitlines(keepends=True)
    last_import_line = 0
    for i, line in enumerate(lines):
        if _IMPORT_LINE_RE.match(line):
            last_import_line = i + 1

    return "".join(lines[:last_import_line] + export_stmts + lines[last_import_line:])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_file_splits(
    classified: ClassifiedEntities,
    plan: FileLimiterPlan,
    post_source: str,
    original_path: str,
) -> SplitResult:
    """Generate new file contents and the updated original source.

    When *plan* is aborted or has no placements, returns :class:`SplitResult`
    with the original source unchanged (``abort`` mirrors ``plan.abort``).
    """
    if plan.abort:
        return SplitResult(new_files={}, original_source=post_source, abort=True)

    if not plan.placements:
        return SplitResult(new_files={}, original_source=post_source, abort=False)

    lines = post_source.splitlines(keepends=True)
    entity_map = {e.name: e for e in classified.entities}

    # Build entity source map (name → stripped source string).
    entity_source_map: Dict[str, str] = {}
    for entity in classified.entities:
        entity_source_map[entity.name] = "".join(
            lines[entity.start_line - 1 : entity.end_line]
        ).rstrip()

    # All entity-defined names (used to limit import matching scope).
    all_entity_names: Set[str] = {
        name for e in classified.entities for name in e.names_defined
    }

    # Extract import info from post-refactor source.
    import_infos = _extract_import_info(post_source)

    # Group placements by target file (preserving order for topo sort).
    file_entity_names: Dict[str, List[str]] = {}
    for placement in plan.placements:
        file_entity_names.setdefault(placement.target_file, []).extend(placement.group)

    # All migrated entity names.
    migrated_names: Set[str] = {name for p in plan.placements for name in p.group}

    # Generate new file contents.
    new_files: Dict[str, str] = {}
    for target_file, ent_names in file_entity_names.items():
        needed = _find_needed_imports(
            ent_names, entity_source_map, import_infos, all_entity_names
        )
        entity_srcs = [
            src for name in ent_names if (src := entity_source_map.get(name))
        ]
        parts: List[str] = []
        if needed:
            parts.append("\n".join(needed))
        parts.extend(entity_srcs)
        new_files[target_file] = "\n\n".join(parts) + "\n"

    # Build updated original source.
    updated = _remove_entity_lines(post_source, migrated_names, entity_map)
    updated = _add_re_exports(updated, plan.placements, entity_map)

    return SplitResult(new_files=new_files, original_source=updated, abort=False)
