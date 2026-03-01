"""Code generation for FileLimiter: build new files and update original source."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from .advisor import FileLimiterPlan, GroupPlacement
from .classifier import ClassifiedEntities
from .dep_graph import find_sccs
from .entity_parser import Entity, EntityKind


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

# Matches a `from __future__ import …` line (with optional trailing newline).
_FUTURE_IMPORT_LINE_RE = re.compile(r"^from __future__ import .*\n?", re.MULTILINE)


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


def _find_cross_file_imports(
    entity_names: List[str],
    entity_source_map: Dict[str, str],
    name_to_target_file: Dict[str, str],
    current_target: str,
) -> List[str]:
    """Return ``from .module import name`` statements for sibling-file dependencies.

    When an entity being moved to *current_target* references a name that is
    defined by another entity being moved to a different target file, the new
    file needs an explicit import for that name.
    """
    referenced: Set[str] = set()
    for name in entity_names:
        src = entity_source_map.get(name, "")
        referenced |= _collect_name_loads(src)
    from_modules: Dict[str, List[str]] = {}
    for ref_name in sorted(referenced):
        source_file = name_to_target_file.get(ref_name)
        if source_file and source_file != current_target:
            module = _target_module_name(source_file)
            from_modules.setdefault(module, []).append(ref_name)
    return [
        f"from .{module} import {', '.join(sorted(names))}"
        for module, names in sorted(from_modules.items())
    ]


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


def _extract_shared_helpers(
    file_entity_names: Dict[str, List[str]],
    entity_source_map: Dict[str, str],
    entity_map: Dict[str, Entity],
    classified: ClassifiedEntities,
    name_to_target_file: Dict[str, str],
    migrated_names: Set[str],
    original_basename: str,
) -> List[GroupPlacement]:
    """Extract non-migrated functions/classes referenced by migrated entities.

    When a migrated entity in new file F references a non-migrated function X
    from the original O, the generated ``from .O import X`` combined with O's
    re-export ``from .F import …`` creates a cycle O→F→O.

    Fix: pull X (and all helpers X transitively depends on) into the first new
    file that references them.  Mutually-dependent helpers (those forming a
    cycle among themselves) are grouped into the same file via SCC analysis so
    that no inter-new-file cycle can arise (F1→F2→F1).

    Mutates *file_entity_names*, *migrated_names*, and *name_to_target_file*
    in place.  Returns synthetic :class:`GroupPlacement` objects for the
    extracted entities so that :func:`_add_re_exports` can re-import them from
    their new location in the updated original source.
    """
    # Build defined-name → entity-name map for non-migrated FUNCTION/CLASS entities.
    defined_to_entity: Dict[str, str] = {}
    for entity in classified.entities:
        if entity.name in migrated_names:
            continue
        if entity.kind not in (EntityKind.FUNCTION, EntityKind.CLASS):
            continue
        for defined_name in entity.names_defined:
            if name_to_target_file.get(defined_name) == original_basename:
                defined_to_entity[defined_name] = entity.name

    # Collect directly-wanted helpers: entity_name → first target_file that needs it.
    wanted: Dict[str, str] = {}
    for target_file, ent_names in list(file_entity_names.items()):
        for ent_name in ent_names:
            src = entity_source_map.get(ent_name, "")
            for ref_name in _collect_name_loads(src):
                entity_name = defined_to_entity.get(ref_name)
                if entity_name and entity_name not in wanted:
                    wanted[entity_name] = target_file

    if not wanted:
        return []

    # Transitively pull in helpers referenced by already-wanted helpers,
    # preventing O→new-file→O cycles introduced by the extracted helpers.
    queue = list(wanted.keys())
    while queue:
        entity_name = queue.pop(0)
        src = entity_source_map.get(entity_name, "")
        for ref_name in _collect_name_loads(src):
            dep_name = defined_to_entity.get(ref_name)
            if dep_name and dep_name not in wanted:
                wanted[dep_name] = wanted[entity_name]
                queue.append(dep_name)

    # SCC analysis on the sub-graph of wanted helpers to co-locate
    # mutually-dependent helpers and prevent inter-new-file cycles (F1→F2→F1).
    sub_graph: Dict[str, Set[str]] = {
        name: {d for d in classified.graph.get(name, set()) if d in wanted}
        for name in wanted
    }
    sccs = find_sccs(sub_graph)

    plan_order = {tf: i for i, tf in enumerate(file_entity_names)}
    synthetic_placements: List[GroupPlacement] = []
    for scc in sccs:
        scc_targets = {wanted[name] for name in scc}
        chosen = min(scc_targets, key=lambda t: plan_order.get(t, len(plan_order)))
        # Prepend extracted helpers so they appear before the functions that use them.
        file_entity_names[chosen] = list(scc) + file_entity_names[chosen]
        for entity_name in scc:
            migrated_names.add(entity_name)
            entity = entity_map[entity_name]
            for defined_name in entity.names_defined:
                name_to_target_file[defined_name] = chosen
        synthetic_placements.append(GroupPlacement(group=list(scc), target_file=chosen))
    return synthetic_placements


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

        # Mark every line of this import for removal.
        for ln in range(node.lineno, node.end_lineno + 1):
            replacements[ln] = None

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

    # Placements whose target_file matches the original filename would create a
    # self-referential import (e.g. `from .duplicate_extractor import Foo` inside
    # duplicate_extractor.py).  Drop them — entities stay in the original file.
    original_basename = Path(original_path).name
    valid_placements = [
        p for p in plan.placements if p.target_file != original_basename
    ]

    # Group placements by target file (preserving order for topo sort).
    file_entity_names: Dict[str, List[str]] = {}
    for placement in valid_placements:
        file_entity_names.setdefault(placement.target_file, []).extend(placement.group)

    # All migrated entity names.
    migrated_names: Set[str] = {name for p in valid_placements for name in p.group}

    # Build name → target-file map for cross-file import detection.
    name_to_target_file: Dict[str, str] = {}
    for target_file, ent_names in file_entity_names.items():
        for ent_name in ent_names:
            entity = entity_map.get(ent_name)
            if entity:
                for defined_name in entity.names_defined:
                    name_to_target_file[defined_name] = target_file

    # Also map names from non-migrated entities to the original file so that
    # split files can import helpers (e.g. _run) that stayed behind.
    # Exclude import-derived names since _find_needed_imports handles those.
    import_defined_names = {name for info in import_infos for name in info.names}
    for entity in classified.entities:
        if entity.name not in migrated_names:
            for defined_name in entity.names_defined:
                if defined_name not in import_defined_names:
                    name_to_target_file.setdefault(defined_name, original_basename)

    # Extract non-migrated FUNCTION/CLASS entities referenced by migrated ones
    # into the new files that use them, breaking O→F→O import cycles.
    synthetic_placements = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        original_basename,
    )

    # Generate new file contents.
    new_files: Dict[str, str] = {}
    for target_file, ent_names in file_entity_names.items():
        needed = _find_needed_imports(
            ent_names, entity_source_map, import_infos, all_entity_names
        )
        cross = _find_cross_file_imports(
            ent_names, entity_source_map, name_to_target_file, target_file
        )
        entity_srcs = [
            _FUTURE_IMPORT_LINE_RE.sub("", src).rstrip()
            for name in ent_names
            if (src := entity_source_map.get(name))
        ]
        entity_srcs = [s for s in entity_srcs if s]
        parts: List[str] = []
        all_imports = needed + cross
        if all_imports:
            parts.append("\n".join(all_imports))
        parts.extend(entity_srcs)
        new_files[target_file] = _prune_unused_imports("\n\n".join(parts) + "\n")

    # Build updated original source.
    updated = _remove_entity_lines(post_source, migrated_names, entity_map)
    updated = _add_re_exports(
        updated, valid_placements + synthetic_placements, entity_map
    )

    return SplitResult(new_files=new_files, original_source=updated, abort=False)
