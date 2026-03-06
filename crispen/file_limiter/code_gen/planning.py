from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
from ..advisor import FileLimiterPlan
from ..classifier import ClassifiedEntities
from ..dep_graph import find_sccs
from ..entity_parser import EntityKind
from .analysis import _collect_name_loads, _remove_entity_lines
from .docstrings import _extract_module_docstring, _strip_module_docstring
from .helpers_extraction import _extract_shared_helpers
from .imports import (
    _bump_relative_imports,
    _extract_import_info,
    _find_cross_file_imports,
    _find_needed_imports,
    _prune_inline_redundant_imports,
    _prune_unused_imports,
    _strip_top_level_import_lines,
)
from .paths import _abs_package_for_dir, _collect_external_imported_names
from .reexports import _add_re_exports
from .split_core import _FUTURE_IMPORT_LINE_RE


@dataclass
class SplitResult:
    """Output of :func:`generate_file_splits`."""

    new_files: Dict[str, str]  # {target_file: source_code}
    original_source: str  # updated original file source
    abort: bool  # True if generation failed / nothing to split
    abort_reason: str = ""  # human-readable explanation when abort=True


def generate_file_splits(
    classified: ClassifiedEntities,
    plan: FileLimiterPlan,
    post_source: str,
    original_path: str,
    subdir_name: Optional[str] = None,
) -> SplitResult:
    """Generate new file contents and the updated original source.

    When *plan* is aborted or has no placements, returns :class:`SplitResult`
    with the original source unchanged (``abort`` mirrors ``plan.abort``).

    When *subdir_name* is set (e.g. ``"service"``), the file is being split
    into a package subdirectory.  The "original" file is treated as
    ``service/__init__.py`` for dependency-graph and import-prefix purposes,
    so cross-file imports within the package use correct relative paths.
    """
    if plan.abort:
        return SplitResult(
            new_files={},
            original_source=post_source,
            abort=True,
            abort_reason=plan.abort_reason,
        )

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
    # In subdir-split mode the "original" is the package __init__.py; use that
    # name throughout so dependency-graph edges and import prefixes are correct.
    original_basename = (
        f"{subdir_name}/__init__.py" if subdir_name else Path(original_path).name
    )
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
    # Exclude import-derived names (_find_needed_imports handles those).
    import_defined_names = {name for info in import_infos for name in info.names}
    name_to_target_file: Dict[str, str] = {}
    for target_file, ent_names in file_entity_names.items():
        for ent_name in ent_names:
            entity = entity_map.get(ent_name)
            if entity:
                for defined_name in entity.names_defined:
                    if defined_name not in import_defined_names:
                        name_to_target_file[defined_name] = target_file

    # Also map names from non-migrated entities to the original file so that
    # split files can import helpers (e.g. _run) that stayed behind.
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

    # Collect names that external files (outside the module being split) import
    # from the original file.  Private symbols in this set must get a re-export
    # proxy even though they are no longer referenced by the remaining source.
    external_loads = _collect_external_imported_names(original_path)

    # Detect circular imports.  Cycles can pass through the original file:
    # a new file that imports a non-migrated name from the original can form a
    # chain back to the original via the re-exports the original adds.
    # Model the original as an explicit node in the dependency graph.
    #
    # Original's outgoing edges: it will re-export a migrated name when the
    # name is public (no _/test_ prefix), referenced by a non-migrated entity,
    # or imported by an external file (external_loads).
    non_migrated_loads: Set[str] = set()
    for ent_name, src in entity_source_map.items():
        if ent_name not in migrated_names:
            non_migrated_loads |= _collect_name_loads(src)

    all_dep_nodes = set(file_entity_names.keys()) | {original_basename}
    file_deps: Dict[str, Set[str]] = {node: set() for node in all_dep_nodes}
    for target_file, ent_names in file_entity_names.items():
        for ent_name in ent_names:
            src = entity_source_map.get(ent_name, "")
            for ref_name in _collect_name_loads(src):
                dep_file = name_to_target_file.get(ref_name)
                if dep_file and dep_file != target_file and dep_file in file_deps:
                    file_deps[target_file].add(dep_file)
    for placement in valid_placements + synthetic_placements:
        for ent_name in placement.group:
            entity = entity_map.get(ent_name)
            if entity:
                for defined_name in entity.names_defined:
                    if (
                        (
                            not defined_name.startswith("_")
                            and not defined_name.startswith("test_")
                        )
                        or defined_name in non_migrated_loads
                        or defined_name in external_loads
                    ):
                        file_deps[original_basename].add(placement.target_file)
                        break
    if any(len(scc) > 1 for scc in find_sccs(file_deps)):
        return SplitResult(
            new_files={},
            original_source=post_source,
            abort=True,
            abort_reason="proposed split would create circular file imports",
        )

    # Use absolute imports when the original file is a test file.  Pytest's
    # default import mode loads test files as top-level modules (not package
    # members), so relative imports like `from .helpers import foo` would
    # raise ImportError at collection time.
    abs_pkg: Optional[str] = None
    if Path(original_path).name.startswith("test_"):
        abs_pkg = _abs_package_for_dir(original_path)

    # In subdir-split mode, new files live inside a package subdirectory and
    # can use relative imports for cross-file references within that package.
    abs_pkg_for_new_files: Optional[str] = None if subdir_name else abs_pkg

    # Generate new file contents.
    new_files: Dict[str, str] = {}
    for target_file, ent_names in file_entity_names.items():
        needed = _find_needed_imports(
            ent_names, entity_source_map, import_infos, all_entity_names
        )
        if subdir_name is not None:
            needed = [_bump_relative_imports(s) for s in needed]
        cross = _find_cross_file_imports(
            ent_names,
            entity_source_map,
            name_to_target_file,
            target_file,
            abs_pkg=abs_pkg_for_new_files,
        )
        entity_srcs = []
        for _ent_name in ent_names:
            _src = entity_source_map.get(_ent_name)
            if _src is None:
                continue
            _entity = entity_map.get(_ent_name)
            if _entity and _entity.kind == EntityKind.TOP_LEVEL:
                # Imports are emitted separately by _find_needed_imports; strip
                # them from the entity body to prevent duplicate import stmts.
                _src = _strip_top_level_import_lines(_src)
                if subdir_name is not None:
                    # In subdir-split mode the module docstring belongs in
                    # __init__.py rather than in one of the child modules.
                    _src = _strip_module_docstring(_src)
            else:
                _src = _FUTURE_IMPORT_LINE_RE.sub("", _src)
            _src = _src.rstrip()
            entity_srcs.append(_src)
        entity_srcs = [s for s in entity_srcs if s]
        parts: List[str] = []
        all_imports = needed + cross
        if all_imports:
            parts.append("\n".join(all_imports))
        parts.extend(entity_srcs)
        pruned = _prune_unused_imports("\n\n".join(parts) + "\n")
        new_files[target_file] = _prune_inline_redundant_imports(pruned)

    # Build updated original source.
    updated = _remove_entity_lines(
        post_source, migrated_names, entity_map, entity_source_map
    )
    updated = _prune_unused_imports(updated)
    # For non-test subdir splits, re-exports from the __init__.py use relative
    # import prefixes computed from inside the package (e.g. ".utils" not
    # ".service.utils").  For test files the original keeps existing abs_pkg
    # behaviour so pytest can find the re-exported symbols.
    is_test_file = Path(original_path).name.startswith("test_")
    # In a non-test subdir split the updated source becomes subdir/__init__.py,
    # which sits one directory level deeper than the original file.  Any
    # relative imports it still contains (e.g. ``from .. import llm_client``
    # or ``from .base import Foo``) therefore need one extra dot so they keep
    # pointing at the same modules.  Re-exports added by _add_re_exports below
    # are already computed from the __init__.py's perspective and are correct.
    if subdir_name is not None and not is_test_file:
        updated = _bump_relative_imports(updated)
    if subdir_name is not None:
        # If the original file had a module docstring and it was migrated away,
        # place it in subdir/__init__.py in both cases: for non-test splits
        # the docstring is prepended to `updated` which runner.py redirects to
        # __init__.py; for test splits it is written directly to __init__.py
        # (runner.py does not redirect `updated` for test files).
        _module_doc = _extract_module_docstring(post_source)
        if _module_doc and not _extract_module_docstring(updated):
            if is_test_file:
                new_files[f"{subdir_name}/__init__.py"] = _module_doc + "\n"
            else:
                updated = _module_doc + "\n\n" + updated
    relative_from: Optional[str] = (
        f"{subdir_name}/__init__.py" if (subdir_name and not is_test_file) else None
    )
    updated = _add_re_exports(
        updated,
        valid_placements + synthetic_placements,
        entity_map,
        entity_source_map,
        external_loads=external_loads,
        abs_pkg=abs_pkg,
        relative_from=relative_from,
    )

    return SplitResult(new_files=new_files, original_source=updated, abort=False)
