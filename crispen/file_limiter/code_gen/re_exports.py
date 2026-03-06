from __future__ import annotations
import ast
from typing import Dict, List, Optional, Set
from ..advisor import GroupPlacement
from ..entity_parser import Entity, EntityKind
from .import_utils import (
    _collect_name_loads,
    _import_derived_names,
    _relative_import_prefix,
    _target_module_name,
)


def _add_re_exports(
    source: str,
    placements: List[GroupPlacement],
    entity_map: Dict[str, Entity],
    entity_source_map: Dict[str, str],
    external_loads: Set[str] = frozenset(),
    abs_pkg: Optional[str] = None,
    relative_from: Optional[str] = None,
) -> str:
    """Add ``from .module import name`` imports for migrated entities.

    Public names are always re-exported so external callers can still import
    them from the original module.  Private names (starting with ``_``) are
    re-imported when the remaining *source* still references them, or when
    they appear in *external_loads* (names imported from the original module
    by other files in the project).

    When *relative_from* is set (e.g. ``"service/__init__.py"``), import
    prefixes are computed via :func:`_relative_import_prefix` so that
    re-exports from a package ``__init__.py`` reference sibling modules
    correctly (e.g. ``from .utils import Foo`` instead of
    ``from .service.utils import Foo``).

    Import-derived names (names introduced by ``import`` / ``from … import``
    statements inside a TOP_LEVEL entity) are never re-exported: they were
    kept in the original file by :func:`_remove_entity_lines` and cannot
    meaningfully be re-exported from a new module.

    Inserts after the last import line in *source*.  Returns *source* unchanged
    when there are no names to import.
    """
    still_loaded = _collect_name_loads(source)
    re_exports: Dict[str, List[str]] = {}
    # Names added solely for external re-export (not referenced in remaining source).
    # These need "# noqa F401" to suppress flake8 false positives.
    noqa_names: Set[str] = set()
    for placement in placements:
        # Compute the import prefix for this placement's target file.
        if relative_from is not None:
            import_prefix = _relative_import_prefix(
                relative_from, placement.target_file
            )
        elif abs_pkg is not None:
            module = _target_module_name(placement.target_file)
            import_prefix = f"{abs_pkg}.{module}" if abs_pkg else module
        else:
            module = _target_module_name(placement.target_file)
            import_prefix = f".{module}"
        to_import: List[str] = []
        for entity_name in placement.group:
            if entity_name in entity_map:
                entity = entity_map[entity_name]
                defined = entity.names_defined
                if entity.kind == EntityKind.TOP_LEVEL:
                    skip = _import_derived_names(entity_source_map.get(entity_name, ""))
                    defined = [n for n in defined if n not in skip]
            else:
                defined = [entity_name]
            for defined_name in defined:
                if (
                    (
                        not defined_name.startswith("_")
                        and not defined_name.startswith("test_")
                    )
                    or defined_name in still_loaded
                    or (defined_name.startswith("_") and defined_name in external_loads)
                ):
                    to_import.append(defined_name)
                    if defined_name not in still_loaded:
                        noqa_names.add(defined_name)
        if to_import:
            re_exports.setdefault(import_prefix, []).extend(to_import)

    if not re_exports:
        return source

    # Build export statements.  When a name is only there for external re-export
    # (not referenced in the remaining source), add "# noqa F401" so flake8
    # does not flag it as an unused import.  Split mixed imports into two lines
    # so that the noqa comment does not suppress warnings for used names.
    export_stmts: List[str] = []
    for prefix, names in sorted(re_exports.items()):
        sorted_names = sorted(names)
        used = [n for n in sorted_names if n not in noqa_names]
        noqa = [n for n in sorted_names if n in noqa_names]
        if used:
            export_stmts.append(f"from {prefix} import {', '.join(used)}\n")
        for name in noqa:
            export_stmts.append(f"from {prefix} import {name}  # noqa F401\n")

    lines = source.splitlines(keepends=True)
    last_import_line = 0
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_line = max(last_import_line, node.end_lineno)

    return "".join(lines[:last_import_line] + export_stmts + lines[last_import_line:])
