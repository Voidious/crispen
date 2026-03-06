from __future__ import annotations
import ast
from pathlib import Path
from typing import Dict, List, Optional, Set
from ..advisor import GroupPlacement
from ..entity_parser import Entity, EntityKind
from .extractors import _collect_name_loads, _import_derived_names
from .utils import _REL_IMPORT_RE


def _bump_relative_imports(source: str) -> str:
    """Increment the level of every relative import in *source* by one.

    Used when file content is moved one directory level deeper, e.g. when the
    source originally written for ``pkg/module.py`` becomes the content of
    ``pkg/module/__init__.py``, or when new files go into a subdirectory
    package instead of sitting next to the original file.

    ``from .foo`` → ``from ..foo``, ``from ..bar`` → ``from ...bar``, etc.
    Absolute imports are not affected.
    """
    return _REL_IMPORT_RE.sub(lambda m: f"from .{m.group(1)}", source)


def _relative_import_prefix(from_file: str, to_file: str) -> str:
    """Return the Python relative-import prefix for *to_file* as seen from *from_file*.

    Both paths are relative to the same base directory (the original file's
    directory).  Examples::

        _relative_import_prefix("utils.py", "helpers.py")          → ".helpers"
        _relative_import_prefix("sub/a.py", "helpers/b.py")        → "..helpers.b"
        _relative_import_prefix("sub/a.py", "sub/b.py")            → ".b"
    """
    from_parts = Path(from_file).parent.parts  # () for top-level files
    to_module_parts = Path(to_file).with_suffix("").parts  # ("helpers", "b")
    to_dir_parts = Path(to_file).parent.parts  # ("helpers",)

    # Length of the common directory prefix between from_dir and to_dir.
    common_len = 0
    for fp, tp in zip(from_parts, to_dir_parts):
        if fp == tp:
            common_len += 1
        else:
            break

    levels_up = len(from_parts) - common_len
    module = ".".join(to_module_parts[common_len:])
    return "." * (levels_up + 1) + module


def _target_module_name(target_file: str) -> str:
    """Convert a relative target filename to a dotted module name.

    ``"utils.py"`` → ``"utils"``, ``"helpers/io.py"`` → ``"helpers.io"``.
    """
    path = Path(target_file)
    parts = list(path.with_suffix("").parts)
    return ".".join(parts)


def _find_cross_file_imports(
    entity_names: List[str],
    entity_source_map: Dict[str, str],
    name_to_target_file: Dict[str, str],
    current_target: str,
    abs_pkg: Optional[str] = None,
) -> List[str]:
    """Return ``from … import name`` statements for other-file dependencies.

    When an entity being moved to *current_target* references a name that is
    defined by another entity being moved to a different target file, the new
    file needs an explicit import for that name.

    When *abs_pkg* is ``None`` the import prefix is relative (e.g.
    ``from .constants import _CONST``).  When *abs_pkg* is set the import is
    absolute (e.g. ``from tests.constants import _CONST``), which is required
    for test files that pytest loads as top-level modules.
    """
    referenced: Set[str] = set()
    for name in entity_names:
        src = entity_source_map.get(name, "")
        referenced |= _collect_name_loads(src)
    from_files: Dict[str, List[str]] = {}  # source_file → names
    for ref_name in sorted(referenced):
        source_file = name_to_target_file.get(ref_name)
        if source_file and source_file != current_target:
            from_files.setdefault(source_file, []).append(ref_name)

    result = []
    for source_file, names in sorted(from_files.items()):
        if abs_pkg is not None:
            mod = _target_module_name(source_file)
            prefix = f"{abs_pkg}.{mod}" if abs_pkg else mod
        else:
            prefix = _relative_import_prefix(current_target, source_file)
        result.append(f"from {prefix} import {', '.join(sorted(names))}")
    return result


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
