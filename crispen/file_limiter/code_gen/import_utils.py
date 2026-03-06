from __future__ import annotations
import ast
from pathlib import Path
from typing import Dict, List, Optional, Set
from ..entity_parser import Entity, EntityKind
from .models import ImportInfo
from .utils import _REL_IMPORT_RE


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
