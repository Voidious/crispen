from __future__ import annotations
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
from .analysis_utils import _collect_name_loads


@dataclass
class ImportInfo:
    """A top-level import statement and the names it introduces."""

    names: List[str]  # names made available by this import
    source: str  # the import statement text (no trailing newline)
    is_future: bool  # True if `from __future__ import ...`


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
