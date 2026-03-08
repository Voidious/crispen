from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
import ast
import re


@dataclass
class ImportInfo:
    """A top-level import statement and the names it introduces."""

    names: List[str]  # names made available by this import
    source: str  # the import statement text (no trailing newline)
    is_future: bool  # True if `from __future__ import ...`


# Matches any line that is an import statement (plain or from-import).
_IMPORT_LINE_RE = re.compile(r"^(import\s+|from\s+\S.*\s+import\s+)")

# Matches a `from __future__ import …` line (with optional trailing newline).
_FUTURE_IMPORT_LINE_RE = re.compile(r"^from __future__ import .*\n?", re.MULTILINE)

# Matches the leading dots of a relative import (``from .foo`` or ``from ..``).
_REL_IMPORT_RE = re.compile(r"^from (\.+)", re.MULTILINE)


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
            # Reconstruct as a normalized single-line import so that
            # multi-line parenthesized imports (e.g. ``from X import (\n
            # Y,\n Z,\n)``) don't break _merge_from_imports, whose regex
            # only matches the first line.
            dots = "." * (node.level or 0)
            mod = node.module or ""
            alias_strs = [
                f"{a.name} as {a.asname}" if a.asname else a.name for a in node.names
            ]
            src = f"from {dots}{mod} import {', '.join(alias_strs)}"
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


def _bump_relative_imports(source: str, n: int = 1) -> str:
    """Increment the level of every relative import in *source* by *n*.

    Used when file content is moved directory levels deeper, e.g. when the
    source originally written for ``pkg/module.py`` becomes the content of
    ``pkg/module/__init__.py``, or when new files go into a subdirectory
    package instead of sitting next to the original file.

    With n=1: ``from .foo`` → ``from ..foo``, ``from ..bar`` → ``from ...bar``.
    With n=2: ``from .foo`` → ``from ...foo``, etc.
    Absolute imports are not affected.
    """
    for _ in range(n):
        source = _REL_IMPORT_RE.sub(lambda m: f"from .{m.group(1)}", source)
    return source


def _relative_import_prefix(from_file: str, to_file: str) -> str:
    """Return the Python relative-import prefix for *to_file* as seen from *from_file*.

    Both paths are relative to the same base directory (the original file's
    directory).  Examples::

        _relative_import_prefix("utils.py", "helpers.py")          → ".helpers"
        _relative_import_prefix("sub/a.py", "helpers/b.py")        → "..helpers.b"
        _relative_import_prefix("sub/a.py", "sub/b.py")            → ".b"
        _relative_import_prefix("a.py", "__init__.py")             → "."
        _relative_import_prefix("sub/a.py", "sub/__init__.py")     → "."
    """
    to_path = Path(to_file)
    from_parts = Path(from_file).parent.parts  # () for top-level files
    # __init__.py represents the package itself, not a submodule named "__init__".
    if to_path.stem == "__init__":
        to_module_parts = to_path.parent.parts
    else:
        to_module_parts = to_path.with_suffix("").parts  # ("helpers", "b")
    to_dir_parts = to_path.parent.parts  # ("helpers",)

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

    ``"utils.py"`` → ``"utils"``, ``"helpers/io.py"`` → ``"helpers.io"``,
    ``"pkg/__init__.py"`` → ``"pkg"`` (package, not ``"pkg.__init__"``).
    """
    path = Path(target_file)
    if path.stem == "__init__":
        parts = list(path.parent.parts)
    else:
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


_FROM_IMPORT_RE = re.compile(r"^(from\s+\S+)\s+import\s+(.*)")


def _merge_from_imports(imports: List[str]) -> List[str]:
    """Merge ``from X import …`` lines that share the same module prefix.

    When multiple entities each contribute a ``from X import`` for the same
    module but with different name subsets, the naive per-entity approach
    produces duplicate imports such as::

        from .conversion import lua_to_python, python_to_lua
        from .conversion import lua_to_python_preserve_wrapped, python_to_lua

    This function collapses them into a single statement per prefix, with
    names sorted and deduplicated::

        from .conversion import lua_to_python, lua_to_python_preserve_wrapped, python_to_lua  # noqa: E501

    Plain ``import X`` statements are preserved unchanged and appended after
    the merged from-imports.
    """
    from_map: Dict[str, List[str]] = {}
    order: List[str] = []  # first-seen order of prefixes
    plain: List[str] = []
    for imp in imports:
        m = _FROM_IMPORT_RE.match(imp)
        if not m:
            plain.append(imp)
            continue
        prefix = m.group(1)
        names = [n.strip() for n in m.group(2).split(",") if n.strip()]
        if prefix not in from_map:
            from_map[prefix] = []
            order.append(prefix)
        from_map[prefix].extend(names)
    result = []
    for prefix in order:
        unique = sorted(dict.fromkeys(from_map[prefix]))
        result.append(f"{prefix} import {', '.join(unique)}")
    return result + plain


def _find_project_root(path: Path) -> Optional[Path]:
    """Walk up from *path* to find the project root directory.

    Returns the first directory containing ``pyproject.toml``, ``setup.py``,
    ``setup.cfg``, or ``.git``.  Returns ``None`` when the filesystem root is
    reached without finding any of these markers.
    """
    markers = {"pyproject.toml", "setup.py", "setup.cfg", ".git"}
    current = path if path.is_dir() else path.parent
    while True:
        if any((current / m).exists() for m in markers):
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _module_path_from_file(project_root: Path, file_path: Path) -> Optional[str]:
    """Return the dotted Python module path of *file_path* relative to *project_root*.

    Returns ``None`` when *file_path* is not under *project_root*.
    """
    try:
        rel = file_path.relative_to(project_root)
    except ValueError:
        return None
    return ".".join(rel.with_suffix("").parts)


def _abs_package_for_dir(file_path: str) -> Optional[str]:
    """Return the dotted package path of the directory containing *file_path*.

    Used to generate absolute imports for test files so that pytest's default
    import mode (which loads test files as top-level modules, not package
    members) does not choke on ``from .module import …`` syntax.

    Returns an empty string for files sitting directly in the project root,
    ``None`` when the project root cannot be determined.
    """
    orig = Path(file_path).resolve()
    project_root = _find_project_root(orig.parent)
    if project_root is None:
        return None
    try:
        rel = orig.parent.relative_to(project_root)
    except ValueError:
        return None
    return ".".join(rel.parts)


def _collect_external_imported_names(original_path: str) -> Set[str]:
    """Return names imported from *original_path* by other Python files.

    Scans all Python files under the project root for ``from <module> import``
    statements targeting the module corresponding to *original_path*, and
    returns the union of all imported original names (before any ``as`` alias).

    Returns an empty set when *original_path* does not resolve to an existing
    file, the project root cannot be determined, or the path cannot be mapped
    to a module.  Both absolute and relative paths are accepted; relative paths
    are resolved against the current working directory (the repo root when
    crispen is invoked as ``git diff | crispen``).
    """
    orig = Path(original_path).resolve()
    if not orig.exists():
        return set()
    project_root = _find_project_root(orig.parent)
    if project_root is None:
        return set()
    # project_root is an ancestor of orig (derived by walking up from orig.parent),
    # so _module_path_from_file always returns a non-None string here.
    target_module = _module_path_from_file(project_root, orig)
    result: Set[str] = set()
    for py_file in project_root.rglob("*.py"):
        if py_file.resolve() == orig:
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except Exception:
            continue
        # Compute this file's dotted module path for relative-import resolution.
        file_module = _module_path_from_file(project_root, py_file)
        file_pkg_parts = file_module.split(".")[:-1] if file_module else []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 0:
                imported_from = node.module or ""
            else:
                # Relative import: go up (level - 1) packages from file_pkg_parts.
                up = node.level - 1
                if up > len(file_pkg_parts):
                    continue
                base = file_pkg_parts[: len(file_pkg_parts) - up]
                sub = node.module or ""
                imported_from = ".".join(base + ([sub] if sub else []))
            if imported_from != target_module:
                continue
            for alias in node.names:
                result.add(alias.name)
    return result
