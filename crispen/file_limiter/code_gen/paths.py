from __future__ import annotations
import ast
from pathlib import Path
from typing import Optional, Set


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
