"""Shared repo-wide module index: scan .py files, resolve imports, list definitions.

Originally built inside patch_rewriter.py for its call-graph BFS resolution;
extracted here so other repo-wide passes (e.g. duplicate_extractor's
match-function-across-the-codebase mode) can reuse the same scan instead of
re-walking the repo themselves.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

# Directory names excluded from repo-wide file scans.
EXCLUDED_DIR_NAMES = frozenset(
    {".venv", "venv", "env", ".tox", "__pycache__", "node_modules"}
)


def collect_defined_names(source: str) -> Set[str]:
    """Return top-level function and class names defined in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def file_to_module_and_package(file_path: Path, repo_root: Path) -> Tuple[str, str]:
    """Return ``(module_dotted_path, package_dotted_path)`` for a ``.py`` file.

    ``pkg/utils/__init__.py`` → ``("pkg.utils", "pkg.utils")``.
    ``pkg/utils/helpers.py``  → ``("pkg.utils.helpers", "pkg.utils")``.
    """
    rel = file_path.relative_to(repo_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
        module = ".".join(parts)
        package = module
    else:
        parts[-1] = parts[-1][:-3]
        module = ".".join(parts)
        package = ".".join(parts[:-1])
    return module, package


def parse_imports(source: str, package: str) -> Dict[str, Tuple[str, str]]:
    """Parse import statements in *source*.

    Returns ``local_name → (module, orig_name)``.

    *package* is the dotted path of the package that contains this module
    (used to resolve relative imports).  For example, for a file at
    ``pkg/utils/helpers.py`` the package is ``"pkg.utils"``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    result: Dict[str, Tuple[str, str]] = {}
    pkg_parts = package.split(".") if package else []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname if alias.asname else alias.name.split(".")[0]
                result[local] = (alias.name, alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                # Relative import: level=1 means same package (0 levels up),
                # level=2 means one level up, etc.
                go_up = node.level - 1
                if go_up > len(pkg_parts):
                    continue  # invalid — can't go above root
                base_parts = pkg_parts[: len(pkg_parts) - go_up] if go_up else pkg_parts
                base = ".".join(base_parts)
                if node.module:
                    mod_path = f"{base}.{node.module}" if base else node.module
                else:
                    mod_path = base
            else:
                mod_path = node.module or ""  # pragma: no branch
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname if alias.asname else alias.name
                result[local] = (mod_path, alias.name)
    return result


@dataclass
class RepoIndex:
    """Pre-built repo-wide module index for cross-file resolution.

    Built once per run by scanning the repo; callers may extend the dict
    fields in place afterward (e.g. to layer in-memory sources not yet
    written to disk on top of the scan). Import maps are resolved lazily
    and cached on first access.
    """

    module_to_source: Dict[str, str]  # dotted module path → source
    module_to_package: Dict[str, str]  # dotted module path → package path
    module_to_defs: Dict[str, Set[str]]  # dotted module path → top-level names
    file_to_module: Dict[str, str]  # abs file path → dotted module path
    _import_cache: Dict[str, Dict[str, Tuple[str, str]]] = field(default_factory=dict)

    def get_imports(self, module: str) -> Dict[str, Tuple[str, str]]:
        """Return ``{local_name: (module_path, orig_name)}`` for *module*, cached."""
        if module not in self._import_cache:
            src = self.module_to_source.get(module, "")
            pkg = self.module_to_package.get(module, "")
            self._import_cache[module] = parse_imports(src, pkg)
        return self._import_cache[module]


def build_repo_index(
    repo_root: Optional[str],
    per_file_sources: Optional[Dict[str, str]] = None,
) -> RepoIndex:
    """Scan all ``.py`` files under *repo_root* into a :class:`RepoIndex`.

    *per_file_sources* (abs file path → source) overrides on-disk content for
    files already modified in memory. Returns an empty index if *repo_root*
    is ``None``.
    """
    module_to_source: Dict[str, str] = {}
    module_to_package: Dict[str, str] = {}
    module_to_defs: Dict[str, Set[str]] = {}
    file_to_module: Dict[str, str] = {}

    if repo_root is not None:
        sources = per_file_sources or {}
        repo_root_path = Path(repo_root).resolve()
        for py_file in repo_root_path.rglob("*.py"):
            rel_parts = py_file.relative_to(repo_root_path).parts
            if any(p in EXCLUDED_DIR_NAMES for p in rel_parts[:-1]):
                continue
            try:
                mod, pkg = file_to_module_and_package(py_file, repo_root_path)
            except ValueError:  # pragma: no cover
                continue
            abs_path = str(py_file.resolve())
            src = sources.get(abs_path)
            if src is None:
                try:
                    src = py_file.read_text(encoding="utf-8")
                except OSError:
                    continue
            module_to_source[mod] = src
            module_to_package[mod] = pkg
            module_to_defs[mod] = collect_defined_names(src)
            file_to_module[abs_path] = mod

    return RepoIndex(
        module_to_source=module_to_source,
        module_to_package=module_to_package,
        module_to_defs=module_to_defs,
        file_to_module=file_to_module,
    )
