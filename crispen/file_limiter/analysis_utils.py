from __future__ import annotations
import ast
from pathlib import Path
from typing import Dict, List, Optional, Set
from .dep_graph import find_sccs
from .entity_parser import Entity
from .project_paths import _find_project_root, _module_path_from_file


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


def _topo_depth(graph: Dict[str, Set[str]]) -> Dict[str, int]:
    """Return topological depth for each node in a DAG.

    Depth 0 = leaf (no outgoing edges).  A node's depth is 1 + the maximum
    depth of its dependencies.  All dependency nodes must be keys in *graph*.
    On non-DAG inputs (cycles detected), returns 0 for every node as a safe
    fallback so that callers degrade to arbitrary candidate ordering.
    """
    if any(len(s) > 1 for s in find_sccs(graph)):
        return {node: 0 for node in graph}
    depths: Dict[str, int] = {}

    def dfs(node: str) -> int:
        if node in depths:
            return depths[node]
        depths[node] = 1 + max((dfs(dep) for dep in graph[node]), default=-1)
        return depths[node]

    for node in graph:
        dfs(node)
    return depths


def _prune_inline_redundant_imports(source: str) -> str:
    """Remove function-body imports that duplicate module-level imports.

    When a function-local ``from x import y`` re-imports a name that is
    already provided by a top-level import, flake8 reports an F811
    redefinition warning.  This function removes such redundant inner imports
    (or narrows them when only some names are redundant).

    Returns *source* unchanged when it cannot be parsed or nothing needs
    pruning.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    # Names already available from top-level (module-level) imports.
    top_level_names: Set[str] = set()
    top_level_node_ids: Set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_node_ids.add(id(node))
            for alias in node.names:
                top_level_names.add(
                    alias.asname if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            top_level_node_ids.add(id(node))
            for alias in node.names:
                top_level_names.add(alias.asname if alias.asname else alias.name)

    if not top_level_names:
        return source

    # Find all import nodes that are NOT at module level.
    inner_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and id(node) not in top_level_node_ids
    ]

    if not inner_imports:
        return source

    lines = source.splitlines(keepends=True)
    # Maps 1-based line number → replacement line (None = remove that line).
    line_ops: Dict[int, Optional[str]] = {}

    for stmt in inner_imports:
        if isinstance(stmt, ast.Import):
            kept = [
                a
                for a in stmt.names
                if (a.asname if a.asname else a.name.split(".")[0])
                not in top_level_names
            ]
        else:
            kept = [
                a
                for a in stmt.names
                if (a.asname if a.asname else a.name) not in top_level_names
            ]

        if len(kept) == len(stmt.names):
            continue  # no redundancy — nothing to remove

        # Mark every line of this import for removal.
        for ln in range(stmt.lineno, stmt.end_lineno + 1):
            line_ops[ln] = None

        if kept:
            # Rebuild a narrowed import preserving original indentation.
            alias_strs = [
                f"{a.name} as {a.asname}" if a.asname else a.name for a in kept
            ]
            orig_line = lines[stmt.lineno - 1]
            indent = orig_line[: len(orig_line) - len(orig_line.lstrip())]
            if isinstance(stmt, ast.ImportFrom):
                dots = "." * (stmt.level or 0)
                mod = stmt.module or ""
                new_line = f"{indent}from {dots}{mod} import {', '.join(alias_strs)}\n"
            else:
                new_line = f"{indent}import {', '.join(alias_strs)}\n"
            line_ops[stmt.lineno] = new_line

    if not line_ops:
        return source

    result: List[str] = []
    for i, line in enumerate(lines, 1):
        if i in line_ops:
            repl = line_ops[i]
            if repl is not None:
                result.append(repl)
            # else: None → line is removed
        else:
            result.append(line)
    return "".join(result)


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
