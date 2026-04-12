from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import ast
import os
import threading
import time
from libcst.metadata import FullRepoManager, QualifiedNameProvider
import libcst as cst
from .core import _EXCLUDED_DIR_NAMES, _SCOPE_ANALYSIS_TIMEOUT


def _has_callers_outside_ranges(
    source: str, func_name: str, ranges: List[Tuple[int, int]]
) -> bool:
    """Return True if func_name is called at any line outside the given ranges."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == func_name
        ):
            line = node.lineno
            if not any(start <= line <= end for start, end in ranges):
                return True
    return False


def _blocked_private_scopes(source: str, ranges: List[Tuple[int, int]]) -> Set[str]:
    """Return names of private functions that have callers outside the diff ranges."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    blocked: Set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("_")
        ):
            line = node.lineno
            if not any(start <= line <= end for start, end in ranges):
                blocked.add(node.func.id)
    return blocked


def _find_repo_root(changed: Dict[str, List]) -> Optional[str]:
    """Find git repo root by searching parent directories for .git."""
    for filepath in changed.keys():
        p = Path(filepath).resolve().parent
        while p != p.parent:
            if (p / ".git").is_dir():
                return str(p)
            p = p.parent
    return None


def _file_to_module(repo_root: str, filepath: str) -> str:
    """Convert an absolute file path to a dotted Python module name."""
    path = Path(filepath).resolve().relative_to(Path(repo_root).resolve())
    module = str(path.with_suffix("")).replace(os.sep, ".")
    if module.endswith(".__init__"):
        module = module[:-9]
    return module


def _compute_qname(repo_root: str, filepath: str, func_name: str) -> str:
    """Compute the qualified name of a function defined in filepath."""
    return f"{_file_to_module(repo_root, filepath)}.{func_name}"


def _build_alias_map(repo_root: str, canonical_qnames: Set[str]) -> Dict[str, str]:
    """Map alias qualified names → canonical qualified names.

    Handles explicit re-exports like ``from .service import get_user`` in
    ``pkg/__init__.py``, which creates the alias ``pkg.get_user`` for the
    canonical name ``pkg.service.get_user``.
    """
    alias_map: Dict[str, str] = {q: q for q in canonical_qnames}

    for init_path in Path(repo_root).rglob("__init__.py"):
        pkg_parts = list(init_path.relative_to(repo_root).parts[:-1])
        pkg_qname = ".".join(pkg_parts)

        try:
            source = init_path.read_text(encoding="utf-8")
            tree = cst.parse_module(source)
        except Exception:
            continue

        for stmt in tree.body:
            if not isinstance(stmt, cst.SimpleStatementLine):
                continue
            for s in stmt.body:
                if not isinstance(s, cst.ImportFrom):
                    continue
                if isinstance(s.names, cst.ImportStar) or not isinstance(
                    s.names, (list, tuple)
                ):
                    continue
                for al in s.names:
                    if not isinstance(al, cst.ImportAlias) or not isinstance(
                        al.name, cst.Name
                    ):
                        continue  # pragma: no cover
                    func_name = al.name.value
                    alias_qname = f"{pkg_qname}.{func_name}" if pkg_qname else func_name
                    # Map this alias to a canonical qname if unambiguous
                    matches = [
                        c for c in canonical_qnames if c.split(".")[-1] == func_name
                    ]
                    if len(matches) == 1:
                        alias_map[alias_qname] = matches[0]

    return alias_map


class _CallerFinder(cst.CSTVisitor):
    """Visit a file and record which target qualified names are called."""

    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(self, target_qnames: Set[str]) -> None:
        self.target_qnames = target_qnames
        self.found: Set[str] = set()

    def visit_Call(self, node: cst.Call) -> None:
        qnames = self.get_metadata(QualifiedNameProvider, node.func, set())
        for qn in qnames:
            if qn.name in self.target_qnames:
                self.found.add(qn.name)


def _visit_with_timeout(wrapper, finder, timeout: float) -> bool:
    """Run wrapper.visit(finder) in a daemon thread with a wall-clock timeout.

    Returns True if the call completed within *timeout* seconds, False if it
    timed out (libcst scope analysis can hang on large files).
    """
    done = threading.Event()

    def _target():
        try:
            wrapper.visit(finder)
        finally:
            done.set()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return done.wait(timeout=timeout)


def _find_outside_callers(
    repo_root: str,
    target_qnames: Set[str],
    diff_files: Set[str],
) -> Set[str]:
    """Return the subset of *target_qnames* called in files outside *diff_files*."""
    if not target_qnames:
        return set()

    repo_root_path = Path(repo_root)
    outside_py = [
        p
        for p in repo_root_path.rglob("*.py")
        if str(p.resolve()) not in diff_files
        and not any(
            part in _EXCLUDED_DIR_NAMES
            for part in p.relative_to(repo_root_path).parts[:-1]
        )
    ]
    if not outside_py:
        return set()

    rel_paths = [str(p.relative_to(repo_root)) for p in outside_py]

    try:
        manager = FullRepoManager(repo_root, rel_paths, {QualifiedNameProvider})
    except Exception:
        # Can't build the manager → conservatively block all transforms.
        return set(target_qnames)

    found_outside: Set[str] = set()
    deadline = time.monotonic() + _SCOPE_ANALYSIS_TIMEOUT
    for rel_path in rel_paths:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Total budget exhausted: conservatively block all remaining.
            found_outside.update(target_qnames)
            break
        try:
            wrapper = manager.get_metadata_wrapper_for_path(rel_path)
            finder = _CallerFinder(target_qnames)
            if not _visit_with_timeout(wrapper, finder, remaining):
                # This file timed out: conservatively block all transforms.
                found_outside.update(target_qnames)
                break
            found_outside.update(finder.found)
        except Exception:
            continue

    return found_outside
