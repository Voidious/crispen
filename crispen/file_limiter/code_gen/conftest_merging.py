from __future__ import annotations
from typing import Dict, List, Set, Tuple
import ast


def _merge_conftest_sources(existing: str, new_content: str) -> str:
    """Merge *new_content* into an existing conftest.py without duplicating anything.

    When multiple file splits each contribute fixtures to the same conftest.py,
    naively appending produces duplicate import statements, duplicate function
    definitions, and E402 errors (imports after function definitions).

    This function avoids all three:
    - Duplicate import statements (same module + same names) are skipped.
    - Function/class definitions whose names already appear in *existing* are skipped.
    - Non-duplicate imports from *new_content* are inserted after the last existing
      import (before any existing function definitions), preventing E402.
    - Non-duplicate definitions are appended at the end.

    Falls back to simple concatenation when either source cannot be parsed.
    """
    try:
        existing_tree = ast.parse(existing)
        new_tree = ast.parse(new_content)
    except SyntaxError:
        return existing.rstrip() + "\n\n\n" + new_content

    existing_lines = existing.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    def _import_key(node: ast.stmt) -> str:
        if isinstance(node, ast.Import):
            return "I:" + ",".join(
                sorted(f"{a.name}:{a.asname or ''}" for a in node.names)
            )
        assert isinstance(node, ast.ImportFrom)
        dots = "." * (node.level or 0)
        mod = node.module or ""
        return (
            "F:"
            + dots
            + mod
            + ":"
            + ",".join(sorted(f"{a.name}:{a.asname or ''}" for a in node.names))
        )

    # What is already in existing?
    existing_import_keys: Set[str] = {
        _import_key(n)
        for n in existing_tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    }
    existing_defined_names: Set[str] = {
        n.name
        for n in existing_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    # Last import line in existing (0-indexed insertion point).
    last_import_lineno: int = 0
    for n in existing_tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            last_import_lineno = max(last_import_lineno, n.end_lineno)

    # Collect new, non-duplicate imports and definitions from new_content.
    imports_to_insert: List[str] = []
    defs_to_append: List[str] = []

    for node in new_tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if _import_key(node) not in existing_import_keys:
                src = "".join(new_lines[node.lineno - 1 : node.end_lineno]).rstrip()
                imports_to_insert.append(src)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name not in existing_defined_names:
                first_line = (
                    node.decorator_list[0].lineno
                    if node.decorator_list
                    else node.lineno
                )
                src = "".join(new_lines[first_line - 1 : node.end_lineno]).rstrip()
                defs_to_append.append(src)

    if not imports_to_insert and not defs_to_append:
        return existing

    result_lines = list(existing_lines)
    if imports_to_insert:
        # Insert new imports directly after the last existing import line.
        insert_at = last_import_lineno  # 0-indexed position after last import
        new_import_lines = [imp + "\n" for imp in imports_to_insert]
        result_lines = (
            result_lines[:insert_at] + new_import_lines + result_lines[insert_at:]
        )

    result = "".join(result_lines).rstrip()
    if defs_to_append:
        result = result + "\n\n\n" + "\n\n\n".join(defs_to_append) + "\n"
    else:
        result = result + "\n"
    return result


def _rewrite_module_var_names(src: str, rewrites: Dict[str, str]) -> str:
    """Replace bare ``Name`` loads with ``module.name`` attribute accesses.

    Uses the AST to locate exact positions of ``Name`` load nodes whose
    ``id`` is in *rewrites*, replacing each with its qualified form (e.g.
    ``"SAFE_MODE"`` → ``"conversion.SAFE_MODE"``).

    Because ``ast.Name`` nodes **never** represent the attribute part of an
    ``Attribute`` node (which stores ``attr`` as a plain string), this
    approach is immune to the corruption that a regex would cause on
    ``obj.SAFE_MODE`` and naturally skips string literals and comments.

    After rewriting, the result is re-parsed and every original ``Name``
    load for each rewritten identifier is verified to be absent.  If
    verification fails the original source is returned unchanged so that
    callers can fall back to direct-import semantics rather than corrupt
    the output.
    """
    if not rewrites:
        return src
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src

    lines = src.splitlines(keepends=True)

    # Collect (lineno, col_offset, end_col_offset, new_text).
    # ast uses 1-indexed lineno and 0-indexed col_offset / end_col_offset.
    edits: List[Tuple[int, int, int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in rewrites
        ):
            edits.append(
                (node.lineno, node.col_offset, node.end_col_offset, rewrites[node.id])
            )

    if not edits:
        return src

    # Apply edits from last to first within each line to keep earlier offsets valid.
    edits.sort(key=lambda e: (e[0], e[1]), reverse=True)
    for lineno, col_start, col_end, new_text in edits:
        line = lines[lineno - 1]
        lines[lineno - 1] = line[:col_start] + new_text + line[col_end:]

    result = "".join(lines)

    # Verification: re-parse and confirm no bare Name loads remain for any
    # rewritten identifier.  If the result is unparseable or a bare name
    # survives, return the original source to avoid corrupting the output.
    try:
        new_tree = ast.parse(result)
    except SyntaxError:
        return src
    for node in ast.walk(new_tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in rewrites
        ):
            return src

    return result


def _rewrite_module_level_stores(src: str, rewrites: Dict[str, str]) -> str:
    """Rewrite module-level Name store targets to ``module.name`` attribute stores.

    Only statements at the top level of the module are affected
    (``ast.Module.body``).  Assignments inside function or class bodies are
    left unchanged so that local variable bindings are not corrupted.

    Used for the non-migrated home file: when a non-migrated entity reassigns
    a TOP_LEVEL constant that was moved to a sub-file, the assignment must be
    rewritten as ``module.CONST = expr`` so that the mutation updates the
    canonical value in the sub-file rather than creating an orphaned local
    binding.
    """
    if not rewrites:
        return src
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    lines = src.splitlines(keepends=True)
    edits: List[Tuple[int, int, int, str]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in rewrites:
                    edits.append(
                        (
                            target.lineno,
                            target.col_offset,
                            target.end_col_offset,
                            rewrites[target.id],
                        )
                    )
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id in rewrites:
                edits.append(
                    (
                        node.target.lineno,
                        node.target.col_offset,
                        node.target.end_col_offset,
                        rewrites[node.target.id],
                    )
                )
        elif isinstance(node, ast.AnnAssign):
            if (
                node.value is not None
                and isinstance(node.target, ast.Name)
                and node.target.id in rewrites
            ):
                edits.append(
                    (
                        node.target.lineno,
                        node.target.col_offset,
                        node.target.end_col_offset,
                        rewrites[node.target.id],
                    )
                )
    if not edits:
        return src
    edits.sort(key=lambda e: (e[0], e[1]), reverse=True)
    for lineno, col_start, col_end, new_text in edits:
        line = lines[lineno - 1]
        lines[lineno - 1] = line[:col_start] + new_text + line[col_end:]
    return "".join(lines)
