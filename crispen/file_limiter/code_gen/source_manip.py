from __future__ import annotations
from typing import Dict, List, Optional, Set
from ..entity_parser import Entity, EntityKind
from .import_utils import _EXCESS_BLANK_RE, _collect_name_loads
import ast


def _normalize_blank_lines(source: str) -> str:
    """Collapse runs of 3+ blank lines to 2; ensure exactly one trailing newline.

    Removes blank-line artefacts produced by entity removal (original file)
    and entity-source stripping (new files).  PEP 8 / flake8 E303 allows at
    most two blank lines between top-level definitions.
    """
    source = _EXCESS_BLANK_RE.sub("\n\n\n", source)
    return source.rstrip("\n") + "\n"


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


def _strip_top_level_import_lines(src: str) -> str:
    """Return *src* with all top-level import statements removed.

    Uses AST to locate the exact line range of each import node, correctly
    handling multi-line imports.  Returns *src* unchanged when it cannot be
    parsed as Python.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    remove: Set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for ln in range(node.lineno, node.end_lineno + 1):
                remove.add(ln)
    if not remove:
        return src
    lines = src.splitlines(keepends=True)
    return "".join(line for i, line in enumerate(lines, 1) if i not in remove)


def _extract_module_docstring(source: str) -> Optional[str]:
    """Return the module-level docstring source text, or None if absent."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if not (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        return None
    node = tree.body[0]
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno]).rstrip()


def _strip_module_docstring(src: str) -> str:
    """Return *src* with the leading module-level docstring removed."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    if not (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        return src
    node = tree.body[0]
    remove = set(range(node.lineno, node.end_lineno + 1))
    lines = src.splitlines(keepends=True)
    return "".join(line for i, line in enumerate(lines, 1) if i not in remove)


def _inject_inline_imports(entity_src: str, imports: List[str]) -> str:
    """Inject *imports* at the top of a function or class body in *entity_src*.

    The imports are inserted after any leading docstring.  Returns
    *entity_src* unchanged when it cannot be parsed or the top-level node
    is not a function or class (TOP_LEVEL entities have no body scope).
    """
    if not imports:
        return entity_src
    try:
        tree = ast.parse(entity_src)
    except SyntaxError:
        return entity_src
    if not tree.body:
        return entity_src
    top = tree.body[0]
    if not isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return entity_src
    first_stmt = top.body[0]
    insert_line = first_stmt.lineno
    if (
        isinstance(first_stmt, ast.Expr)
        and isinstance(first_stmt.value, ast.Constant)
        and isinstance(first_stmt.value.value, str)
        and len(top.body) > 1
    ):
        insert_line = top.body[1].lineno
    lines = entity_src.splitlines(keepends=True)
    body_line = lines[insert_line - 1]
    indent = body_line[: len(body_line) - len(body_line.lstrip())]
    import_lines = [f"{indent}{imp}\n" for imp in imports]
    return "".join(lines[: insert_line - 1] + import_lines + lines[insert_line - 1 :])
