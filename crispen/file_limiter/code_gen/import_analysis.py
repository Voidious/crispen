from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import ast
import re


@dataclass
class ImportInfo:
    """A top-level import statement and the names it introduces."""

    names: List[str]  # names made available by this import
    source: str  # the import statement text (no trailing newline)
    is_future: bool  # True if `from __future__ import ...`
    is_type_checking: bool = False  # True if inside `if TYPE_CHECKING:` block


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
    """Return Name loads in *source* that are not shadowed by function parameters
    or local variable assignments.

    For each function or async function, names that appear as parameters of that
    function or are assigned anywhere in the function body are excluded from Name
    loads within its body.  This prevents generating spurious cross-file imports
    for names that are satisfied locally (e.g. pytest fixture names that appear as
    test function parameters, or local variables like ``helpers = tmp_path / ...``).

    Decorators, argument default values, and return/argument annotations are
    always evaluated in the outer scope and are never excluded.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: Set[str] = set()

    def _body_stores(stmts) -> frozenset:
        """Names stored/deleted at this scope level in *stmts*.

        Recurses into control-flow nodes (if/for/while/with/try) but stops at
        nested FunctionDef/AsyncFunctionDef/ClassDef scopes so only names that
        are local to the current function are returned.
        """
        stores: Set[str] = set()
        work = list(stmts)
        while work:
            node = work.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.Name) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                stores.add(node.id)
            work.extend(ast.iter_child_nodes(node))
        return frozenset(stores)

    def _walk(node: ast.AST, excluded: frozenset) -> None:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in excluded:
                names.add(node.id)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            own_params: frozenset = frozenset(
                a.arg
                for a in (
                    args.args
                    + args.posonlyargs
                    + args.kwonlyargs
                    + ([args.vararg] if args.vararg else [])
                    + ([args.kwarg] if args.kwarg else [])
                )
            )
            # Decorators are evaluated in the outer scope.
            for dec in node.decorator_list:
                _walk(dec, excluded)
            # Default values are evaluated in the outer scope.
            for default in args.defaults + args.kw_defaults:
                if default is not None:
                    _walk(default, excluded)
            # Annotations are in the outer scope (PEP 563 / regular annotations).
            for arg in args.args + args.posonlyargs + args.kwonlyargs:
                if arg.annotation:
                    _walk(arg.annotation, excluded)
            if args.vararg and args.vararg.annotation:
                _walk(args.vararg.annotation, excluded)
            if args.kwarg and args.kwarg.annotation:
                _walk(args.kwarg.annotation, excluded)
            if node.returns:
                _walk(node.returns, excluded)
            # Function body uses params + local stores as the excluded set.
            own_locals = _body_stores(node.body)
            new_excluded = excluded | own_params | own_locals
            for child in node.body:
                _walk(child, new_excluded)
            return
        for child in ast.iter_child_nodes(node):
            _walk(child, excluded)

    _walk(tree, frozenset())
    return names


def _collect_quoted_annotation_names(source: str) -> Set[str]:
    """Return names referenced inside quoted type annotations in *source*.

    Finds names like ``_LLMAccumulator`` in ``Optional["_LLMAccumulator"]``
    (string literals used as forward references in type annotations).  These
    names are only needed at type-checking time — not at runtime — and should
    be imported under ``if TYPE_CHECKING:`` rather than as regular imports.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: Set[str] = set()

    def _scan_annotation(node: ast.AST) -> None:
        """Recursively scan an annotation, extracting names from string constants."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                inner = ast.parse(node.value, mode="eval")
                for n in ast.walk(inner):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
            except SyntaxError:
                pass
            return
        for child in ast.iter_child_nodes(node):
            _scan_annotation(child)

    def _walk(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in args.args + args.posonlyargs + args.kwonlyargs:
                if arg.annotation:
                    _scan_annotation(arg.annotation)
            if args.vararg and args.vararg.annotation:
                _scan_annotation(args.vararg.annotation)
            if args.kwarg and args.kwarg.annotation:
                _scan_annotation(args.kwarg.annotation)
            if node.returns:
                _scan_annotation(node.returns)
            for child in node.body:
                _walk(child)
            return
        if isinstance(node, ast.AnnAssign):
            _scan_annotation(node.annotation)
            if node.value:
                _walk(node.value)
            return
        for child in ast.iter_child_nodes(node):
            _walk(child)

    _walk(tree)
    return names


def _collect_name_stores(source: str) -> Set[str]:
    """Return names assigned at module level in *source*.

    Detects ``x = ...``, ``x += ...``, and annotated assignments with a value
    (``x: int = ...``) at the top level of the module.  Used to identify
    TOP_LEVEL constants that are mutated outside their defining entity so that
    cross-file references must use ``module.NAME`` rather than a plain
    ``from .module import NAME``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    stores: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    stores.add(target.id)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                stores.add(node.target.id)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None and isinstance(node.target, ast.Name):
                stores.add(node.target.id)
    return stores


def _extract_import_info(source: str) -> List[ImportInfo]:
    """Return :class:`ImportInfo` for each top-level import in *source*.

    Also includes imports found inside module-level ``if TYPE_CHECKING:``
    blocks, marked with ``is_type_checking=True``.  These are used by
    :func:`_find_type_checking_needed_imports` to distribute forward-reference
    imports to the correct sub-files after a split.
    """
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
        elif (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for child in node.body:
                if isinstance(child, ast.Import):
                    tc_names = [
                        alias.asname if alias.asname else alias.name.split(".")[0]
                        for alias in child.names
                    ]
                    tc_src = "".join(
                        lines[child.lineno - 1 : child.end_lineno]
                    ).rstrip()
                    result.append(
                        ImportInfo(
                            names=tc_names,
                            source=tc_src,
                            is_future=False,
                            is_type_checking=True,
                        )
                    )
                elif isinstance(child, ast.ImportFrom):
                    tc_names = [
                        alias.asname if alias.asname else alias.name
                        for alias in child.names
                    ]
                    tc_dots = "." * (child.level or 0)
                    tc_mod = child.module or ""
                    tc_alias_strs = [
                        f"{a.name} as {a.asname}" if a.asname else a.name
                        for a in child.names
                    ]
                    tc_src = f"from {tc_dots}{tc_mod} import {', '.join(tc_alias_strs)}"
                    result.append(
                        ImportInfo(
                            names=tc_names,
                            source=tc_src,
                            is_future=False,
                            is_type_checking=True,
                        )
                    )

    return result


def _inject_type_checking_imports(source: str, imports: List[str]) -> str:
    """Add *imports* under a module-level ``if TYPE_CHECKING:`` guard in *source*.

    If a TYPE_CHECKING block already exists, new imports are appended to it
    (skipping any already present).  Otherwise a new block is inserted after
    the last top-level import statement, along with ``from typing import
    TYPE_CHECKING`` when that name is not already imported.
    """
    if not imports:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    # Determine which imports are not already in an existing TC block.
    existing_tc = {i.source for i in _extract_import_info(source) if i.is_type_checking}
    new_imports = [imp for imp in imports if imp not in existing_tc]
    if not new_imports:
        return source

    lines = source.splitlines(keepends=True)

    # Append to an existing TYPE_CHECKING block if one is present.
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            insert_line = node.end_lineno
            new_lines = ["    " + imp + "\n" for imp in sorted(new_imports)]
            return "".join(lines[:insert_line] + new_lines + lines[insert_line:])

    # No existing block: insert one after the last top-level import.
    last_import_line = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_line = max(last_import_line, node.end_lineno)
    insert_after = last_import_line

    tc_already_imported = any(
        isinstance(n, ast.ImportFrom)
        and n.module == "typing"
        and any((a.asname or a.name) == "TYPE_CHECKING" for a in n.names)
        for n in tree.body
    )
    new_lines = []
    if not tc_already_imported:
        new_lines.append("from typing import TYPE_CHECKING\n")
    new_lines.append("if TYPE_CHECKING:\n")
    for imp in sorted(new_imports):
        new_lines.append("    " + imp + "\n")
    new_lines.append("\n")
    return "".join(lines[:insert_after] + new_lines + lines[insert_after:])


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
        if info.is_type_checking:
            continue  # handled by _find_type_checking_needed_imports
        if info.is_future or any(n in referenced for n in info.names):
            needed.append(info.source)
            seen.add(info.source)

    return needed


def _narrow_import_source(import_src: str, keep_names: Set[str]) -> str:
    """Return a copy of *import_src* keeping only the exposed names in *keep_names*.

    For ``from X import A, B, C`` with ``keep_names={A}``, returns
    ``from X import A``.  Non-ImportFrom statements are returned unchanged.
    """
    try:
        node = ast.parse(import_src).body[0]
    except (SyntaxError, IndexError):
        return import_src
    if not isinstance(node, ast.ImportFrom):
        return import_src
    dots = "." * (node.level or 0)
    mod = node.module or ""
    alias_strs = [
        f"{a.name} as {a.asname}" if a.asname else a.name
        for a in node.names
        if (a.asname or a.name) in keep_names
    ]
    if not alias_strs:
        return import_src
    return f"from {dots}{mod} import {', '.join(alias_strs)}"


def _find_type_checking_needed_imports(
    entity_names: List[str],
    entity_source_map: Dict[str, str],
    import_infos: List[ImportInfo],
) -> List[str]:
    """Return import statements needed only for quoted type annotations.

    These should be placed under ``if TYPE_CHECKING:`` because the names are
    only referenced inside string-valued annotations (forward references) and
    are not needed at runtime.  Names that appear in regular (non-annotation)
    loads are excluded via ``annotation_only = quoted - runtime``, which
    guarantees that any name emitted here will be pruned from regular imports
    by ``_prune_unused_imports`` — so no duplicate imports can arise.
    ``__future__`` imports are always excluded since they are handled by
    ``_find_needed_imports``.
    """
    runtime: Set[str] = set()
    quoted: Set[str] = set()
    for name in entity_names:
        src = entity_source_map.get(name, "")
        runtime |= _collect_name_loads(src)
        quoted |= _collect_quoted_annotation_names(src)

    annotation_only = quoted - runtime
    if not annotation_only:
        return []

    needed: List[str] = []
    seen: Set[str] = set()
    for info in import_infos:
        if info.source in seen:
            continue
        if info.is_future:
            continue
        tc_names = {n for n in info.names if n in annotation_only}
        if not tc_names:
            continue
        # Narrow the import to only the names actually needed for type checking,
        # avoiding unused-import warnings for names from multi-name imports that
        # are not referenced in this file.
        src = (
            info.source
            if len(tc_names) == len(info.names)
            else _narrow_import_source(info.source, tc_names)
        )
        if src in seen:
            continue
        needed.append(src)
        seen.add(src)
    return needed


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

    # Collect import node IDs inside module-level 'if TYPE_CHECKING:' blocks.
    # These are intentional type-checking guards and must not be treated as
    # redundant even when the same name is already imported at module level —
    # removing them would leave an empty (and therefore invalid) if-block.
    tc_guard_import_ids: Set[int] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    tc_guard_import_ids.add(id(child))

    # Find all import nodes that are NOT at module level and NOT inside a
    # module-level 'if TYPE_CHECKING:' guard.
    inner_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and id(node) not in top_level_node_ids
        and id(node) not in tc_guard_import_ids
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

        # Preserve intentional re-export stubs added by _add_re_exports.
        # These carry "# noqa: F401" and must not be pruned even when the
        # name is no longer referenced in the file body — they exist solely
        # to keep the module's public/private API intact for external callers.
        import_lines = lines[node.lineno - 1 : node.end_lineno]
        if any("noqa: F401" in line for line in import_lines):
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
