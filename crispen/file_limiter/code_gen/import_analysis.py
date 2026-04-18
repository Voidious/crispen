from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set
import ast


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


def _inject_module_level_imports(source: str, imports: List[str]) -> str:
    """Insert *imports* after the last existing import line in *source*.

    Uses the same insertion logic as :func:`_add_re_exports` so that module
    imports for reassigned TOP_LEVEL variables land in the same position as
    other imports added to the original file.
    """
    if not imports:
        return source
    lines = source.splitlines(keepends=True)
    last_import_line = 0
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "\n".join(sorted(imports)) + "\n\n" + source
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_line = max(last_import_line, node.end_lineno)
    insert_after = last_import_line
    if insert_after == 0 and tree.body:
        first = tree.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            insert_after = first.end_lineno
    import_lines = [imp + "\n" for imp in sorted(imports)]
    return "".join(lines[:insert_after] + import_lines + lines[insert_after:])


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


def _test_names_in_decorators(source: str, names: Set[str]) -> Set[str]:
    """Return the subset of *names* that appear as Name loads inside a decorator.

    Decorators are evaluated before function bodies run, so a symbol that
    only reaches a file via an inline import (injected into the function body)
    will not be in scope when the decorator is evaluated.  This helper detects
    that situation so callers can abort the split rather than generate broken
    code.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                for child in ast.walk(dec):
                    if (
                        isinstance(child, ast.Name)
                        and isinstance(child.ctx, ast.Load)
                        and child.id in names
                    ):
                        found.add(child.id)
    return found


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
