from __future__ import annotations
from typing import Set
import ast


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
