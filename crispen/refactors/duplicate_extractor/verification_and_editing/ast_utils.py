from __future__ import annotations
from typing import List, Tuple
import ast
import re
import textwrap


def _seq_source_contains_yield(source: str) -> bool:
    """Return True if *source* contains ``yield`` or ``yield from`` outside
    any nested function definition.

    Sequences with a yield cannot be safely extracted into a plain helper
    function: extraction would make the helper a generator, forcing call sites
    to iterate via ``for``/``async for`` instead of calling it directly.  This
    is a semantic transformation (e.g. ``async with X as c: yield c`` →
    ``async for c in helper(): yield c``) that the extractor must not attempt.
    """
    wrapped = "def _f():\n" + textwrap.indent(textwrap.dedent(source), "    ")
    try:
        tree = ast.parse(wrapped)
    except SyntaxError:
        return False
    if not tree.body or not isinstance(
        tree.body[0], ast.FunctionDef
    ):  # pragma: no cover
        return False

    def _walk(nodes):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # don't cross into nested scope
            if isinstance(node, (ast.Yield, ast.YieldFrom)):
                return True
            if _walk(ast.iter_child_nodes(node)):
                return True
        return False

    return _walk(tree.body[0].body)


def _collect_ast_store_names(node: ast.AST, names: List[str]) -> None:
    """Recursively collect Name ids from an assignment target (Store context)."""
    if isinstance(node, ast.Name):
        names.append(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            _collect_ast_store_names(elt, names)


def _replace_unused_in_target(
    target: ast.AST, following_src: str
) -> Tuple[ast.AST, bool, bool]:
    """Replace unused Name nodes in *target* with ``_``.

    Returns ``(new_target, all_replaced, any_replaced)`` where:
    - *all_replaced*: every name in the target was replaced (all unused).
    - *any_replaced*: at least one name was replaced.

    Non-Name, non-Tuple/List targets (Attribute, Subscript, …) are treated as
    *used* so we never accidentally strip an assignment we cannot analyse.
    """
    if isinstance(target, ast.Name):
        if re.search(r"\b" + re.escape(target.id) + r"\b", following_src):
            return target, False, False  # used → keep
        return ast.Name(id="_", ctx=ast.Store()), True, True  # unused → _
    if isinstance(target, (ast.Tuple, ast.List)):
        new_elts: List[ast.AST] = []
        all_replaced = True
        any_replaced = False
        for elt in target.elts:
            new_elt, elt_all, elt_any = _replace_unused_in_target(elt, following_src)
            new_elts.append(new_elt)
            if not elt_all:
                all_replaced = False
            if elt_any:
                any_replaced = True
        new_target = type(target)(elts=new_elts, ctx=ast.Store())
        return new_target, all_replaced, any_replaced
    # Attribute, Subscript, Starred, etc. — treat as used.
    return target, False, False


_MUTABLE_CONSTRUCTORS = frozenset({"set", "list", "dict", "frozenset", "bytearray"})


def _collect_attribute_names(source: str) -> set:
    """Return all attribute names (dot-access names) anywhere in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def _collect_called_attr_names(source: str) -> set:
    """Return attribute names used as method calls in *source*.

    Unlike :func:`_collect_attribute_names`, this only returns names that
    appear as the attribute of a call expression (i.e. ``obj.method(...)``).
    Plain attribute reads and type annotations like ``ast.AST`` are ignored,
    so the new-method-call check does not produce false positives for
    standard-library type references.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def _has_funcdef(func_name: str, source: str) -> bool:
    """Return True if func_name is defined anywhere in source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        ):
            return True
    return False


def _has_call_to(func_name: str, source: str) -> bool:
    """Return True if func_name is called anywhere in source.

    Checks both direct calls (``func_name(...)``) and attribute calls
    (``obj.func_name(...)``), covering both module-level helpers and
    staticmethod calls.  Returns False if source cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == func_name:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
            return True
    return False


def _is_pure_literal(node: ast.expr) -> bool:
    """Return True if *node* is a side-effect-free literal expression.

    Covers ``ast.Constant`` (numbers, strings, bytes, True/False/None) and
    recursively-pure container literals (list, tuple, set, dict).  Anything
    involving a function call or attribute access returns False.
    """
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_pure_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (k is None or _is_pure_literal(k)) and _is_pure_literal(v)
            for k, v in zip(node.keys, node.values)
        )
    return False


def _names_assigned_in(block_source: str) -> set:
    """Return names assigned at the top level of block_source.

    Covers bare ``x = ...`` (ast.Assign) and augmented ``x += ...``
    (ast.AugAssign) statements only; other assignment forms are ignored.
    """
    try:
        tree = ast.parse(textwrap.dedent(block_source))
    except SyntaxError:
        return set()
    names: set = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for n in ast.walk(target):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(node, ast.AugAssign):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    names.add(n.id)
    return names


def _extract_defined_names(source: str) -> set:
    """Return all function and class names defined anywhere in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
