from __future__ import annotations
from typing import List
import ast
import textwrap


def _collect_called_names(source: str) -> set:
    """Return a set of all names called (as functions) in *source*.

    Uses ast.parse + ast.walk to find all ast.Call nodes.  Returns the
    called name: func.id for ast.Name callees, func.attr for ast.Attribute
    callees.  On SyntaxError, returns an empty set.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


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


def _missing_free_vars(
    block_src: str, call_srcs: List[str], helper_src: str, source: str
) -> set:
    """Return locally-scoped free variable names from block_src absent from the
    replacement.

    Free variables are names that are *read* (appear in a ``Load`` context) but
    not locally *assigned* (``Store``/``Del``) within the original block.  To
    avoid false positives from module-level names (imported symbols, globally-
    defined functions) that the extracted helper can reference directly, the
    check is restricted to names that appear as assignment targets or function
    parameters somewhere in *source* — these are variables that live in a local
    scope and cannot be reached by the helper without being threaded through as
    arguments.

    After this filtering, every remaining name must appear as a bare ``Name``
    node somewhere in the call-site replacements or the helper body.  A name
    that vanishes from both indicates the LLM silently changed the data flow —
    for example by turning a local variable reference into an attribute access
    on one of the parameters (``new_source`` → ``transformer.new_source``).

    Returns the set of names that are absent from both.  An empty set means the
    check passes.  Returns an empty set on any ``SyntaxError`` so a parse
    failure does not block the extraction — the later ``compile()`` guard will
    catch real syntax problems.
    """
    try:
        block_tree = ast.parse(textwrap.dedent(block_src))
    except SyntaxError:
        return set()

    reads: set = set()
    stores: set = set()
    for node in ast.walk(block_tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                reads.add(node.id)
            else:
                stores.add(node.id)

    free_vars = reads - stores
    if not free_vars:
        return set()

    # Restrict to names that are locally assigned or are function/lambda
    # parameters somewhere in the full source.  Module-level names that are
    # only ever read (e.g. imported functions, global constants) are in scope
    # from the helper definition too and do not need to be passed as args.
    try:
        source_tree = ast.parse(source)
    except SyntaxError:
        return set()
    source_locals: set = set()
    for node in ast.walk(source_tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            source_locals.add(node.id)
        elif isinstance(node, ast.arg):
            source_locals.add(node.arg)

    free_vars = free_vars & source_locals
    if not free_vars:
        return set()

    replacement_names: set = set()
    for src in list(call_srcs) + [helper_src]:
        try:
            repl_tree = ast.parse(textwrap.dedent(src))
        except SyntaxError:
            return set()
        for node in ast.walk(repl_tree):
            if isinstance(node, ast.Name):
                replacement_names.add(node.id)

    return free_vars - replacement_names


def _helper_imports_local_name(helper_source: str, original_source: str) -> bool:
    """Return True if helper_source imports a name that is only a local in original.

    Detects the LLM mistake of writing ``import X`` in the helper when ``X``
    was a function parameter or other local name in the original file, not an
    importable module.  Such imports fail at runtime with ModuleNotFoundError.
    """
    try:
        helper_tree = ast.parse(textwrap.dedent(helper_source))
    except SyntaxError:
        return False

    helper_imports: set = set()
    for node in ast.walk(helper_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                helper_imports.add(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                helper_imports.add(name)

    if not helper_imports:
        return False

    try:
        orig_tree = ast.parse(original_source)
    except SyntaxError:
        return False

    # Names already imported at the top level of the original file.
    orig_top_imports: set = set()
    for node in orig_tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                orig_top_imports.add(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                orig_top_imports.add(name)

    new_helper_imports = helper_imports - orig_top_imports
    if not new_helper_imports:
        return False

    # Parameter names in the original file (potential mock-injected locals).
    orig_params: set = set()
    for node in ast.walk(orig_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                orig_params.add(arg.arg)
            if node.args.vararg:
                orig_params.add(node.args.vararg.arg)
            if node.args.kwarg:
                orig_params.add(node.args.kwarg.arg)

    return bool(new_helper_imports & orig_params)


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
