from __future__ import annotations
import ast
import textwrap
from typing import List, Optional
from .analysis_parsing import _parse_source_and_init_names
from .blocks import _MUTABLE_CONSTRUCTORS
from .helpers import _parse_dedented_block


def _has_mutable_literal_is_check(source: str) -> bool:
    """Return True if *source* contains identity checks against mutable literals.

    Patterns like ``x is set()``, ``x is []``, or ``x is {}`` are always
    False in Python because each literal creates a new object at runtime.
    Such patterns are a common LLM mistake when using a ``set()`` sentinel.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for op, comp in zip(node.ops, node.comparators):
            if not isinstance(op, (ast.Is, ast.IsNot)):
                continue
            if isinstance(comp, (ast.List, ast.Set, ast.Dict, ast.Tuple)):
                return True
            if (
                isinstance(comp, ast.Call)
                and isinstance(comp.func, ast.Name)
                and comp.func.id in _MUTABLE_CONSTRUCTORS
            ):
                return True
    return False


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


def _has_param_overwritten_before_read(helper_source: str) -> bool:
    """Return True if any parameter is assigned before it is first read.

    This detects a common LLM mistake where a parameter is included in the
    function signature but then immediately overwritten on the first line,
    making the parameter useless and causing UnboundLocalError at call sites
    that try to pass a value that was not yet assigned.
    """
    try:
        tree = ast.parse(textwrap.dedent(helper_source))
    except SyntaxError:  # pragma: no cover
        return False  # pragma: no cover
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {arg.arg for arg in node.args.args}
        params |= {arg.arg for arg in node.args.posonlyargs}
        params |= {arg.arg for arg in node.args.kwonlyargs}
        if node.args.vararg:
            params.add(node.args.vararg.arg)
        if node.args.kwarg:
            params.add(node.args.kwarg.arg)
        for stmt in node.body:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Name) and n.id in params:
                    if isinstance(n.ctx, ast.Store):
                        return True
                    params.discard(n.id)  # first use is a read — param is legitimate
    return False


def _verify_extraction(
    helper_source: Optional[str], call_replacements: List[str]
) -> bool:
    """Verify the extraction produces syntactically valid Python.

    Replacements are dedented and then wrapped in a dummy function before
    compilation so that ``return`` / ``yield`` statements — which are legal
    inside a function body — do not cause false SyntaxError rejections.
    Pass helper_source=None to skip the helper compilation check (used when
    replacing with an existing function rather than a newly extracted one).
    """
    if helper_source is not None:
        dedented_helper = textwrap.dedent(helper_source)
        try:
            compile(dedented_helper, "<helper>", "exec")
        except SyntaxError:
            return False
        if _has_param_overwritten_before_read(helper_source):
            return False
        # Dedent before checking: helper may be indented (e.g. staticmethod).
        # compile() already confirmed it's valid Python, so ast.parse will succeed.
        if _has_mutable_literal_is_check(dedented_helper):
            return False
    for replacement in call_replacements:
        dedented = textwrap.dedent(replacement)
        # Wrap in a dummy function that contains a for loop so that
        # ``return`` / ``yield`` (valid inside a function body) AND
        # ``continue`` / ``break`` (valid inside a loop body) do not cause
        # false SyntaxError rejections.  Replacements are always placed back
        # inside the caller's original context, which may include a loop.
        wrapped = "def _check():\n    for _ in []:\n" + textwrap.indent(
            dedented, "        "
        )
        try:
            compile(wrapped, "<replacement>", "exec")
        except SyntaxError:
            # Retry with async wrapper for replacements that contain `await`
            async_wrapped = "async def _check():\n    for _ in []:\n" + textwrap.indent(
                dedented, "        "
            )
            try:
                compile(async_wrapped, "<replacement>", "exec")
            except SyntaxError:
                return False
            wrapped = async_wrapped
        # Check the wrapped form so that indented/return-containing replacements
        # parse successfully and give a definitive True/False answer.
        if _has_mutable_literal_is_check(wrapped):
            return False
    return True


def _pyflakes_new_undefined_names(original: str, candidate: str) -> set:
    """Return undefined names (F821) introduced by the edit.

    Compares pyflakes output before and after the edit and returns only names
    that are newly undefined in the candidate — not ones already present in the
    original source. This avoids false positives from pre-existing bare function
    calls or module-level references that are valid in context but not resolvable
    from a standalone snippet.
    """
    import pyflakes.api
    import pyflakes.messages

    class _Collector:
        def __init__(self):
            self.names: set = set()

        def unexpectedError(self, filename, msg):  # pragma: no cover
            pass

        def syntaxError(self, filename, msg, lineno, offset, text):  # pragma: no cover
            pass

        def flake(self, msg):
            if isinstance(msg, pyflakes.messages.UndefinedName):
                self.names.add(msg.message_args[0])

    before = _Collector()
    pyflakes.api.check(original, "<original>", reporter=before)
    after = _Collector()
    pyflakes.api.check(candidate, "<rewritten>", reporter=after)
    return after.names - before.names


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
    block_tree = _parse_dedented_block(block_src)
    if block_tree is None:
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
    parsed = _parse_source_and_init_names(source)
    if parsed is None:
        return set()
    source_tree, source_locals = parsed
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
