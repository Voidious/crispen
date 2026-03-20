from __future__ import annotations
from typing import List, Optional
import ast
import re
import textwrap
from ..core import _FunctionInfo, _SeqInfo
from .ast_and_name_utils import (
    _collect_ast_store_names,
    _is_pure_literal,
    _names_assigned_in,
)


_MUTABLE_CONSTRUCTORS = frozenset({"set", "list", "dict", "frozenset", "bytearray"})


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


def _pyflakes_strip_unused_simple_assigns(source: str, allowed_names: set) -> str:
    """Remove simple literal initializations that became unused after extraction.

    Only considers assignments whose target name is in *allowed_names* — the
    set of variable names that the extraction actually touched.  This prevents
    the cleaner from making unrelated changes to variables that were already
    unused before the extraction ran.

    Runs pyflakes ``UnusedVariable`` (F841) detection on *source* and strips
    any ``Assign`` statement whose right-hand side is a pure literal (no
    function calls, no attribute accesses), so we never discard side effects.

    A ``compile()`` check guards against the rare case where the removed line
    was the only statement in its block — if the result is invalid Python the
    original source is returned unchanged.
    """
    import pyflakes.api
    import pyflakes.messages

    class _Collector:
        def __init__(self):
            self.linenos: set = set()

        def unexpectedError(self, filename, msg):  # pragma: no cover
            pass

        def syntaxError(self, filename, msg, lineno, offset, text):  # pragma: no cover
            pass

        def flake(self, msg):
            if isinstance(msg, pyflakes.messages.UnusedVariable):
                self.linenos.add(msg.lineno)

    reporter = _Collector()
    pyflakes.api.check(source, "<candidate>", reporter=reporter)
    if not reporter.linenos:
        return source

    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover
        return source  # pragma: no cover

    lines_to_remove: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if node.lineno not in reporter.linenos:
            continue
        # Restrict to names the extraction actually touched.
        assigned: List[str] = []
        _collect_ast_store_names(node.targets[0], assigned)
        if not assigned or not set(assigned).issubset(allowed_names):
            continue
        if _is_pure_literal(node.value):
            lines_to_remove.update(range(node.lineno, node.end_lineno + 1))

    if not lines_to_remove:
        return source

    lines = source.splitlines(keepends=True)
    cleaned = "".join(
        line for i, line in enumerate(lines, 1) if i not in lines_to_remove
    )
    try:
        compile(cleaned, "<stripped>", "exec")
    except SyntaxError:
        return source
    return cleaned


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


def _seq_ends_with_return(seq: _SeqInfo) -> bool:
    """Return True if the last top-level statement is a non-None return.

    Detects the case where the LLM includes a ``return`` statement inside the
    duplicate block but the generated replacement omits it, producing a
    function that silently returns ``None`` instead of the original value.

    Bare ``return`` and ``return None`` are excluded: both are semantically
    equivalent to falling off the end of a function, so dropping them in a
    replacement causes no behavioral change.
    """
    try:
        tree = ast.parse(textwrap.dedent(seq.source))
    except SyntaxError:
        return False
    if not tree.body:
        return False
    last = tree.body[-1]
    if not isinstance(last, ast.Return):
        return False
    # Bare `return` and `return None` are equivalent to implicit None return.
    if last.value is None:
        return False
    if isinstance(last.value, ast.Constant) and last.value.value is None:
        return False
    return True


def _replacement_contains_return(replacement: str) -> bool:
    """Return True if *replacement* contains any return statement.

    Wraps the replacement in a dummy function before parsing so that
    ``return`` statements — which are legal inside a function body — do not
    cause false SyntaxError rejections.
    """
    try:
        wrapped = "def _check():\n" + textwrap.indent(
            textwrap.dedent(replacement), "    "
        )
        tree = ast.parse(wrapped)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            return True
    return False


def _replacement_steals_post_block_line(
    group: List[_SeqInfo], call_replacements: List[str], source_lines: List[str]
) -> bool:
    """Return True if any replacement's last line duplicates the line after its block.

    The LLM occasionally appends the first statement *after* the replaced block
    to the end of the replacement text.  When applied, that statement then appears
    twice in the assembled output: once inside the replacement and once as the
    original untouched line.
    """
    for seq, replacement in zip(group, call_replacements):
        next_idx = seq.end_line  # 0-based index of the first line after the block
        # Scan forward past blank lines to find the first real post-block line.
        while next_idx < len(source_lines) and not source_lines[next_idx].strip():
            next_idx += 1
        if next_idx >= len(source_lines):
            continue
        post_block = source_lines[next_idx].strip()
        repl_lines = [ln.strip() for ln in replacement.splitlines() if ln.strip()]
        if repl_lines and repl_lines[-1] == post_block:
            return True
    return False


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


def _find_escaping_vars(group: List[_SeqInfo], source_lines: List[str]) -> set:
    """Return names assigned in any group sequence that are referenced after it.

    A variable "escapes" when the block assigns it and subsequent code in the
    same scope (at the same or deeper indentation level) references it.
    The helper must return these variables so callers that need them can
    capture the return value.
    """
    escaping: set = set()
    for seq in group:
        block_src = "".join(source_lines[seq.start_line - 1 : seq.end_line])
        assigned = _names_assigned_in(block_src)
        if not assigned:
            continue

        # Infer the block's indentation level from its first non-empty line.
        first_line = next(
            (
                ln
                for ln in source_lines[seq.start_line - 1 : seq.end_line]
                if ln.strip()
            ),
            "",
        )
        block_indent = len(first_line) - len(first_line.lstrip())

        # Collect lines that follow the block within the same scope.
        # For indented blocks: stop when indentation falls below block_indent.
        # For module-level (indent 0): stop at the next def/class statement.
        after_lines: List[str] = []
        for line in source_lines[seq.end_line :]:
            if not line.strip():
                after_lines.append(line)
                continue
            line_indent = len(line) - len(line.lstrip())
            if block_indent == 0:
                if re.match(r"def |class ", line):
                    break
            elif line_indent < block_indent:
                break
            after_lines.append(line)

        if not after_lines:
            continue

        after_src = "".join(after_lines)
        try:
            after_tree = ast.parse(textwrap.dedent(after_src))
        except SyntaxError:
            continue

        used_after = {n.id for n in ast.walk(after_tree) if isinstance(n, ast.Name)}
        escaping |= assigned & used_after

    return escaping


def _would_create_proxy_wrappers(
    group: List[_SeqInfo], all_functions: List[_FunctionInfo]
) -> bool:
    """Return True if extracting this group would leave *some but not all* members
    as trivial proxy wrappers.

    A function becomes a trivial proxy wrapper when its entire body is the
    extracted block — after extraction it would contain only a single call to
    the new helper, with no meaningful logic of its own.

    When *every* member of the group would become a proxy, extraction is still
    worthwhile: all functions delegate to the same helper, which eliminates the
    duplication.  The problematic case is a mixed group where some members lose
    all their logic while others keep meaningful bodies.
    """
    proxy_count = 0
    non_module_count = 0
    for seq in group:
        if seq.scope == "<module>":
            continue
        non_module_count += 1
        func_outer_scope = (
            seq.class_scope if seq.class_scope is not None else "<module>"
        )
        for func in all_functions:
            if func.name == seq.scope and func.scope == func_outer_scope:
                if len(seq.stmts) == func.body_stmt_count:
                    proxy_count += 1
                break
    return 0 < proxy_count < non_module_count
