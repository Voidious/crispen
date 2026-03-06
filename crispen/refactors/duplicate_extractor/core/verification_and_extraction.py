from __future__ import annotations
import ast
import re
import textwrap
from typing import List, Optional, Tuple
import libcst as cst
from .sequence_analysis import _MUTABLE_CONSTRUCTORS, _SeqInfo


def _strip_helper_docstring(helper_source: str) -> str:
    """Remove the docstring from helper_source if the first function has one."""
    try:
        tree = cst.parse_module(textwrap.dedent(helper_source))
    except cst.ParserSyntaxError:
        return helper_source

    if not tree.body or not isinstance(tree.body[0], cst.FunctionDef):
        return helper_source

    func = tree.body[0]
    body = func.body
    if not isinstance(body, cst.IndentedBlock) or not body.body:  # pragma: no cover
        return helper_source

    first = body.body[0]
    if not (
        isinstance(first, cst.SimpleStatementLine)
        and len(first.body) == 1
        and isinstance(first.body[0], cst.Expr)
        and isinstance(first.body[0].value, (cst.SimpleString, cst.ConcatenatedString))
    ):
        return helper_source

    rest = list(body.body[1:])
    if not rest:
        return helper_source

    new_func = func.with_changes(body=body.with_changes(body=rest))
    return tree.with_changes(body=[new_func] + list(tree.body[1:])).code


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


def _build_helper_insertion(
    source_lines: List[str],
    insert_pos: int,
    helper_source: str,
    placement: str,
) -> Tuple[int, int, str]:
    """Build an edit tuple that inserts helper_source with correct surrounding blanks.

    Absorbs existing blank lines around the insertion point so the result has
    exactly 2 blank lines before and after module-level helpers, or 1 blank
    line for staticmethod insertions inside a class body.
    """
    blank_lines = 1 if placement.startswith("staticmethod:") else 2

    # Count consecutive blank lines immediately before insert_pos.
    before_blanks = 0
    i = insert_pos - 1
    while i >= 0 and not source_lines[i].strip():
        before_blanks += 1
        i -= 1

    # Count consecutive blank lines at and immediately after insert_pos.
    after_blanks = 0
    i = insert_pos
    while i < len(source_lines) and not source_lines[i].strip():
        after_blanks += 1
        i += 1

    # Replace surrounding blank lines so we don't double-count them.
    start = insert_pos - before_blanks
    end = insert_pos + after_blanks
    clean = helper_source.strip("\n") + "\n"
    text = "\n" * blank_lines + clean + "\n" * blank_lines
    return (start, end, text)


def _find_insertion_point(source: str, scope: str) -> int:
    """Return 0-based line index to insert before.

    For module scope, inserts after the last import.
    For a named scope, inserts before the def/class line.

    If the named scope resolves to an indented ``def`` (i.e. a class method),
    inserting a module-level helper immediately before it would end the class
    definition prematurely — the remaining class methods would be silently
    re-parsed as nested functions of the helper, producing valid-syntax but
    broken code that ``compile()`` does not catch.  In that case we walk
    backwards to the enclosing class definition and insert before it instead.
    """
    source_lines = source.splitlines()
    if scope == "<module>":
        last_import = -1
        for i, line in enumerate(source_lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                last_import = i
        return last_import + 1

    pattern = re.compile(rf"^\s*(?:async\s+def|def|class)\s+{re.escape(scope)}\s*[\(:]")
    for i, line in enumerate(source_lines):
        if pattern.match(line):
            method_indent = len(line) - len(line.lstrip())
            if method_indent > 0:
                # The def is inside a class body.  Walk backwards to find the
                # enclosing class definition and insert before that instead.
                # If the first lower-indent non-blank line is NOT a class
                # definition (i.e. the def is a nested function inside a
                # regular function), stop immediately so we don't mis-identify
                # an unrelated class above the outer function as the enclosing
                # class.
                for j in range(i - 1, -1, -1):
                    prev = source_lines[j]
                    if not prev.strip():
                        continue
                    prev_indent = len(prev) - len(prev.lstrip())
                    if prev_indent < method_indent:
                        if re.match(r"\s*class\s+\w+", prev):
                            return j
                        break  # nested function — fall through to decorator walk
            # Walk backwards over any preceding decorator lines (including
            # multi-line decorator arguments) so the helper is inserted
            # before the decorator block, not between decorators and the def.
            j = i - 1
            paren_depth = 0
            while j >= 0:
                stripped = source_lines[j].strip()
                if not stripped:
                    break
                for ch in stripped:
                    if ch == ")":
                        paren_depth += 1
                    elif ch == "(":
                        paren_depth -= 1
                if paren_depth == 0 and not stripped.startswith("@"):
                    break
                j -= 1
            return j + 1
    return 0
