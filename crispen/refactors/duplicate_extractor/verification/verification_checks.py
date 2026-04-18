from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import ast
import re
import textwrap
from ....import_sort import _sort_imports_pep8
from .ast_and_replacement_utils import _collect_ast_store_names, _is_pure_literal


def _scope_end_line(source_lines: List[str], scope: str, after_line: int) -> int:
    """Return the exclusive slice index into *source_lines* for the end of *scope*.

    ``after_line`` is the 1-based line number of the last line of the replaced
    block.  The returned index is suitable for ``source_lines[after_line:idx]``
    to get only the lines inside the enclosing scope that follow the block.

    For ``"<module>"`` scope the whole rest of the file is in scope, so
    ``len(source_lines)`` is returned.  For named function/class scopes the
    innermost definition whose name matches *scope* and that contains
    *after_line* is located via the AST; its end line is returned as the
    exclusive slice bound (1-based end_lineno used directly as a 0-based
    exclusive index is correct because line N is at index N-1, so slicing up
    to index N includes line N).  Falls back to ``len(source_lines)`` on any
    parse error or if no matching scope is found.
    """
    if scope == "<module>":
        return len(source_lines)

    source = "".join(source_lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return len(source_lines)

    # ast.walk is BFS, so outer scopes are visited before inner ones.  Always
    # overwriting best_end means the last match wins — which is the innermost
    # (smallest) scope that still contains after_line.
    best_end: int = len(source_lines)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name != scope:
            continue
        if not (node.lineno <= after_line <= node.end_lineno):
            continue
        best_end = node.end_lineno

    return best_end


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


def _names_in_edit_texts(extraction_groups) -> set:
    """Return all bare ``Name`` ids found in every edit text of *extraction_groups*.

    ``extraction_groups`` is the list of ``(func_name, group_edits, msg)``
    tuples accepted at the end of ``DuplicateExtractor._transform``.  Each
    ``group_edits`` entry is a ``(start, end, text)`` triple; *text* may be
    the helper function source or a call-site replacement.  Collecting names
    from all of them gives the set of variables that the extraction actually
    touched.
    """
    names: set = set()
    for _, g_edits, _ in extraction_groups:
        for _start, _end, text in g_edits:
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    names.add(node.id)
    return names


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


def _lift_and_dedup_imports(source: str) -> str:
    """Lift misplaced module-level imports to the import block and deduplicate.

    When a helper is inserted before a function that is not the first in the
    file, its leading ``from X import Y`` lines land after the first
    ``def``/``class``, violating PEP 8.  When a helper re-imports names
    already present at the top, flake8 reports F811.  This function fixes both:

    1. Collect every simple, unindented ``from X import …`` / ``import X``
       line from anywhere in the file.
    2. Merge names for the same module (deduplicate).
    3. Emit the merged set within the top-of-file import block (before the
       first ``def``/``class``), removing all later occurrences.

    Only single-line imports without parentheses, backslash continuations, or
    inline comments are handled.  Indented imports (``if TYPE_CHECKING:``,
    function-local lazy imports, etc.) and wildcard imports are left untouched.
    """
    lines = source.splitlines(keepends=True)
    n = len(lines)

    # ── pass 1: find the import block boundary ──────────────────────────────
    # The import block ends at the first unindented def/class line.
    first_funcdef_idx = n
    for i, line in enumerate(lines):
        if line[:1] in (" ", "\t"):
            continue
        if re.match(r"^(?:async\s+def|def|class)\s", line.strip()):
            first_funcdef_idx = i
            break

    # ── pass 2: collect simple unindented import lines ──────────────────────
    _FROM_RE = re.compile(r"^from\s+(\S+)\s+import\s+([^(\\#]+)$")
    _PLAIN_RE = re.compile(r"^import\s+(\S+)$")

    all_imports: List[Tuple[int, str]] = []  # (line_idx, stripped_text)
    import_indices: set = set()
    last_block_import_idx = -1

    for i, line in enumerate(lines):
        if line[:1] in (" ", "\t"):
            continue
        stripped = line.strip()
        mf = _FROM_RE.match(stripped)
        if mf:
            names_str = mf.group(2).strip()
            if not names_str or names_str == "*":
                continue
            names = [nm.strip() for nm in names_str.split(",") if nm.strip()]
            if not names:
                continue
            all_imports.append((i, stripped))
            import_indices.add(i)
            if i < first_funcdef_idx:
                last_block_import_idx = i
            continue
        mp = _PLAIN_RE.match(stripped)
        if mp:
            all_imports.append((i, stripped))
            import_indices.add(i)
            if i < first_funcdef_idx:
                last_block_import_idx = i

    if not all_imports:
        return source

    # ── pass 3: build merged import map (ordered by first appearance) ───────
    from_map: Dict[str, List[str]] = {}  # module -> merged name list
    from_order: List[str] = []
    plain_order: List[str] = []
    plain_seen: set = set()

    for _, text in all_imports:
        mf = _FROM_RE.match(text)
        if mf:
            module = mf.group(1)
            names = [nm.strip() for nm in mf.group(2).split(",") if nm.strip()]
            if module not in from_map:
                from_map[module] = list(names)
                from_order.append(module)
            else:
                existing_set = set(from_map[module])
                for name in names:
                    if name not in existing_set:
                        from_map[module].append(name)
                        existing_set.add(name)
        else:
            # Must be a plain import — guaranteed by pass 2 filter.
            module = _PLAIN_RE.match(text).group(1)  # type: ignore[union-attr]
            if module not in plain_seen:
                plain_order.append(module)
                plain_seen.add(module)

    # ── early exit if nothing to do ─────────────────────────────────────────
    has_misplaced = any(i >= first_funcdef_idx for i, _ in all_imports)
    from_counts: Dict[str, int] = {}
    plain_counts: Dict[str, int] = {}
    for _, text in all_imports:
        mf = _FROM_RE.match(text)
        if mf:
            mod = mf.group(1)
            from_counts[mod] = from_counts.get(mod, 0) + 1
        else:
            mod = _PLAIN_RE.match(text).group(1)  # type: ignore[union-attr]
            plain_counts[mod] = plain_counts.get(mod, 0) + 1
    if not (
        has_misplaced
        or any(v > 1 for v in from_counts.values())
        or any(v > 1 for v in plain_counts.values())
    ):
        return source

    # ── pass 4: build the complete sorted import block ──────────────────────
    # Combine every merged import (existing block + newly lifted) and sort the
    # whole list so stdlib never ends up after third-party just because it was
    # a newly lifted import appended at the end.
    all_final_imports = [
        f"from {mod} import {', '.join(from_map[mod])}" for mod in from_order
    ] + [f"import {mod}" for mod in plain_order]
    sorted_imports = _sort_imports_pep8(all_final_imports)

    first_block_import_idx = min(
        (i for i, _ in all_imports if i < first_funcdef_idx), default=-1
    )

    # ── pass 5: rebuild source ───────────────────────────────────────────────
    # Emit the sorted block at the first block import position (or just before
    # the first def/class if there are no block imports).  Skip all original
    # import lines and blank lines within the original block region — the
    # sorted block replaces them entirely.
    result: List[str] = []
    import_block_emitted = False

    for i, line in enumerate(lines):
        # Edge case: no block imports — insert before the first def/class.
        if i == first_funcdef_idx and not import_block_emitted:
            for imp in sorted_imports:
                result.append(imp + "\n")
            import_block_emitted = True

        # Emit the sorted block at the position of the first block import.
        if i == first_block_import_idx:
            for imp in sorted_imports:
                result.append(imp + "\n")
            import_block_emitted = True
            continue  # the original import line is replaced by the block above

        # Drop all other import lines (block duplicates and misplaced).
        if i in import_indices:
            continue

        # Drop blank lines that fell between import lines in the original block
        # — they were section separators that the sorted block supersedes.
        if (
            first_block_import_idx >= 0
            and first_block_import_idx < i <= last_block_import_idx
            and not line.strip()
        ):
            continue

        result.append(line)

    result_str = "".join(result)
    # When a misplaced import is removed, the blank line that visually separated
    # it from the following def/class is left behind.  Combined with the two
    # trailing blank lines already written after the previous helper, this
    # produces three consecutive blank lines — a PEP 8 / E303 violation.
    # Collapse any run of 3+ blank lines down to exactly 2 (the PEP 8 maximum
    # between top-level definitions).  Four or more '\n' in a row means three
    # or more blank lines; replace with exactly three '\n' (= two blank lines).
    result_str = re.sub(r"\n{4,}", "\n\n\n", result_str)
    return result_str


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
