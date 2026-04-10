from __future__ import annotations
from typing import List, Tuple
import ast
import re
import textwrap
from .utils import _SeqInfo


def _normalize_replacement_indentation(seq: _SeqInfo, replacement: str) -> str:
    """Re-indent *replacement* to match the original block's leading whitespace.

    The LLM sometimes returns replacements at column 0.  This function
    re-indents them to match the indentation of the corresponding original
    block, so the assembled edit remains valid Python.
    """
    orig_lines = [ln for ln in seq.source.splitlines() if ln.strip()]
    if not orig_lines:
        return replacement
    first = orig_lines[0]
    expected_indent = first[: len(first) - len(first.lstrip())]
    dedented = textwrap.dedent(replacement)
    if not expected_indent:
        return dedented
    return textwrap.indent(dedented, expected_indent)


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


def _strip_unused_call_assignments(replacement: str, following_lines: List[str]) -> str:
    """Clean up unused assignment targets in a call-site replacement.

    For each ``Assign`` node whose right-hand side is a ``Call``:

    * **Single target** — unused ``Name`` elements in the target are replaced
      with ``_``.  If every element is unused the whole assignment is dropped
      and only the call expression is emitted.  Example::

          result = _helper(x)          →  _helper(x)
          a, b   = _helper(x)  (b used)  →  a, _ = _helper(x)

    * **Chained assignment** (``a = b = call()``) — stripped to just the call
      only when every name across every target is unused; otherwise left alone.

    Augmented (``+=``) and annotated assignments are never touched.  Assignment
    targets that are not plain names or tuples/lists (e.g. ``self.x``) are
    treated as *used* so we never accidentally remove live assignments.

    This prevents flake8 F841 "local variable assigned but never used" errors
    introduced by the extraction.
    """
    following_src = "".join(following_lines)
    try:
        dedented = textwrap.dedent(replacement)
        tree = ast.parse(dedented)
    except SyntaxError:
        return replacement

    # Build a list of (start_ln, end_ln, new_src) edits.  new_src is the
    # replacement text for that statement (without leading indentation).
    edits: List[Tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value_node = node.value
        if isinstance(value_node, ast.Await) and isinstance(value_node.value, ast.Call):
            pass  # treat `result = await helper(...)` like `result = helper(...)`
        elif not isinstance(value_node, ast.Call):
            continue

        call_src = ast.unparse(value_node)

        if len(node.targets) == 1:
            new_target, all_replaced, any_replaced = _replace_unused_in_target(
                node.targets[0], following_src
            )
            if all_replaced:
                edits.append((node.lineno, node.end_lineno, call_src))
            elif any_replaced:
                edits.append(
                    (
                        node.lineno,
                        node.end_lineno,
                        ast.unparse(new_target) + " = " + call_src,
                    )
                )
        else:
            # Chained assignment: strip only when every name is unused.
            all_names: List[str] = []
            for t in node.targets:
                _collect_ast_store_names(t, all_names)
            if not all_names:
                continue
            if not any(
                re.search(r"\b" + re.escape(n) + r"\b", following_src)
                for n in all_names
            ):
                edits.append((node.lineno, node.end_lineno, call_src))

    if not edits:
        return replacement

    # Determine the leading indentation from the first non-empty line.
    first_content = next((ln for ln in replacement.splitlines() if ln.strip()), "")
    indent = first_content[: len(first_content) - len(first_content.lstrip())]

    # Apply edits in reverse line order so earlier indices stay valid.
    dedented_lines = dedented.splitlines(keepends=True)
    for start_ln, end_ln, new_src in sorted(edits, key=lambda x: x[0], reverse=True):
        dedented_lines[start_ln - 1 : end_ln] = [new_src + "\n"]

    return textwrap.indent("".join(dedented_lines), indent)


def _build_helper_insertion(
    source_lines: List[str],
    insert_pos: int,
    helper_source: str,
    placement: str,
) -> Tuple[int, int, str]:
    """Build an edit tuple that inserts helper_source with correct surrounding blanks.

    Always returns a pure insertion (start == end) so that two groups inserting
    before the same scope are never in conflict: pure insertions are not subject
    to the overlap-skip logic in _apply_edits.

    The insertion point is placed after all blank lines that already exist
    around insert_pos (right before the def/decorator line).  Leading blank
    lines are prepended only to make up the difference so the result always
    has exactly ``blank_lines`` blank lines before the helper.
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

    # Insert right before the def/decorator (after all surrounding blanks).
    insert_at = insert_pos + after_blanks
    # Prepend only as many blank lines as are still missing.
    leading = max(0, blank_lines - (before_blanks + after_blanks))
    clean = helper_source.strip("\n") + "\n"
    text = "\n" * leading + clean + "\n" * blank_lines
    return (insert_at, insert_at, text)


def _apply_edits(source: str, edits: List[Tuple[int, int, str]]) -> str:
    """Apply (start_0, end_0, text) edits bottom-to-top.

    Indices are 0-based; lines[start_0:end_0] is replaced with text.
    An insertion before line N uses start_0 == end_0 == N.
    Overlapping replacement ranges are skipped.
    """
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    applied: List[Tuple[int, int]] = []
    for start, end, text in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        is_insertion = start == end
        if not is_insertion:
            if any(a_start < end and a_end > start for a_start, a_end in applied):
                continue
            applied.append((start, end))
        new_lines = text.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        lines[start:end] = new_lines

    return "".join(lines)


def _skip_class_docstring(source_lines: List[str], after_class_line: int) -> int:
    """Return the 0-based line index after the class docstring, if any.

    Given the line immediately after ``class Foo:`` (or its colon line),
    advance past any leading blank lines and then past a string-literal
    docstring (single- or triple-quoted).  If no docstring is present,
    returns ``after_class_line`` unchanged.
    """
    i = after_class_line
    n = len(source_lines)
    # Skip blank lines inside the class body.
    while i < n and not source_lines[i].strip():
        i += 1
    if i >= n:
        return after_class_line
    stripped = source_lines[i].lstrip()
    # Check for a triple-quoted docstring.
    for q in ('"""', "'''"):
        if stripped.startswith(q):
            # Check whether the closing quote is on the same line (after the
            # opening).
            rest = stripped[len(q) :]
            if q in rest:
                # Single-line triple-quoted docstring.
                return i + 1
            # Multi-line: scan forward for the closing triple-quote.
            i += 1
            while i < n:
                if q in source_lines[i]:
                    return i + 1
                i += 1
            return i  # malformed, best-effort
    # Single-quoted docstring (rare but valid).
    for q in ('"', "'"):
        if stripped.startswith(q) and not stripped.startswith(q * 2):
            return i + 1
    return after_class_line


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
