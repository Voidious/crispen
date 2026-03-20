from __future__ import annotations
from typing import Dict, List, Tuple
import ast
import re
import textwrap
import libcst as cst
from ....import_sort import _sort_imports_pep8
from ..core import _SeqInfo
from .ast_utils import _collect_ast_store_names, _replace_unused_in_target


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
