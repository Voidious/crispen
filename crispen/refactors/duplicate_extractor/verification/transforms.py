from __future__ import annotations
from typing import Dict, List, Tuple
import ast
import re
import textwrap
from ....import_sort import _sort_imports_pep8
from .utils import _collect_ast_store_names, _is_pure_literal


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
