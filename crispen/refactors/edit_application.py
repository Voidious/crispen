from __future__ import annotations
import re
from typing import List, Tuple


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
