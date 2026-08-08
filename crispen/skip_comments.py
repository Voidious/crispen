"""Shared detection for ``# crispen: skip`` escape-hatch comments.

Format (mirrors ``# noqa`` / ``# type: ignore`` / ``# pylint: disable=...``):

* ``# crispen: skip`` — protect the attached statement/function/entity from
  every refactor.
* ``# crispen: skip=<name>[,<name>...]`` — protect it from only the named
  refactor(s). Names match ``CrispenConfig.enabled_refactors`` /
  ``disabled_refactors``: "if_not_else", "duplicate_extractor",
  "function_splitter", "tuple_dataclass", "file_limiter".
* ``# crispen: skip-file`` — protect the whole file from every refactor.

A marker "attaches" to whatever starts on its own line: either a trailing
comment on that exact line, or a standalone comment on an immediately
preceding line (blank lines and decorator lines are transparent, so a
marker placed above a stack of decorators still attaches to the def/class
below them).
"""

from __future__ import annotations

import io
import re
import tokenize
from typing import Dict, List, Optional

# Excludes "skip-file" via the negative lookahead so it is only ever
# recognized by _SKIP_FILE_RE, not treated as a bare/global "skip".
_SKIP_RE = re.compile(r"#\s*crispen:\s*skip(?!-)(?:=([\w,]+))?\b")
_SKIP_FILE_RE = re.compile(r"#\s*crispen:\s*skip-file\b")


def extract_comments(source: str) -> Dict[int, str]:
    """Map 1-indexed line number to raw comment text (e.g. ``"# crispen: skip"``).

    Uses the tokenizer (not a text search) so a ``#`` inside a string literal
    is never mistaken for a comment. Returns an empty map for unparseable
    source rather than raising.
    """
    comments: Dict[int, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                comments[tok.start[0]] = tok.string
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        pass
    return comments


def _scope_matches(scope: Optional[str], refactor_name: str) -> bool:
    """Return True if a skip marker's scope (the part after ``=``) covers
    *refactor_name*. ``None`` (bare ``# crispen: skip``) covers everything."""
    if scope is None:
        return True
    names = {n.strip() for n in scope.split(",") if n.strip()}
    return refactor_name in names


def is_skipped(
    start_line: int,
    refactor_name: str,
    source_lines: List[str],
    comments_by_line: Dict[int, str],
) -> bool:
    """Return True if a skip marker protects *start_line* (1-indexed) from
    *refactor_name*.

    Checks a trailing comment on *start_line* itself, then walks upward
    through blank lines, decorator lines (``@...``), and other standalone
    comment lines looking for a leading marker.
    """
    comment = comments_by_line.get(start_line)
    if comment and _line_marker_matches(comment, refactor_name):
        return True

    idx = min(start_line - 2, len(source_lines) - 1)  # 0-indexed, clamped in range
    while idx >= 0:
        stripped = source_lines[idx].strip()
        if not stripped or stripped.startswith("@"):
            idx -= 1
            continue
        if stripped.startswith("#"):
            comment = comments_by_line.get(idx + 1)
            if comment and _line_marker_matches(comment, refactor_name):
                return True
            idx -= 1
            continue
        break
    return False


def _line_marker_matches(comment: str, refactor_name: str) -> bool:
    m = _SKIP_RE.match(comment)
    return bool(m and _scope_matches(m.group(1), refactor_name))


def has_skip_file_marker(source: str) -> bool:
    """Return True if ``# crispen: skip-file`` appears anywhere in *source*."""
    return any(_SKIP_FILE_RE.match(c) for c in extract_comments(source).values())
