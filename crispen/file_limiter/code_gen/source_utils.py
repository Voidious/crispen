from __future__ import annotations
from typing import Optional, Set
import ast
import re
from ..entity_parser import _parse_section_headers


# Matches any line that is an import statement (plain or from-import).
_IMPORT_LINE_RE = re.compile(r"^(import\s+|from\s+\S.*\s+import\s+)")

# Matches a `from __future__ import …` line (with optional trailing newline).
_FUTURE_IMPORT_LINE_RE = re.compile(r"^from __future__ import .*\n?", re.MULTILINE)

# Matches the leading dots of a relative import (``from .foo`` or ``from ..``).
_REL_IMPORT_RE = re.compile(r"^from (\.+)", re.MULTILINE)

# Matches four or more consecutive newlines (= 3+ blank lines between entities).
_EXCESS_BLANK_RE = re.compile(r"\n{4,}")
# Matches 3+ consecutive newlines followed by indented content (= 2+ blank lines
# inside a function/class body, where flake8 E303 allows at most one blank line).
_EXCESS_BLANK_BODY_RE = re.compile(r"\n{3,}(?=[ \t])")


def _normalize_blank_lines(source: str) -> str:
    """Collapse excess blank lines; ensure exactly one trailing newline.

    Removes blank-line artefacts produced by entity removal (original file)
    and entity-source stripping (new files):

    - Strips leading blank lines at the start of the file (E303).
    - Collapses 3+ consecutive blank lines between top-level definitions to 2
      (E303; PEP 8 allows at most two blank lines at module level).
    - Collapses 2+ consecutive blank lines inside indented bodies to 1
      (E303; PEP 8 allows at most one blank line inside a function/class).

    Returns an empty string when *source* contains only whitespace, signalling
    that the file should be deleted rather than written with a lone blank line.
    """
    source = _EXCESS_BLANK_RE.sub("\n\n\n", source)
    source = _EXCESS_BLANK_BODY_RE.sub("\n\n", source)
    source = source.lstrip("\n")
    stripped = source.rstrip("\n")
    if not stripped.strip():
        return ""
    return stripped + "\n"


def _strip_orphaned_section_headers(source: str) -> str:
    """Remove section header comment blocks with no substantive code after them.

    When entities are removed from the original file, section headers that
    labelled a group of functions may be left with nothing beneath them.
    This function detects both 3-line (``# ---...--- / # Label / # ---...---``)
    and single-line (``# --- Label ---``, ``# === LABEL ===``) patterns and
    removes any whose remaining content (non-blank, non-header lines) has
    been entirely stripped away.
    """
    lines = source.splitlines(keepends=True)
    headers = _parse_section_headers(lines)
    if not headers:
        return source

    # 1-indexed set of lines that belong to any header block.
    header_1idx: Set[int] = set()
    for start, end, _ in headers:
        header_1idx.update(range(start, end + 1))

    # A header is orphaned when no substantive line (non-blank and not part of
    # any header block) falls between it and the *next* header (or EOF).
    orphaned_0idx: Set[int] = set()
    for h_idx, (start_1, end_1, _) in enumerate(headers):
        # Scan only up to the start of the next header so that content beneath
        # a later header does not rescue an earlier, empty one.
        if h_idx + 1 < len(headers):
            scan_end_0 = headers[h_idx + 1][0] - 1  # 0-indexed exclusive
        else:
            scan_end_0 = len(lines)
        has_content = False
        for j0 in range(end_1, scan_end_0):  # 0-indexed, past the header block
            stripped = lines[j0].strip()
            if stripped and (j0 + 1) not in header_1idx:
                has_content = True
                break
        if not has_content:
            for i1 in range(start_1, end_1 + 1):
                orphaned_0idx.add(i1 - 1)  # convert to 0-indexed

    if not orphaned_0idx:
        return source
    return "".join(line for i, line in enumerate(lines) if i not in orphaned_0idx)


def _strip_orphaned_indented_comments(source: str) -> str:
    """Remove indented comment lines that appear at module level.

    After FileLimiter moves a function to a new file using AST line ranges,
    trailing comments that were inside the function body may be left behind
    in the original file.  These comments retain their original indentation
    (e.g. four spaces) even though they are now at module level, causing
    flake8 E116 (unexpected indentation: comment).

    This function uses ``ast.parse`` to build the set of line numbers covered
    by any AST node.  Any comment line with leading whitespace whose line
    number falls outside that set is considered orphaned and removed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    covered: Set[int] = set()
    for node in ast.walk(tree):
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            for lineno in range(node.lineno, node.end_lineno + 1):
                covered.add(lineno)

    lines = source.splitlines(keepends=True)
    result = []
    for i, line in enumerate(lines):
        lineno = i + 1  # 1-indexed
        stripped = line.lstrip()
        is_indented_comment = stripped.startswith("#") and len(line) > len(stripped)
        if is_indented_comment and lineno not in covered:
            continue
        result.append(line)
    return "".join(result)


def _extract_module_docstring(source: str) -> Optional[str]:
    """Return the module-level docstring source text, or None if absent."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if not (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        return None
    node = tree.body[0]
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno]).rstrip()


def _strip_module_docstring(src: str) -> str:
    """Return *src* with the leading module-level docstring removed."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    if not (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        return src
    node = tree.body[0]
    remove = set(range(node.lineno, node.end_lineno + 1))
    lines = src.splitlines(keepends=True)
    return "".join(line for i, line in enumerate(lines, 1) if i not in remove)


def _source_is_only_docstring(source: str) -> bool:
    """Return True if *source* contains only a module-level docstring."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return (
        len(tree.body) == 1
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    )
