from __future__ import annotations
from typing import List, Tuple


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
