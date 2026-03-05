from __future__ import annotations
from typing import List, Tuple
from .models import _SeqInfo


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
