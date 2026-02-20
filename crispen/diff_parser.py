"""Parse unified diffs into changed line ranges per file."""

from typing import Dict, List, Tuple

from unidiff import PatchSet


def parse_diff(diff_text: str) -> Dict[str, List[Tuple[int, int]]]:
    """Parse a unified diff string into a map of filename -> changed line ranges.

    Returns a dict mapping each modified file path to a list of (start, end)
    line number tuples (1-based, inclusive) covering added/modified lines.
    """
    patch = PatchSet.from_string(diff_text)
    result: Dict[str, List[Tuple[int, int]]] = {}

    for patched_file in patch:
        path = patched_file.path
        if not path.endswith(".py"):
            continue
        ranges: List[Tuple[int, int]] = []

        for hunk in patched_file:
            added_lines = [
                line.target_line_no
                for line in hunk
                if line.is_added and line.target_line_no is not None
            ]
            if not added_lines:
                continue
            # Merge consecutive line numbers into ranges
            start = added_lines[0]
            prev = added_lines[0]
            for lineno in added_lines[1:]:
                if lineno == prev + 1:
                    prev = lineno
                else:
                    ranges.append((start, prev))
                    start = lineno
                    prev = lineno
            ranges.append((start, prev))

        if ranges:
            result[path] = ranges

    return result
