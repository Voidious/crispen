"""Abstract base class for CST-based refactors."""

from typing import List, Sequence, Tuple

import libcst as cst
from libcst.metadata import PositionProvider


class Refactor(cst.CSTTransformer):
    """Base class for all Crispen refactors.

    Subclasses receive the set of changed line ranges and should only
    transform nodes that overlap with those ranges.
    """

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, changed_ranges: List[Tuple[int, int]], source: str = "") -> None:
        super().__init__()
        self.changed_ranges = changed_ranges
        self.changes_made: List[str] = []

    def _in_changed_range(self, node: cst.CSTNode) -> bool:
        """Return True if the node's start line overlaps any changed range."""
        try:
            pos = self.get_metadata(PositionProvider, node)
        except KeyError:  # pragma: no cover
            return False
        node_start = pos.start.line
        node_end = pos.end.line
        for range_start, range_end in self.changed_ranges:
            if node_start <= range_end and node_end >= range_start:
                return True
        return False

    def _line_in_changed_range(self, lineno: int) -> bool:
        """Return True if a single line number is within any changed range."""
        for start, end in self.changed_ranges:
            if start <= lineno <= end:
                return True
        return False

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    def get_changes(self) -> Sequence[str]:
        return self.changes_made
