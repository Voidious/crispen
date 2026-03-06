from __future__ import annotations
from crispen.file_limiter.advisor import (
    _find_conflicting_placement_indices,
    GroupPlacement,
)


def test_find_conflicting_idx_plan_vs_plan():
    """Flat file + subdir with same stem both appear → both indices returned."""
    placements = [
        GroupPlacement(group=["foo"], target_file="utils.py"),
        GroupPlacement(group=["bar"], target_file="utils/io.py"),
        GroupPlacement(group=["baz"], target_file="helpers.py"),
    ]
    idxs = _find_conflicting_placement_indices(placements, frozenset(), frozenset())
    assert idxs == [0, 1]


def test_find_conflicting_idx_file_vs_existing_dir():
    """Flat .py target whose stem matches an existing directory → index returned."""
    placements = [GroupPlacement(group=["foo"], target_file="models.py")]
    idxs = _find_conflicting_placement_indices(
        placements, frozenset(), frozenset({"models"})
    )
    assert idxs == [0]


def test_find_conflicting_idx_subdir_vs_existing_file():
    """Subdir target whose top matches an existing .py file → index returned."""
    placements = [GroupPlacement(group=["bar"], target_file="helpers/io.py")]
    idxs = _find_conflicting_placement_indices(
        placements, frozenset({"helpers.py"}), frozenset()
    )
    assert idxs == [0]


def test_find_conflicting_idx_no_conflict():
    """Clean plan with no conflicts → empty list."""
    placements = [
        GroupPlacement(group=["foo"], target_file="utils.py"),
        GroupPlacement(group=["bar"], target_file="helpers.py"),
    ]
    assert (
        _find_conflicting_placement_indices(placements, frozenset(), frozenset()) == []
    )
