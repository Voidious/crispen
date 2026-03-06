from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement

_CONFLICTING_PLACEMENTS = [
    GroupPlacement(group=["foo"], target_file="utils.py"),  # plan-vs-plan conflict
    GroupPlacement(group=["bar"], target_file="utils/io.py"),  # plan-vs-plan conflict
    GroupPlacement(group=["baz"], target_file="helpers.py"),  # not conflicting
]

_CLEAN_PLACEMENTS = [
    GroupPlacement(group=["foo"], target_file="utils.py"),
    GroupPlacement(group=["bar"], target_file="helpers.py"),
]
