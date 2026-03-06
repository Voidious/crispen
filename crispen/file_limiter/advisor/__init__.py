"""LLM advisor for FileLimiter: plans entity migration to new files."""

from __future__ import annotations
from .planning_workflow import FileLimiterPlan  # noqa F401
from .planning_workflow import GroupPlacement  # noqa F401
from .planning_workflow import _PLACEMENT_CHUNK_SIZE  # noqa F401
from .planning_workflow import _advise_set3  # noqa F401
from .planning_workflow import _assign_placements_chunk  # noqa F401
from .planning_workflow import _build_group_mermaid  # noqa F401
from .planning_workflow import _compute_projected_lines  # noqa F401
from .planning_workflow import _find_conflicting_placement_indices  # noqa F401
from .planning_workflow import _group_summary  # noqa F401
from .planning_workflow import _propose_files_step  # noqa F401
from .planning_workflow import _refine_merge_tiny  # noqa F401
from .planning_workflow import advise_file_limiter  # noqa F401
from .planning_workflow import resolve_naming_conflicts  # noqa F401


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LLM tool schemas
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
