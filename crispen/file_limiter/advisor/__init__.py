"""LLM advisor for FileLimiter: plans entity migration to new files."""

from __future__ import annotations
from .conflict_resolution import _find_conflicting_placement_indices  # fmt: skip # noqa: F401, E501
from .conflict_resolution import resolve_naming_conflicts  # fmt: skip # noqa: F401, E501
from .placement_planner import FileLimiterPlan  # fmt: skip # noqa: F401, E501
from .placement_planner import GroupPlacement  # fmt: skip # noqa: F401, E501
from .placement_planner import _PLACEMENT_CHUNK_SIZE  # fmt: skip # noqa: F401, E501
from .placement_planner import _advise_set3  # fmt: skip # noqa: F401, E501
from .placement_planner import _assign_placements_chunk  # fmt: skip # noqa: F401, E501
from .placement_planner import _build_group_mermaid  # fmt: skip # noqa: F401, E501
from .placement_planner import _compute_projected_lines  # fmt: skip # noqa: F401, E501
from .placement_planner import _group_summary  # fmt: skip # noqa: F401, E501
from .placement_planner import _propose_files_step  # fmt: skip # noqa: F401, E501
from .placement_planner import _refine_merge_tiny  # fmt: skip # noqa: F401, E501
from .placement_planner import advise_file_limiter  # fmt: skip # noqa: F401, E501


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
