"""LLM advisor for FileLimiter: plans entity migration to new files."""

from __future__ import annotations
from .llm_schemas import GroupPlacement  # fmt: skip # noqa: F401, E501
from .llm_schemas import _PLACEMENT_CHUNK_SIZE  # fmt: skip # noqa: F401, E501
from .planning import FileLimiterPlan  # fmt: skip # noqa: F401, E501
from .planning import _LLMAccumulator  # fmt: skip # noqa: F401, E501
from .planning import _advise_set3  # fmt: skip # noqa: F401, E501
from .planning import _assign_placements_chunk  # fmt: skip # noqa: F401, E501
from .planning import _build_group_mermaid  # fmt: skip # noqa: F401, E501
from .planning import _compute_projected_lines  # fmt: skip # noqa: F401, E501
from .planning import _find_conflicting_placement_indices  # fmt: skip # noqa: F401, E501
from .planning import _group_summary  # fmt: skip # noqa: F401, E501
from .planning import _propose_files_step  # fmt: skip # noqa: F401, E501
from .planning import _refine_merge_tiny  # fmt: skip # noqa: F401, E501
from .planning import advise_file_limiter  # fmt: skip # noqa: F401, E501
from .planning import resolve_naming_conflicts  # fmt: skip # noqa: F401, E501
