"""LLM advisor for FileLimiter: plans entity migration to new files."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ...config import CrispenConfig
from ...llm_client import get_api_key, make_client
from ..classifier import ClassifiedEntities
from .llm_tools import GroupPlacement  # fmt: skip # noqa: F401, E501
from .llm_tools import _LLMAccumulator  # fmt: skip # noqa: F401, E501
from .llm_tools import _PLACEMENT_CHUNK_SIZE  # fmt: skip # noqa: F401, E501
from .planning import _assign_placements
from .planning import FileLimiterPlan  # fmt: skip # noqa: F401, E501
from .planning import _advise_set3  # fmt: skip # noqa: F401, E501
from .planning import _assign_placements_chunk  # fmt: skip # noqa: F401, E501
from .planning import _build_group_mermaid  # fmt: skip # noqa: F401, E501
from .planning import _compute_projected_lines  # fmt: skip # noqa: F401, E501
from .planning import _find_conflicting_placement_indices  # fmt: skip # noqa: F401, E501
from .planning import _group_summary  # fmt: skip # noqa: F401, E501
from .planning import _propose_files_step  # fmt: skip # noqa: F401, E501
from .planning import _refine_merge_tiny  # fmt: skip # noqa: F401, E501
from .planning import resolve_naming_conflicts  # fmt: skip # noqa: F401, E501


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def advise_file_limiter(
    classified: ClassifiedEntities,
    original_path: str,
    config: CrispenConfig,
    existing_files: frozenset = frozenset(),
    prev_set3_failure: str = "",
    prev_placement_failure: str = "",
    verbose: bool = False,
    timing: str = "detailed",
    subdir_name: Optional[str] = None,
) -> FileLimiterPlan:
    """Ask the LLM to plan entity placement across new files.

    Returns a :class:`FileLimiterPlan` with ``abort=True`` when planning fails
    or the file cannot be split (e.g. single SCC covering all entities).
    """
    if classified.abort:
        return FileLimiterPlan(
            set3_migrate=[],
            placements=[],
            abort=True,
            abort_reason=classified.abort_reason,
        )

    if not classified.set_2_groups and not classified.set_3_groups:
        return FileLimiterPlan(set3_migrate=[], placements=[], abort=False)

    api_key = get_api_key(config.provider, caller="FileLimiter")
    client = make_client(
        config.provider, api_key, timeout=config.api_timeout, base_url=config.base_url
    )

    counter = _LLMAccumulator()

    # Call 1: advise Set 3 groups (only if set_3 is non-empty).
    # In a test-file subdir split every group must migrate — there is no public
    # API contract that forces anything to stay, and leaving test functions
    # behind in the original causes fixture-visibility problems.  Skip the LLM
    # call and treat all set-3 groups as migrating.
    is_test_subdir = subdir_name is not None and Path(original_path).name.startswith(
        "test_"
    )
    set3_migrate: List[List[str]] = []
    if classified.set_3_groups:
        if is_test_subdir:
            set3_migrate = classified.set_3_groups
        else:
            result = _advise_set3(
                classified,
                original_path,
                client,
                config,
                prev_failure=prev_set3_failure,
                verbose=verbose,
                timing=timing,
                _acc=counter,
            )
            if result is None:
                return FileLimiterPlan(
                    set3_migrate=[],
                    placements=[],
                    abort=True,
                    abort_reason="LLM failed to plan set-3 groups",
                    llm_calls=counter.calls,
                    llm_elapsed=counter.elapsed,
                    llm_input_tokens=counter.input_tokens,
                    llm_output_tokens=counter.output_tokens,
                )
            set3_migrate = result

    # Calls 2+: propose files, assign groups, refine (merge tiny).
    groups_to_place = classified.set_2_groups + set3_migrate
    if not groups_to_place:
        return FileLimiterPlan(
            set3_migrate=set3_migrate,
            placements=[],
            abort=False,
            llm_calls=counter.calls,
            llm_elapsed=counter.elapsed,
            llm_input_tokens=counter.input_tokens,
            llm_output_tokens=counter.output_tokens,
        )

    entity_map = {e.name: e for e in classified.entities}
    total_lines = sum(
        entity_map[name].end_line - entity_map[name].start_line + 1
        for group in groups_to_place
        for name in group
        if name in entity_map
    )
    original_target = (
        max(2, -(-(2 * total_lines) // config.max_file_lines)) if total_lines > 0 else 2
    )
    placements = _assign_placements(
        groups_to_place,
        classified,
        original_path,
        existing_files,
        client,
        config,
        prev_failure=prev_placement_failure,
        verbose=verbose,
        timing=timing,
        _acc=counter,
        subdir_name=subdir_name,
        target_files=original_target,
    )
    if placements is None:
        return FileLimiterPlan(
            set3_migrate=set3_migrate,
            placements=[],
            abort=True,
            abort_reason="LLM failed to assign file placements",
            llm_calls=counter.calls,
            llm_elapsed=counter.elapsed,
            llm_input_tokens=counter.input_tokens,
            llm_output_tokens=counter.output_tokens,
        )

    return FileLimiterPlan(
        set3_migrate=set3_migrate,
        placements=placements,
        abort=False,
        llm_calls=counter.calls,
        llm_elapsed=counter.elapsed,
        llm_input_tokens=counter.input_tokens,
        llm_output_tokens=counter.output_tokens,
    )
