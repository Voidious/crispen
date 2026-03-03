"""LLM advisor for FileLimiter: plans entity migration to new files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..config import CrispenConfig
from ..llm_client import call_with_tool, get_api_key, make_client
from .classifier import ClassifiedEntities
from .entity_parser import Entity


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class GroupPlacement:
    """Placement decision for one SCC group."""

    group: List[str]  # entity names in the SCC
    target_file: str  # relative filename (e.g. "utils.py")


@dataclass
class FileLimiterPlan:
    """Complete placement plan from the LLM advisor."""

    # Set 3 groups the LLM chose to migrate (rest stay in original file).
    set3_migrate: List[List[str]]
    # Placement for set_2 groups + migrating set_3 groups.
    placements: List[GroupPlacement]
    # True if planning failed and the file should not be split.
    abort: bool
    abort_reason: str = ""  # human-readable explanation when abort=True


# ---------------------------------------------------------------------------
# LLM tool schemas
# ---------------------------------------------------------------------------


_SET3_TOOL: dict = {
    "name": "advise_set3_actions",
    "description": (
        "For each modified-entity group, decide whether to migrate it to a new "
        "file or leave it in the original file."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group_id": {
                            "type": "integer",
                            "description": "0-based index of the group",
                        },
                        "action": {
                            "type": "string",
                            "enum": ["migrate", "stay"],
                            "description": (
                                "'migrate' to move to a new file, "
                                "'stay' to keep in original"
                            ),
                        },
                    },
                    "required": ["group_id", "action"],
                },
            }
        },
        "required": ["decisions"],
    },
}

_PLACEMENT_TOOL: dict = {
    "name": "assign_file_placements",
    "description": (
        "Assign each entity group to a target Python filename. "
        "Each group will be written to a new file in the same directory "
        "as the original."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "placements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group_id": {
                            "type": "integer",
                            "description": "0-based index of the group",
                        },
                        "target_file": {
                            "type": "string",
                            "description": (
                                "Relative filename, e.g. 'utils.py' or "
                                "'helpers/io.py'"
                            ),
                        },
                    },
                    "required": ["group_id", "target_file"],
                },
            }
        },
        "required": ["placements"],
    },
}


# Maximum number of groups per placement LLM call.  Large files may have
# dozens of groups; sending them all in one call frequently causes timeouts
# or incomplete responses.  This limit keeps each call small and reliable.
_PLACEMENT_CHUNK_SIZE = 20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _group_summary(group: List[str], entity_map: Dict[str, Entity]) -> str:
    """Return a brief text description of an SCC group for LLM context."""
    parts = []
    for name in group:
        ent = entity_map.get(name)
        if ent:
            size = ent.end_line - ent.start_line + 1
            parts.append(f"{name} ({size} lines)")
        else:
            parts.append(name)
    return ", ".join(parts)


def _advise_set3(
    classified: ClassifiedEntities,
    original_path: str,
    client: object,
    config: CrispenConfig,
    prev_failure: str = "",
) -> Optional[List[List[str]]]:
    """Ask the LLM which Set 3 groups should migrate. Returns None on failure."""
    entity_map = {e.name: e for e in classified.entities}
    group_lines = []
    for idx, group in enumerate(classified.set_3_groups):
        summary = _group_summary(group, entity_map)
        group_lines.append(f"  [{idx}]: {summary}")
    groups_text = "\n".join(group_lines)

    n_groups = len(classified.set_3_groups)
    content = (
        f"The file '{original_path}' is over the maximum line limit and MUST "
        "be reduced in size by splitting it. The following entity groups are "
        "MODIFIED (they existed before the diff and were changed by the "
        "current diff). Each group is a mutual dependency cycle and must be "
        "moved as an indivisible unit — it cannot be split further.\n\n"
        f"Groups:\n{groups_text}\n\n"
        "IMPORTANT: 'migrate' is the preferred action. The goal is to move "
        "as many groups as possible to new files so the original file shrinks "
        "below the line limit. Choose 'stay' ONLY if there is a compelling "
        "reason the group cannot be extracted (for example, it is the sole "
        "public API entry-point of the module and callers import it by name "
        "from this specific file). If ALL groups stay, no split will occur "
        "and the file will remain over the limit, which is not acceptable.\n\n"
        "For each group, return 'migrate' (preferred) or 'stay' (exceptional)."
    )
    if prev_failure:
        content += f"\n\nFeedback from the previous attempt: {prev_failure}"
    messages = [{"role": "user", "content": content}]
    max_tokens = max(512, 20 + n_groups * 15)
    result = call_with_tool(
        client,
        config.provider,
        config.model,
        max_tokens,
        _SET3_TOOL,
        "advise_set3_actions",
        messages,
        caller="FileLimiter",
        tool_choice_override=config.tool_choice,
    )
    if result is None:
        return None

    migrate_ids = set()
    for decision in result.get("decisions", []):
        gid = decision.get("group_id")
        action = decision.get("action")
        if isinstance(gid, int) and 0 <= gid < len(classified.set_3_groups):
            if action == "migrate":
                migrate_ids.add(gid)

    return [
        classified.set_3_groups[i]
        for i in range(len(classified.set_3_groups))
        if i in migrate_ids
    ]


def _assign_placements_chunk(
    chunk: List[List[str]],
    classified: ClassifiedEntities,
    original_path: str,
    existing_files: frozenset,
    client: object,
    config: CrispenConfig,
    prev_failure: str = "",
) -> Optional[List[GroupPlacement]]:
    """Ask the LLM to assign filenames to one chunk of groups.

    Groups within *chunk* are numbered 0…N-1 for this call.
    Returns ``None`` on failure (LLM error, missing group, or existing-file
    collision).
    """
    entity_map = {e.name: e for e in classified.entities}
    group_lines = []
    for idx, group in enumerate(chunk):
        summary = _group_summary(group, entity_map)
        group_lines.append(f"  [{idx}]: {summary}")
    groups_text = "\n".join(group_lines)

    exclude_section = ""
    if existing_files:
        file_list = "\n".join(f"  - {f}" for f in sorted(existing_files))
        exclude_section = (
            "\n\nThe following files already exist — do NOT use them as targets:\n"
            + file_list
        )

    n_groups = len(chunk)
    original_basename = Path(original_path).name
    content = (
        f"Assign each entity group to a target Python filename. "
        f"The original file is '{original_path}'. "
        "Each group will be written to a NEW file in the same directory "
        "as the original.\n\n"
        f"Groups to place (you MUST return a target_file for every "
        f"group_id listed):\n{groups_text}\n\n"
        "Rules:\n"
        f"- You MUST assign a target_file to ALL {n_groups} group(s). "
        "Missing any group_id will cause the split to fail.\n"
        f"- Use filenames relative to the same directory "
        f"(e.g. 'utils.py'). Do NOT use '{original_basename}' "
        "(the original file being split).\n"
        "- Multiple groups may share the same target file if they are "
        "semantically related.\n"
        "- Choose descriptive names based on what the entities do "
        "(e.g. 'helpers.py', 'models.py', 'extractors.py')."
        f"{exclude_section}"
    )
    if prev_failure:
        content += f"\n\nFeedback from the previous attempt: {prev_failure}"
    messages = [{"role": "user", "content": content}]
    max_tokens = max(512, 20 + n_groups * 20)
    result = call_with_tool(
        client,
        config.provider,
        config.model,
        max_tokens,
        _PLACEMENT_TOOL,
        "assign_file_placements",
        messages,
        caller="FileLimiter",
        tool_choice_override=config.tool_choice,
    )
    if result is None:
        return None

    placements: List[GroupPlacement] = []
    placed_ids: set = set()
    for item in result.get("placements", []):
        gid = item.get("group_id")
        target = item.get("target_file", "")
        if (
            isinstance(gid, int)
            and 0 <= gid < len(chunk)
            and gid not in placed_ids
            and target
        ):
            if target in existing_files:
                return None
            placements.append(GroupPlacement(group=chunk[gid], target_file=target))
            placed_ids.add(gid)

    if len(placements) != len(chunk):
        return None

    return placements


def _assign_placements(
    groups_to_place: List[List[str]],
    classified: ClassifiedEntities,
    original_path: str,
    existing_files: frozenset,
    client: object,
    config: CrispenConfig,
    prev_failure: str = "",
) -> Optional[List[GroupPlacement]]:
    """Ask the LLM to assign filenames to each group. Returns None on failure.

    When there are more than :data:`_PLACEMENT_CHUNK_SIZE` groups the request
    is split into multiple LLM calls of at most that many groups each.  Each
    chunk gets its own retry budget of ``config.file_limiter_retries`` extra
    attempts so that a transient timeout on one chunk does not require
    restarting all chunks.  If any chunk fails all its attempts, ``None`` is
    returned immediately.
    """
    all_placements: List[GroupPlacement] = []
    for chunk_start in range(0, len(groups_to_place), _PLACEMENT_CHUNK_SIZE):
        chunk = groups_to_place[chunk_start : chunk_start + _PLACEMENT_CHUNK_SIZE]
        chunk_placements: Optional[List[GroupPlacement]] = None
        for _ in range(1 + config.file_limiter_retries):
            chunk_placements = _assign_placements_chunk(
                chunk,
                classified,
                original_path,
                existing_files,
                client,
                config,
                prev_failure,
            )
            if chunk_placements is not None:
                break
        if chunk_placements is None:
            return None
        all_placements.extend(chunk_placements)
    return all_placements


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

    # Call 1: advise Set 3 groups (only if set_3 is non-empty).
    set3_migrate: List[List[str]] = []
    if classified.set_3_groups:
        result = _advise_set3(
            classified, original_path, client, config, prev_failure=prev_set3_failure
        )
        if result is None:
            return FileLimiterPlan(
                set3_migrate=[],
                placements=[],
                abort=True,
                abort_reason="LLM failed to plan set-3 groups",
            )
        set3_migrate = result

    # Call 2: assign filenames for set_2 + migrating set_3.
    groups_to_place = classified.set_2_groups + set3_migrate
    if not groups_to_place:
        return FileLimiterPlan(set3_migrate=set3_migrate, placements=[], abort=False)

    placements = _assign_placements(
        groups_to_place,
        classified,
        original_path,
        existing_files,
        client,
        config,
        prev_failure=prev_placement_failure,
    )
    if placements is None:
        return FileLimiterPlan(
            set3_migrate=set3_migrate,
            placements=[],
            abort=True,
            abort_reason="LLM failed to assign file placements",
        )

    return FileLimiterPlan(
        set3_migrate=set3_migrate, placements=placements, abort=False
    )
