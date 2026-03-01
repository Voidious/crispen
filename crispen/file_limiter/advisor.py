"""LLM advisor for FileLimiter: plans entity migration to new files."""

from __future__ import annotations

from dataclasses import dataclass
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
) -> Optional[List[List[str]]]:
    """Ask the LLM which Set 3 groups should migrate. Returns None on failure."""
    entity_map = {e.name: e for e in classified.entities}
    group_lines = []
    for idx, group in enumerate(classified.set_3_groups):
        summary = _group_summary(group, entity_map)
        group_lines.append(f"  [{idx}]: {summary}")
    groups_text = "\n".join(group_lines)

    messages = [
        {
            "role": "user",
            "content": (
                f"The file '{original_path}' exceeds the line limit and needs to be "
                "split. The following entity groups are MODIFIED (they existed before "
                "and were changed by the current diff). Each group is a dependency "
                "cycle and cannot be split across files.\n\n"
                f"Groups:\n{groups_text}\n\n"
                "For each group, decide: 'migrate' to a new file, or 'stay' in the "
                "original file."
            ),
        }
    ]

    n_groups = len(classified.set_3_groups)
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


def _assign_placements(
    groups_to_place: List[List[str]],
    classified: ClassifiedEntities,
    original_path: str,
    client: object,
    config: CrispenConfig,
) -> Optional[List[GroupPlacement]]:
    """Ask the LLM to assign filenames to each group. Returns None on failure."""
    entity_map = {e.name: e for e in classified.entities}
    group_lines = []
    for idx, group in enumerate(groups_to_place):
        summary = _group_summary(group, entity_map)
        group_lines.append(f"  [{idx}]: {summary}")
    groups_text = "\n".join(group_lines)

    messages = [
        {
            "role": "user",
            "content": (
                f"Assign each entity group to a target Python filename. "
                f"The original file is '{original_path}'. "
                "Use filenames relative to the same directory.\n\n"
                f"Groups to place:\n{groups_text}"
            ),
        }
    ]

    n_groups = len(groups_to_place)
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
            and 0 <= gid < len(groups_to_place)
            and gid not in placed_ids
            and target
        ):
            placements.append(
                GroupPlacement(group=groups_to_place[gid], target_file=target)
            )
            placed_ids.add(gid)

    if len(placements) != len(groups_to_place):
        return None

    return placements


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def advise_file_limiter(
    classified: ClassifiedEntities,
    original_path: str,
    config: CrispenConfig,
) -> FileLimiterPlan:
    """Ask the LLM to plan entity placement across new files.

    Returns a :class:`FileLimiterPlan` with ``abort=True`` when planning fails
    or the file cannot be split (e.g. single SCC covering all entities).
    """
    if classified.abort:
        return FileLimiterPlan(set3_migrate=[], placements=[], abort=True)

    if not classified.set_2_groups and not classified.set_3_groups:
        return FileLimiterPlan(set3_migrate=[], placements=[], abort=False)

    api_key = get_api_key(config.provider, caller="FileLimiter")
    client = make_client(
        config.provider, api_key, timeout=config.api_timeout, base_url=config.base_url
    )

    # Call 1: advise Set 3 groups (only if set_3 is non-empty).
    set3_migrate: List[List[str]] = []
    if classified.set_3_groups:
        result = _advise_set3(classified, original_path, client, config)
        if result is None:
            return FileLimiterPlan(set3_migrate=[], placements=[], abort=True)
        set3_migrate = result

    # Call 2: assign filenames for set_2 + migrating set_3.
    groups_to_place = classified.set_2_groups + set3_migrate
    if not groups_to_place:
        return FileLimiterPlan(set3_migrate=set3_migrate, placements=[], abort=False)

    placements = _assign_placements(
        groups_to_place, classified, original_path, client, config
    )
    if placements is None:
        return FileLimiterPlan(set3_migrate=set3_migrate, placements=[], abort=True)

    return FileLimiterPlan(
        set3_migrate=set3_migrate, placements=placements, abort=False
    )
