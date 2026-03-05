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

_RENAME_CONFLICTS_TOOL: dict = {
    "name": "rename_conflicting_placements",
    "description": (
        "Assign new, non-conflicting target filenames to entity groups "
        "that currently have naming conflicts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "placements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group_id": {"type": "integer"},
                        "target_file": {"type": "string"},
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


def _init_placements_from_result(
    result: Optional[dict],
) -> Optional[tuple[List[GroupPlacement], set]]:
    if result is None:
        return None

    placements: List[GroupPlacement] = []
    placed_ids: set = set()
    return placements, placed_ids


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
    init_result = _init_placements_from_result(result)
    if init_result is None:
        return None

    placements, placed_ids = init_result
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


def _find_conflicting_placement_indices(
    placements: List[GroupPlacement],
    existing_files: frozenset,
    existing_dirs: frozenset,
) -> List[int]:
    """Return indices of placements whose target_file creates a naming conflict.

    Mirrors the logic of ``_detect_naming_conflicts`` in runner.py but returns
    indices of the conflicting entries instead of human-readable strings.
    """
    file_stem_to_idxs: Dict[str, List[int]] = {}
    dir_top_to_idxs: Dict[str, List[int]] = {}
    for i, p in enumerate(placements):
        parts = Path(p.target_file).parts
        if len(parts) == 1:
            file_stem_to_idxs.setdefault(Path(parts[0]).stem, []).append(i)
        else:
            dir_top_to_idxs.setdefault(parts[0], []).append(i)

    conflicting: set = set()
    for stem in set(file_stem_to_idxs) & set(dir_top_to_idxs):  # plan-vs-plan
        conflicting.update(file_stem_to_idxs[stem])
        conflicting.update(dir_top_to_idxs[stem])
    for stem, idxs in file_stem_to_idxs.items():
        if stem in existing_dirs:  # flat file vs disk dir
            conflicting.update(idxs)
    for top, idxs in dir_top_to_idxs.items():
        if f"{top}.py" in existing_files:  # subdir vs disk file
            conflicting.update(idxs)
    return sorted(conflicting)


def _rename_conflicting_chunk(
    chunk: List[GroupPlacement],
    classified: ClassifiedEntities,
    original_path: str,
    forbidden_files: frozenset,
    forbidden_dir_stems: frozenset,
    existing_file_stems: frozenset,
    client: object,
    config: CrispenConfig,
    prev_failure: str = "",
) -> Optional[List[GroupPlacement]]:
    """Ask the LLM to rename conflicting placements in *chunk*.

    Returns ``None`` if the LLM fails, picks a forbidden name, or returns
    an incomplete response.
    """
    entity_map = {e.name: e for e in classified.entities}
    n_groups = len(chunk)
    group_lines = []
    for idx, p in enumerate(chunk):
        summary = _group_summary(p.group, entity_map)
        group_lines.append(
            f"  [{idx}]: current target='{p.target_file}' \u2014 {summary}"
        )
    groups_text = "\n".join(group_lines)

    file_list = "\n".join(f"  - {f}" for f in sorted(forbidden_files))
    forbidden_section = (
        "\n\nForbidden filenames (already taken or reserved):\n" + file_list
    )

    dir_stems_section = ""
    if forbidden_dir_stems:
        dir_list = "\n".join(f"  - {d}/" for d in sorted(forbidden_dir_stems))
        dir_stems_section = (
            "\n\nExisting package directories (do NOT propose a flat 'X.py' "
            "file whose stem matches any of these directory names):\n" + dir_list
        )

    file_stems_section = ""
    if existing_file_stems:
        stem_list = "\n".join(f"  - {s}.py" for s in sorted(existing_file_stems))
        file_stems_section = (
            "\n\nExisting flat modules (do NOT propose a subdirectory whose "
            "top-level name matches any of these stems):\n" + stem_list
        )

    original_basename = Path(original_path).name
    content = (
        "The following entity groups have conflicting target filenames that "
        "would create Python import name collisions (a flat module 'foo.py' "
        "and a package directory 'foo/' share the same import name). "
        "Assign a NEW, non-conflicting target filename to each group.\n\n"
        f"Groups to rename (you MUST return a new target_file for every "
        f"group_id listed):\n{groups_text}\n\n"
        "Rules:\n"
        f"- You MUST assign a target_file to ALL {n_groups} group(s). "
        "Missing any group_id will cause the split to fail.\n"
        f"- Use filenames relative to the same directory "
        f"(e.g. 'utils.py'). Do NOT use '{original_basename}' "
        "(the original file being split).\n"
        "- A Python file 'foo.py' and a package directory 'foo/' share the "
        "same import name and cannot coexist.\n"
        "- Choose descriptive names based on what the entities do."
        f"{forbidden_section}{dir_stems_section}{file_stems_section}"
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
        _RENAME_CONFLICTS_TOOL,
        "rename_conflicting_placements",
        messages,
        caller="FileLimiter",
        tool_choice_override=config.tool_choice,
    )
    return _extract_placements_from_result(chunk, forbidden_files, n_groups, result)


def _extract_placements_from_result(chunk, forbidden_files, n_groups, result):
    init_result = _init_placements_from_result(result)
    if init_result is None:
        return None

    placements, placed_ids = init_result
    for item in result.get("placements", []):
        gid = item.get("group_id")
        target = item.get("target_file", "")
        if (
            isinstance(gid, int)
            and 0 <= gid < len(chunk)
            and gid not in placed_ids
            and target
        ):
            if target in forbidden_files:
                return None
            placements.append(
                GroupPlacement(group=chunk[gid].group, target_file=target)
            )
            placed_ids.add(gid)

    if len(placements) != n_groups:
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


def resolve_naming_conflicts(
    placements: List[GroupPlacement],
    classified: ClassifiedEntities,
    original_path: str,
    existing_files: frozenset,
    existing_dirs: frozenset,
    config: CrispenConfig,
) -> Optional[List[GroupPlacement]]:
    """Attempt a targeted rename of only the placements with naming conflicts.

    Returns the updated placement list on success, or ``None`` if the LLM
    exhausts all retries without producing valid renames.
    :class:`CrispenAPIError` from ``get_api_key`` is propagated to the caller.
    """
    conflict_idxs = _find_conflicting_placement_indices(
        placements, existing_files, existing_dirs
    )
    if not conflict_idxs:
        return list(placements)

    conflict_idx_set = set(conflict_idxs)
    conflicting = [placements[i] for i in conflict_idxs]

    forbidden_files = (
        existing_files
        | frozenset(
            placements[i].target_file
            for i in range(len(placements))
            if i not in conflict_idx_set
        )
        | frozenset({Path(original_path).name})
    )
    forbidden_dir_stems = existing_dirs
    existing_file_stems = frozenset(Path(f).stem for f in existing_files)

    api_key = get_api_key(config.provider, caller="FileLimiter")
    client = make_client(
        config.provider, api_key, timeout=config.api_timeout, base_url=config.base_url
    )

    all_renamed: List[GroupPlacement] = []
    for chunk_start in range(0, len(conflicting), _PLACEMENT_CHUNK_SIZE):
        chunk = conflicting[chunk_start : chunk_start + _PLACEMENT_CHUNK_SIZE]
        chunk_result: Optional[List[GroupPlacement]] = None
        prev_failure = ""
        for _ in range(1 + config.file_limiter_retries):
            chunk_result = _rename_conflicting_chunk(
                chunk,
                classified,
                original_path,
                forbidden_files,
                forbidden_dir_stems,
                existing_file_stems,
                client,
                config,
                prev_failure=prev_failure,
            )
            if chunk_result is not None:
                break
            prev_failure = (
                "Your previous response was incomplete or used a forbidden name. "
                "Please try again with a valid, non-conflicting target filename "
                "for every group."
            )
        if chunk_result is None:
            return None
        all_renamed.extend(chunk_result)

    # Merge back: replace conflicting slots, leave non-conflicting unchanged.
    result = list(placements)
    for renamed, idx in zip(all_renamed, conflict_idxs):
        result[idx] = renamed
    return result


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
