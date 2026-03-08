from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
from ...config import CrispenConfig
from ...llm_client import call_with_tool, get_api_key, make_client
from ..classifier import ClassifiedEntities
from .llm_steps import (
    GroupPlacement,
    _PLACEMENT_CHUNK_SIZE,
    _RENAME_CONFLICTS_TOOL,
    _group_summary,
)
import sys


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
    verbose: bool = False,
    _counter: Optional[List[int]] = None,
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
    if verbose:
        print(
            f"crispen: FileLimiter: asking LLM to resolve naming conflicts"
            f" for {n_groups} group(s) in '{original_path}'",
            file=sys.stderr,
            flush=True,
        )
    if _counter is not None:
        _counter[0] += 1
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
            if target in forbidden_files:
                return None
            placements.append(
                GroupPlacement(group=chunk[gid].group, target_file=target)
            )
            placed_ids.add(gid)

    if len(placements) != n_groups:
        return None

    return placements


def resolve_naming_conflicts(
    placements: List[GroupPlacement],
    classified: ClassifiedEntities,
    original_path: str,
    existing_files: frozenset,
    existing_dirs: frozenset,
    config: CrispenConfig,
    verbose: bool = False,
    _counter: Optional[List[int]] = None,
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
                verbose=verbose,
                _counter=_counter,
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
