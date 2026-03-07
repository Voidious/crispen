from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ...config import CrispenConfig
from ...llm_client import call_with_tool
from ..classifier import ClassifiedEntities
from ..entity_parser import Entity
from .set3_advisor import (
    _PLACEMENT_CHUNK_SIZE,
    _PLACEMENT_TOOL,
    _build_group_mermaid,
    _group_summary,
    _propose_files_step,
)


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
    llm_calls: int = 0  # number of LLM API calls made during planning
    original_target_files: int = 0  # computed target file count (before bonus)


def _compute_projected_lines(
    placements: List[GroupPlacement],
    entity_map: Dict[str, Entity],
) -> Dict[str, int]:
    """Return projected line count per target filename."""
    projected: Dict[str, int] = {}
    for p in placements:
        for name in p.group:
            ent = entity_map.get(name)
            if ent:
                size = ent.end_line - ent.start_line + 1
                projected[p.target_file] = projected.get(p.target_file, 0) + size
    return projected


def _assign_placements_chunk(
    chunk: List[List[str]],
    classified: ClassifiedEntities,
    original_path: str,
    existing_files: frozenset,
    client: object,
    config: CrispenConfig,
    prev_failure: str = "",
    min_files: int = 2,
    verbose: bool = False,
    _counter: Optional[List[int]] = None,
    subdir_name: Optional[str] = None,
    proposed_files: Optional[List[Tuple[str, str]]] = None,
) -> Optional[List[GroupPlacement]]:
    """Ask the LLM to assign filenames to one chunk of groups.

    When *proposed_files* is provided the assignment is constrained: the LLM
    must choose from that list only, and any response that uses a filename
    outside the list is rejected.  When *proposed_files* is ``None`` the
    original free-form behaviour is used.

    Groups within *chunk* are numbered 0…N-1 for this call.
    Returns ``None`` on failure (LLM error, missing group, or invalid target).
    """
    entity_map = {e.name: e for e in classified.entities}
    group_lines = []
    for idx, group in enumerate(chunk):
        summary = _group_summary(group, entity_map)
        group_lines.append(f"  [{idx}]: {summary}")
    groups_text = "\n".join(group_lines)

    n_groups = len(chunk)
    original_basename = Path(original_path).name

    if proposed_files is not None:
        # Constrained assignment: pick from proposed list only.
        file_list_lines = []
        for fname, fdesc in proposed_files:
            file_list_lines.append(f'  {fname} \u2014 "{fdesc}"')
        file_list_text = "\n".join(file_list_lines)
        content = (
            "Assign each entity group to one of the following proposed output "
            "files.\n\n"
            f"Proposed output files:\n{file_list_text}\n\n"
            f"Groups to assign:\n{groups_text}\n\n"
            "Rules:\n"
            f"- You MUST assign a target_file to ALL {n_groups} group(s). "
            "Missing any group_id will cause the split to fail.\n"
            "- Use ONLY the filenames listed in 'Proposed output files' above. "
            "Do NOT invent new filenames.\n"
            "- Multiple groups MAY share the same target file — place "
            "semantically related entities together.\n"
            "- Choose based on semantic fit: what the entity does vs what "
            "the file is described to contain."
        )
    else:
        exclude_section = ""
        if existing_files:
            file_list = "\n".join(f"  - {f}" for f in sorted(existing_files))
            exclude_section = (
                "\n\nThe following files already exist — do NOT use them as targets:\n"
                + file_list
            )
        if subdir_name:
            placement_rule = (
                f"- All target files will be placed inside the new "
                f"'{subdir_name}/' subdirectory, so filenames should be "
                f"relative to that subdirectory (e.g. 'utils.py'). "
                f"Do NOT repeat '{subdir_name}' in the filename — "
                f"it is already provided by the directory. "
                f"Do NOT use '{original_basename}' (the original file being split).\n"
            )
        else:
            placement_rule = (
                f"- Use filenames relative to the same directory "
                f"(e.g. 'utils.py'). Do NOT use '{original_basename}' "
                "(the original file being split).\n"
            )
        content = (
            f"Assign each entity group to a target Python filename. "
            f"The original file is '{original_path}' and is being split because "
            f"it exceeds the {config.max_file_lines}-line limit — "
            "each group must go to a file that does NOT already exist "
            "(a new file created by this split).\n\n"
            f"Groups to place (you MUST return a target_file for every "
            f"group_id listed):\n{groups_text}\n\n"
            "Rules:\n"
            f"- You MUST assign a target_file to ALL {n_groups} group(s). "
            "Missing any group_id will cause the split to fail.\n"
            f"{placement_rule}"
            "- Multiple groups MAY share the same target file — place "
            "semantically related groups together rather than giving each "
            "its own file.\n"
            "- PREFER FEWER, BROADER FILES. Group loosely-related entities "
            "under a shared name like 'utils.py' or 'misc.py' instead of "
            "creating a separate file for every small group. Only create a "
            "dedicated file when a group is large or has a clearly distinct "
            "purpose that warrants its own module.\n"
            f"- The set of target files across all placements should use "
            f"at least {min_files} distinct filename(s).\n"
            "- Choose descriptive names based on what the entities do "
            "(e.g. 'helpers.py', 'models.py', 'extractors.py')."
            f"{exclude_section}"
        )

    mermaid_text = _build_group_mermaid(chunk, classified)
    if mermaid_text:
        content += f"\n\nDependency graph between groups:\n{mermaid_text}"
    if prev_failure:
        content += f"\n\nFeedback from the previous attempt: {prev_failure}"
    messages = [{"role": "user", "content": content}]
    max_tokens = max(512, 20 + n_groups * 20)
    if verbose:
        print(
            f"crispen: FileLimiter: asking LLM to assign file placements"
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
        _PLACEMENT_TOOL,
        "assign_file_placements",
        messages,
        caller="FileLimiter",
        tool_choice_override=config.tool_choice,
    )
    if result is None:
        return None

    proposed_filenames = (
        {f for f, _ in proposed_files} if proposed_files is not None else None
    )

    placements: List[GroupPlacement] = []
    placed_ids: set = set()
    for item in result.get("placements", []):
        gid = item.get("group_id")
        target = item.get("target_file", "")
        if subdir_name and target.startswith(subdir_name + "/"):
            target = target[len(subdir_name) + 1 :]
        if (
            isinstance(gid, int)
            and 0 <= gid < len(chunk)
            and gid not in placed_ids
            and target
        ):
            if proposed_filenames is not None:
                if target not in proposed_filenames:
                    return None
            else:
                if target in existing_files:
                    return None
            placements.append(GroupPlacement(group=chunk[gid], target_file=target))
            placed_ids.add(gid)

    if len(placements) != len(chunk):
        return None

    return placements


def _refine_merge_tiny(
    placements: List[GroupPlacement],
    proposed_files: List[Tuple[str, str]],
    classified: ClassifiedEntities,
    original_path: str,
    client: object,
    config: CrispenConfig,
    verbose: bool = False,
    _counter: Optional[List[int]] = None,
    subdir_name: Optional[str] = None,
) -> List[GroupPlacement]:
    """Reassign groups from tiny output files into larger files.

    A file is "tiny" when its projected line count is greater than zero but
    less than ``max(50, config.max_file_lines // 8)``.  Tiny files are
    merged into the remaining non-tiny proposed files via a single LLM call.

    This step is best-effort: if the LLM call fails the original placements
    are returned unchanged.
    """
    entity_map = {e.name: e for e in classified.entities}
    projected = _compute_projected_lines(placements, entity_map)
    min_size = max(50, config.max_file_lines // 8)

    tiny_set = {f for f, lines in projected.items() if 0 < lines < min_size}
    ok_proposed = [(f, d) for f, d in proposed_files if f not in tiny_set]

    if not tiny_set or not ok_proposed:
        return list(placements)

    tiny_groups = [p.group for p in placements if p.target_file in tiny_set]

    if verbose:
        print(
            f"crispen: FileLimiter: refining: merging {len(tiny_set)} tiny"
            f" file(s) into {len(ok_proposed)} file(s) in '{original_path}'",
            file=sys.stderr,
            flush=True,
        )

    reassigned = _assign_placements_chunk(
        tiny_groups,
        classified,
        original_path,
        frozenset(),
        client,
        config,
        verbose=verbose,
        _counter=_counter,
        subdir_name=subdir_name,
        proposed_files=ok_proposed,
    )
    if reassigned is None:
        return list(placements)

    reassigned_map: Dict[frozenset, str] = {
        frozenset(rp.group): rp.target_file for rp in reassigned
    }
    result: List[GroupPlacement] = []
    for p in placements:
        if p.target_file in tiny_set:
            new_target = reassigned_map.get(frozenset(p.group), p.target_file)
            result.append(GroupPlacement(group=p.group, target_file=new_target))
        else:
            result.append(p)
    return result


def _assign_placements(
    groups_to_place: List[List[str]],
    classified: ClassifiedEntities,
    original_path: str,
    existing_files: frozenset,
    client: object,
    config: CrispenConfig,
    prev_failure: str = "",
    verbose: bool = False,
    _counter: Optional[List[int]] = None,
    subdir_name: Optional[str] = None,
    target_files: int = 2,
) -> Optional[List[GroupPlacement]]:
    """Ask the LLM to assign filenames to each group. Returns None on failure.

    Uses a three-step approach for reliable, compact results:

    1. **Propose** — ask the LLM to name a target-sized set of output files
       (target = ceil(total_lines / max_file_lines)), giving it a global view
       of all entities before any assignment decision is made.
    2. **Assign** — ask the LLM to assign each group to one of the proposed
       files.  The response is constrained to the proposed list, preventing
       free-form proliferation of tiny files.
    3. **Refine** — merge groups assigned to files that are too small (< 20 %
       of the line limit) into the remaining larger files (best-effort).

    When there are more than :data:`_PLACEMENT_CHUNK_SIZE` groups the
    assignment step is split into multiple LLM calls of at most that many
    groups each, each with its own retry budget.  If any chunk fails all its
    attempts, ``None`` is returned immediately.
    """
    # Step 1: Propose output file set.
    proposed_files: Optional[List[Tuple[str, str]]] = None
    prev_propose_failure = ""
    for _ in range(1 + config.file_limiter_retries):
        proposed_files = _propose_files_step(
            groups_to_place,
            classified,
            original_path,
            target_files,
            existing_files,
            client,
            config,
            prev_failure=prev_propose_failure,
            verbose=verbose,
            _counter=_counter,
            subdir_name=subdir_name,
        )
        if proposed_files is not None:
            break
        prev_propose_failure = (
            "Your previous response was incomplete or contained no valid filenames. "
            "Please return a non-empty list of proposed output files."
        )
    if proposed_files is None:
        return None

    # Step 2: Assign groups to proposed files (chunked).
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
                min_files=2,
                verbose=verbose,
                _counter=_counter,
                subdir_name=subdir_name,
                proposed_files=proposed_files,
            )
            if chunk_placements is not None:
                break
        if chunk_placements is None:
            return None
        all_placements.extend(chunk_placements)

    # Step 3: Refine — merge tiny output files (best-effort).
    all_placements = _refine_merge_tiny(
        all_placements,
        proposed_files,
        classified,
        original_path,
        client,
        config,
        verbose=verbose,
        _counter=_counter,
        subdir_name=subdir_name,
    )

    return all_placements
