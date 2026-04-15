from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from ...config import CrispenConfig
from ...llm_client import call_with_tool
from ..classifier import ClassifiedEntities
from ..entity_parser import Entity
from .llm_schemas import (
    _PLACEMENT_CHUNK_SIZE,
    _PLACEMENT_TOOL,
    _PROPOSE_FILES_TOOL,
    _SET3_TOOL,
)


@dataclass
class _LLMAccumulator:
    """Mutable accumulator for LLM call counts, timing, and token usage."""

    calls: int = 0
    elapsed: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


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
    llm_elapsed: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0


def _group_summary(group: List[str], entity_map: Dict[str, Entity]) -> str:
    """Return a brief text description of an SCC group for LLM context."""
    parts = []
    for name in group:
        ent = entity_map.get(name)
        if ent:
            size = ent.end_line - ent.start_line + 1
            desc = f"{name} ({size} lines)"
            extras = []
            if ent.section_header:
                extras.append(f'section: "{ent.section_header}"')
            if ent.docstring:
                flat = ent.docstring.replace("\n", " ")
                idx = flat.find(". ")
                first = flat[: idx + 1] if idx >= 0 else flat
                extras.append(f'"{first}"')
            if ent.params:
                extras.append(f"params: {', '.join(ent.params)}")
            if extras:
                desc += " \u2014 " + "; ".join(extras)
            parts.append(desc)
        else:
            parts.append(name)
    return ", ".join(parts)


def _build_group_mermaid(chunk: List[List[str]], classified: ClassifiedEntities) -> str:
    """Return a Mermaid graph showing inter-group dependencies, or '' if none."""
    entity_to_gid = {name: gid for gid, group in enumerate(chunk) for name in group}
    edges: set = set()
    for gid, group in enumerate(chunk):
        for entity_name in group:
            for dep_name in classified.graph.get(entity_name, set()):
                dep_gid = entity_to_gid.get(dep_name)
                if dep_gid is not None and dep_gid != gid:
                    edges.add((gid, dep_gid))
    if not edges:
        return ""
    lines = ["```mermaid", "graph TD"]
    for g_from, g_to in sorted(edges):
        lines.append(f"    G{g_from} --> G{g_to}")
    lines.append("```")
    return "\n".join(lines)


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


def _advise_set3(
    classified: ClassifiedEntities,
    original_path: str,
    client: object,
    config: CrispenConfig,
    prev_failure: str = "",
    verbose: bool = False,
    timing: str = "detailed",
    _acc: Optional["_LLMAccumulator"] = None,
) -> Optional[List[List[str]]]:
    """Ask the LLM which Set 3 groups should migrate. Returns None on failure."""
    entity_map = {e.name: e for e in classified.entities}
    group_lines = []
    for idx, group in enumerate(classified.set_3_groups):
        summary = _group_summary(group, entity_map)
        group_lines.append(f"  [{idx}]: {summary}")
    groups_text = "\n".join(group_lines)

    mermaid_text = _build_group_mermaid(classified.set_3_groups, classified)
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
        "CIRCULAR IMPORT CONSTRAINT: A migrated group cannot safely reference "
        "names defined by groups that stay in the original — this creates a "
        "circular import (the new file imports from the original while the "
        "original also imports from the new file). You cannot migrate group A "
        "if any group that A depends on is staying. To migrate A, either also "
        "migrate everything A depends on (to the same or a compatible file), "
        "or keep A in the original. Migrating a dependency while leaving the "
        "dependent in the original is always safe. Groups with no outgoing "
        "arrows in the dependency graph (leaf groups) can always be migrated "
        "independently.\n\n"
        "For each group, return 'migrate' (preferred) or 'stay' (exceptional)."
    )
    if mermaid_text:
        content += f"\n\nDependency graph between groups:\n{mermaid_text}"
    if prev_failure:
        content += f"\n\nFeedback from the previous attempt: {prev_failure}"
    messages = [{"role": "user", "content": content}]
    max_tokens = max(512, 20 + n_groups * 15)
    if verbose:
        print(
            f"crispen: FileLimiter: asking LLM whether to migrate"
            f" {n_groups} set-3 group(s) in '{original_path}'",
            file=sys.stderr,
            flush=True,
        )
    if _acc is not None:
        _acc.calls += 1
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
        rate_limit_retries=config.rate_limit_retries,
        rate_limit_backoff=config.rate_limit_backoff,
    )
    if _acc is not None:
        _acc.elapsed += result.elapsed
        _acc.input_tokens += result.input_tokens
        _acc.output_tokens += result.output_tokens
    if verbose and timing == "detailed":
        print(
            f"crispen: FileLimiter:   → done [{result.elapsed:.2f}s,"
            f" {result.input_tokens:,} in / {result.output_tokens:,} out]",
            file=sys.stderr,
            flush=True,
        )
    if result.tool_input is None:
        return None

    migrate_ids = set()
    for decision in result.tool_input.get("decisions", []):
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


def _propose_files_step(
    groups_to_place: List[List[str]],
    classified: ClassifiedEntities,
    original_path: str,
    target_files: int,
    existing_files: frozenset,
    client: object,
    config: CrispenConfig,
    prev_failure: str = "",
    verbose: bool = False,
    timing: str = "detailed",
    _acc: Optional["_LLMAccumulator"] = None,
    subdir_name: Optional[str] = None,
) -> Optional[List[Tuple[str, str]]]:
    """Ask the LLM to propose a set of output filenames.

    Returns a list of (filename, description) pairs, or None on failure.
    The LLM is given a target file count of *target_files* (= ceiling of
    2 * total_lines / max_file_lines) so output files average half the line
    limit, keeping well-sized files even when entity sizes are uneven.
    """
    entity_map = {e.name: e for e in classified.entities}
    group_lines = []
    for idx, group in enumerate(groups_to_place):
        summary = _group_summary(group, entity_map)
        group_lines.append(f"  [{idx}]: {summary}")
    groups_text = "\n".join(group_lines)

    mermaid_text = _build_group_mermaid(groups_to_place, classified)
    exclude_section = ""
    if existing_files:
        file_list = "\n".join(f"  - {f}" for f in sorted(existing_files))
        exclude_section = (
            "\n\nThe following files already exist — do NOT propose them:\n" + file_list
        )

    original_basename = Path(original_path).name
    n_groups = len(groups_to_place)
    if subdir_name:
        placement_rule = (
            f"- All files will be placed inside the '{subdir_name}/' subdirectory. "
            f"Use filenames relative to that subdirectory (e.g. 'utils.py'). "
            f"Do NOT repeat '{subdir_name}' in the filename. "
            f"Do NOT propose '{original_basename}'.\n"
        )
    else:
        placement_rule = (
            f"- Use filenames relative to the same directory "
            f"(e.g. 'utils.py'). Do NOT propose '{original_basename}'.\n"
        )

    content = (
        f"The file '{original_path}' is being split because it exceeds the "
        f"{config.max_file_lines}-line limit. Propose the set of output files "
        f"to hold the {n_groups} entity group(s) listed below.\n\n"
        f"Entity groups to distribute:\n{groups_text}\n\n"
        f"Target: approximately {target_files} output file(s) "
        f"(= ceil(total_entity_lines / {config.max_file_lines})).\n\n"
        "Rules:\n"
        f"- Propose at least {max(2, target_files)} distinct filenames.\n"
        f"{placement_rule}"
        "- Name files descriptively based on the entities they will hold "
        "(e.g. 'models.py', 'utils.py', 'handlers.py').\n"
        "- Include a one-sentence description of what each file will contain.\n"
        "- Aim for semantically cohesive files — group related entities together.\n"
        "- PREFER FEWER, BROADER FILES. "
        f"Only propose as many files as needed to stay under "
        f"{config.max_file_lines} lines each. "
        "Do NOT create a separate file for every small group — "
        "combine loosely-related entities under a shared name like "
        "'utils.py' or 'misc.py'."
        f"{exclude_section}"
    )
    if mermaid_text:
        content += (
            "\n\nDependency graph between groups — groups connected by arrows "
            "have inter-dependencies. Prefer file names that reflect these "
            "clusters; mutually-dependent groups must land in the same file "
            "to avoid circular imports:\n" + mermaid_text
        )
    if prev_failure:
        content += f"\n\nFeedback from the previous attempt: {prev_failure}"
    messages = [{"role": "user", "content": content}]
    max_tokens = max(512, 30 + target_files * 60)
    if verbose:
        print(
            f"crispen: FileLimiter: asking LLM to propose output file set"
            f" for {n_groups} group(s) in '{original_path}'",
            file=sys.stderr,
            flush=True,
        )
    if _acc is not None:
        _acc.calls += 1
    result = call_with_tool(
        client,
        config.provider,
        config.model,
        max_tokens,
        _PROPOSE_FILES_TOOL,
        "propose_output_files",
        messages,
        caller="FileLimiter",
        tool_choice_override=config.tool_choice,
        rate_limit_retries=config.rate_limit_retries,
        rate_limit_backoff=config.rate_limit_backoff,
    )
    if _acc is not None:
        _acc.elapsed += result.elapsed
        _acc.input_tokens += result.input_tokens
        _acc.output_tokens += result.output_tokens
    if verbose and timing == "detailed":
        print(
            f"crispen: FileLimiter:   → done [{result.elapsed:.2f}s,"
            f" {result.input_tokens:,} in / {result.output_tokens:,} out]",
            file=sys.stderr,
            flush=True,
        )
    if result.tool_input is None:
        if verbose:
            print(
                "crispen: FileLimiter:   propose failed: no tool call in response",
                file=sys.stderr,
                flush=True,
            )
        return None

    raw_files = result.tool_input.get("files", [])
    proposed: List[Tuple[str, str]] = []
    seen: set = set()
    filtered_names: List[str] = []
    for item in raw_files:
        filename = item.get("filename", "")
        description = item.get("description", "")
        if filename and filename not in seen and filename not in existing_files:
            proposed.append((filename, description))
            seen.add(filename)
        elif filename:
            filtered_names.append(filename)

    if not proposed:
        if verbose:
            if not raw_files:
                print(
                    "crispen: FileLimiter:   propose failed: LLM returned empty"
                    " files list",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"crispen: FileLimiter:   propose failed: all"
                    f" {len(filtered_names)} filename(s) filtered"
                    f" (existing or duplicate): {filtered_names}",
                    file=sys.stderr,
                    flush=True,
                )
        return None
    return proposed


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
    timing: str = "detailed",
    _acc: Optional["_LLMAccumulator"] = None,
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

    content += (
        "\n\nAvoid creating circular dependencies between output files — "
        "assignments will be validated and retried if circular imports are detected."
    )
    mermaid_text = _build_group_mermaid(chunk, classified)
    if mermaid_text:
        content += f"\n\nDependency graph between groups:\n{mermaid_text}"
    if prev_failure:
        content += f"\n\nFeedback from the previous attempt: {prev_failure}"
    messages = [{"role": "user", "content": content}]
    max_tokens = max(512, 100 + n_groups * 40)
    if verbose:
        print(
            f"crispen: FileLimiter: asking LLM to assign file placements"
            f" for {n_groups} group(s) in '{original_path}'",
            file=sys.stderr,
            flush=True,
        )
    if _acc is not None:
        _acc.calls += 1
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
        rate_limit_retries=config.rate_limit_retries,
        rate_limit_backoff=config.rate_limit_backoff,
    )
    if _acc is not None:
        _acc.elapsed += result.elapsed
        _acc.input_tokens += result.input_tokens
        _acc.output_tokens += result.output_tokens
    if verbose and timing == "detailed":
        print(
            f"crispen: FileLimiter:   → done [{result.elapsed:.2f}s,"
            f" {result.input_tokens:,} in / {result.output_tokens:,} out]",
            file=sys.stderr,
            flush=True,
        )
    if result.tool_input is None:
        return None

    proposed_filenames = (
        {f for f, _ in proposed_files} if proposed_files is not None else None
    )

    placements: List[GroupPlacement] = []
    placed_ids: set = set()
    for item in result.tool_input.get("placements", []):
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
    timing: str = "detailed",
    _acc: Optional["_LLMAccumulator"] = None,
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
        timing=timing,
        _acc=_acc,
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
    timing: str = "detailed",
    _acc: Optional["_LLMAccumulator"] = None,
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
            timing=timing,
            _acc=_acc,
            subdir_name=subdir_name,
        )
        if proposed_files is not None:
            break
        prev_propose_failure = (
            "Your previous response returned no valid filenames — either the"
            " files list was empty or every name was already in use. "
            "You MUST return a non-empty files list with new, unique filenames"
            " that are not on the exclusion list."
        )
    if proposed_files is None:
        return None

    # Step 2: Assign groups to proposed files (chunked).
    all_placements: List[GroupPlacement] = []
    for chunk_start in range(0, len(groups_to_place), _PLACEMENT_CHUNK_SIZE):
        chunk = groups_to_place[chunk_start : chunk_start + _PLACEMENT_CHUNK_SIZE]
        chunk_placements: Optional[List[GroupPlacement]] = None
        chunk_prev_failure = prev_failure
        for _ in range(1 + config.file_limiter_retries):
            chunk_placements = _assign_placements_chunk(
                chunk,
                classified,
                original_path,
                existing_files,
                client,
                config,
                chunk_prev_failure,
                min_files=2,
                verbose=verbose,
                timing=timing,
                _acc=_acc,
                subdir_name=subdir_name,
                proposed_files=proposed_files,
            )
            if chunk_placements is not None:
                break
            chunk_prev_failure = (
                f"Your previous response did not include placements for all "
                f"{len(chunk)} group(s). You MUST return a target_file for "
                f"every group_id from 0 to {len(chunk) - 1}."
            )
        if chunk_placements is None:
            if verbose:
                print(
                    f"crispen: FileLimiter: failed to assign file placements"
                    f" after {1 + config.file_limiter_retries} attempt(s)",
                    file=sys.stderr,
                    flush=True,
                )
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
        timing=timing,
        _acc=_acc,
        subdir_name=subdir_name,
    )

    return all_placements
