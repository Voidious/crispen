"""LLM advisor for FileLimiter: plans entity migration to new files."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    llm_calls: int = 0  # number of LLM API calls made during planning


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

_PROPOSE_FILES_TOOL: dict = {
    "name": "propose_output_files",
    "description": (
        "Propose the set of Python files to create when splitting a large module. "
        "Return exactly the files you plan to use, with descriptive names and "
        "a brief summary of what each file will contain."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Python filename, e.g. 'utils.py'",
                        },
                        "description": {
                            "type": "string",
                            "description": "What this file will contain",
                        },
                    },
                    "required": ["filename", "description"],
                },
            }
        },
        "required": ["files"],
    },
}


# Maximum number of groups per placement LLM call.  Large files may have
# dozens of groups; sending them all in one call frequently causes timeouts
# or incomplete responses.  This limit keeps each call small and reliable.
_PLACEMENT_CHUNK_SIZE = 100


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
            desc = f"{name} ({size} lines)"
            extras = []
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
    _counter: Optional[List[int]] = None,
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
    if _counter is not None:
        _counter[0] += 1
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
    _counter: Optional[List[int]] = None,
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
    max_tokens = max(512, 30 + target_files * 40)
    if verbose:
        print(
            f"crispen: FileLimiter: asking LLM to propose output file set"
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
        _PROPOSE_FILES_TOOL,
        "propose_output_files",
        messages,
        caller="FileLimiter",
        tool_choice_override=config.tool_choice,
    )
    if result is None:
        return None

    proposed: List[Tuple[str, str]] = []
    seen: set = set()
    for item in result.get("files", []):
        filename = item.get("filename", "")
        description = item.get("description", "")
        if filename and filename not in seen and filename not in existing_files:
            proposed.append((filename, description))
            seen.add(filename)

    if not proposed:
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


def advise_file_limiter(
    classified: ClassifiedEntities,
    original_path: str,
    config: CrispenConfig,
    existing_files: frozenset = frozenset(),
    prev_set3_failure: str = "",
    prev_placement_failure: str = "",
    verbose: bool = False,
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

    counter: List[int] = [0]

    # Call 1: advise Set 3 groups (only if set_3 is non-empty).
    set3_migrate: List[List[str]] = []
    if classified.set_3_groups:
        result = _advise_set3(
            classified,
            original_path,
            client,
            config,
            prev_failure=prev_set3_failure,
            verbose=verbose,
            _counter=counter,
        )
        if result is None:
            return FileLimiterPlan(
                set3_migrate=[],
                placements=[],
                abort=True,
                abort_reason="LLM failed to plan set-3 groups",
                llm_calls=counter[0],
            )
        set3_migrate = result

    # Calls 2+: propose files, assign groups, refine (merge tiny).
    groups_to_place = classified.set_2_groups + set3_migrate
    if not groups_to_place:
        return FileLimiterPlan(
            set3_migrate=set3_migrate,
            placements=[],
            abort=False,
            llm_calls=counter[0],
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
        _counter=counter,
        subdir_name=subdir_name,
        target_files=original_target,
    )
    if placements is None:
        return FileLimiterPlan(
            set3_migrate=set3_migrate,
            placements=[],
            abort=True,
            abort_reason="LLM failed to assign file placements",
            llm_calls=counter[0],
        )

    return FileLimiterPlan(
        set3_migrate=set3_migrate,
        placements=placements,
        abort=False,
        llm_calls=counter[0],
    )
