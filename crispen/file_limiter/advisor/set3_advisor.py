from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ...config import CrispenConfig
from ...llm_client import call_with_tool
from ..classifier import ClassifiedEntities
from ..entity_parser import Entity

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
