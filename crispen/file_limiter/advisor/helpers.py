from __future__ import annotations
from typing import Dict, List, Optional, TYPE_CHECKING
import sys
from ...config import CrispenConfig
from ...llm_client import call_with_tool
from ..classifier import ClassifiedEntities
from ..entity_parser import Entity
from .models import GroupPlacement, _SET3_TOOL


if TYPE_CHECKING:
    from .models import _LLMAccumulator


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
    max_tokens = max(512, 20 + n_groups * 25)
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
