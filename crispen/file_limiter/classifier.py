"""Classify top-level entities into sets 1, 2, and 3 for the FileLimiter refactor."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Tuple

from .dep_graph import build_dep_graph, find_sccs
from .entity_parser import Entity, EntityKind, parse_entities


class EntityClass(Enum):
    """Classification of an entity relative to the diff."""

    UNMODIFIED = "unmodified"  # existed before the diff, no lines changed
    MODIFIED = "modified"  # existed before the diff, at least one line changed
    NEW = "new"  # only present in the post-refactor source


@dataclass
class ClassifiedEntities:
    """Output of :func:`classify_entities`."""

    entities: List[Entity]  # all entities from post-refactor source
    entity_class: Dict[str, EntityClass]  # entity name → classification
    graph: Dict[str, Set[str]]  # name-reference dependency graph
    set_1: List[str]  # entity names that must stay in the original file
    set_2_groups: List[List[str]]  # SCC groups to move to new files
    set_3_groups: List[List[str]]  # SCC groups that may be split or migrated
    abort: bool  # True → all entities form one SCC; file cannot be split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity_overlaps_diff(entity: Entity, diff_ranges: List[Tuple[int, int]]) -> bool:
    """Return True if *entity*'s line span overlaps any diff range."""
    for start, end in diff_ranges:
        if entity.start_line <= end and entity.end_line >= start:
            return True
    return False


def _assign_sccs(
    sccs: List[List[str]],
    entity_class: Dict[str, EntityClass],
) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """Assign each SCC to set_1, set_2_groups, or set_3_groups.

    Priority rules (applied per SCC):

    1. If the SCC contains any **UNMODIFIED** entity, the entire SCC goes to
       *set_1* (it must stay in the original file).
    2. Otherwise, if it contains any **NEW** entity, the entire SCC goes to
       *set_2_groups* (it can be moved to a new file).
    3. Otherwise (all **MODIFIED**) the SCC goes to *set_3_groups* for
       potential LLM-guided splitting or migration.
    """
    set_1: List[str] = []
    set_2_groups: List[List[str]] = []
    set_3_groups: List[List[str]] = []
    for scc in sccs:
        scc_classes = {entity_class[name] for name in scc}
        if EntityClass.UNMODIFIED in scc_classes:
            set_1.extend(scc)
        elif EntityClass.NEW in scc_classes:
            set_2_groups.append(scc)
        else:
            set_3_groups.append(scc)
    return set_1, set_2_groups, set_3_groups


def _is_import_only_entity(entity: Entity, source_lines: List[str]) -> bool:
    """Return True if *entity* consists solely of import statements and/or
    module-level string literals (e.g. a module docstring).

    Such entities must always remain in the original file: they define no
    callable symbols that are useful in a standalone module, and removing
    them from the original strips all imports, breaking the file.
    """
    entity_src = "".join(source_lines[entity.start_line - 1 : entity.end_line])
    try:
        tree = ast.parse(entity_src)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_entities(
    original_source: str,
    post_refactor_source: str,
    diff_ranges: List[Tuple[int, int]],
) -> ClassifiedEntities:
    """Classify post-refactor entities and assign them to sets 1, 2, or 3.

    Each entity in *post_refactor_source* is labelled:

    * **NEW** — name is absent from *original_source*.
    * **MODIFIED** — name exists in *original_source* and at least one of its
      lines falls within *diff_ranges*.
    * **UNMODIFIED** — name exists in *original_source* and no lines overlap
      with *diff_ranges*.

    Entities are then grouped into strongly connected components (SCCs) and
    assigned:

    * **set_1** — SCCs that contain any UNMODIFIED entity (must stay in the
      original file to preserve existing callers and avoid circular imports).
    * **set_2_groups** — SCCs (containing at least one NEW entity, and any
      MODIFIED entities sharing the cycle) that can be moved to new files.
    * **set_3_groups** — SCCs of purely MODIFIED entities that a later phase
      may split or migrate to bring the original file under the line limit.

    :attr:`ClassifiedEntities.abort` is ``True`` when all entities form a
    single SCC (the file cannot be cleanly split).
    """
    entities = parse_entities(post_refactor_source)
    original_names = {e.name for e in parse_entities(original_source)}
    graph = build_dep_graph(entities, post_refactor_source)
    sccs = find_sccs(graph)

    entity_class: Dict[str, EntityClass] = {}
    for entity in entities:
        if entity.name not in original_names:
            entity_class[entity.name] = EntityClass.NEW
        elif _entity_overlaps_diff(entity, diff_ranges):
            entity_class[entity.name] = EntityClass.MODIFIED
        else:
            entity_class[entity.name] = EntityClass.UNMODIFIED

    # Import-only TOP_LEVEL entities must always stay in the original file.
    # Moving them would strip all imports from the original, breaking it.
    source_lines = post_refactor_source.splitlines(keepends=True)
    for entity in entities:
        if entity.kind == EntityKind.TOP_LEVEL and _is_import_only_entity(
            entity, source_lines
        ):
            entity_class[entity.name] = EntityClass.UNMODIFIED

    # Abort when the entire file is one strongly connected component.
    abort = len(entities) >= 2 and len(sccs) == 1

    set_1, set_2_groups, set_3_groups = _assign_sccs(sccs, entity_class)

    return ClassifiedEntities(
        entities=entities,
        entity_class=entity_class,
        graph=graph,
        set_1=set_1,
        set_2_groups=set_2_groups,
        set_3_groups=set_3_groups,
        abort=abort,
    )
