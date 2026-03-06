from __future__ import annotations
from typing import Dict, List, Optional, Set
from ..advisor import GroupPlacement
from ..classifier import ClassifiedEntities
from ..dep_graph import find_sccs
from ..entity_parser import Entity, EntityKind
from .graph_utils import _topo_depth
from .import_utils import _collect_name_loads


def _extract_shared_helpers(
    file_entity_names: Dict[str, List[str]],
    entity_source_map: Dict[str, str],
    entity_map: Dict[str, Entity],
    classified: ClassifiedEntities,
    name_to_target_file: Dict[str, str],
    migrated_names: Set[str],
    original_basename: str,
) -> List[GroupPlacement]:
    """Extract non-migrated functions/classes referenced by migrated entities.

    When a migrated entity in new file F references a non-migrated function X
    from the original O, the generated ``from .O import X`` combined with O's
    re-export ``from .F import …`` creates a cycle O→F→O.

    Fix: pull X (and all helpers X transitively depends on) into a new file
    that uses them.  The destination is chosen using topological depth ordering:
    the inter-file dependency graph is built from migrated-entity cross-references
    first, then for each helper SCC the candidates (all files wanting the
    helpers) are sorted by topological depth (deepest / most-downstream first).
    For a DAG the deepest wanting file is always cycle-free on the first try;
    for non-DAG inputs (pre-existing cycles) _topo_depth falls back to 0 for
    all nodes and the loop exhausts all candidates via trial SCC analysis.
    If no cycle-free placement exists the SCC is left in the original file and
    the safety-net in :func:`generate_file_splits` will abort if the result is
    unloadable.

    Mutates *file_entity_names*, *migrated_names*, and *name_to_target_file*
    in place.  Returns synthetic :class:`GroupPlacement` objects for the
    extracted entities so that :func:`_add_re_exports` can re-import them from
    their new location in the updated original source.
    """
    # Build defined-name → entity-name map for non-migrated FUNCTION/CLASS entities.
    defined_to_entity: Dict[str, str] = {}
    for entity in classified.entities:
        if entity.name in migrated_names:
            continue
        if entity.kind not in (EntityKind.FUNCTION, EntityKind.CLASS):
            continue
        for defined_name in entity.names_defined:
            if name_to_target_file.get(defined_name) == original_basename:
                defined_to_entity[defined_name] = entity.name

    # Collect directly-wanted helpers: entity_name → set of target_files that want it.
    wanting: Dict[str, Set[str]] = {}
    for target_file, ent_names in list(file_entity_names.items()):
        for ent_name in ent_names:
            src = entity_source_map.get(ent_name, "")
            for ref_name in _collect_name_loads(src):
                entity_name = defined_to_entity.get(ref_name)
                if entity_name is not None:
                    wanting.setdefault(entity_name, set()).add(target_file)

    if not wanting:
        return []

    # Transitively expand wanting-sets to cover helpers referenced by
    # already-wanted helpers, preventing O→new-file→O cycles.
    # Re-queue a helper whenever its wanting-set gains new target files so that
    # the propagation reaches all transitive dependents.
    queue = list(wanting.keys())
    idx = 0
    while idx < len(queue):
        entity_name = queue[idx]
        idx += 1
        src = entity_source_map.get(entity_name, "")
        for ref_name in _collect_name_loads(src):
            dep_name = defined_to_entity.get(ref_name)
            if dep_name and wanting[entity_name] - wanting.get(dep_name, set()):
                wanting.setdefault(dep_name, set()).update(wanting[entity_name])
                queue.append(dep_name)

    # SCC analysis on the sub-graph of wanted helpers to co-locate
    # mutually-dependent helpers.
    sub_graph: Dict[str, Set[str]] = {
        name: {d for d in classified.graph.get(name, set()) if d in wanting}
        for name in wanting
    }
    sccs = find_sccs(sub_graph)

    # Build the initial inter-file dependency graph from migrated-entity
    # cross-references (before any helper placement).  This is the baseline for
    # the cycle-aware candidate selection below.
    file_deps: Dict[str, Set[str]] = {f: set() for f in file_entity_names}
    for target_file, ent_names in file_entity_names.items():
        for ent_name in ent_names:
            src = entity_source_map.get(ent_name, "")
            for ref_name in _collect_name_loads(src):
                dep_file = name_to_target_file.get(ref_name)
                if (
                    dep_file
                    and dep_file != target_file
                    and dep_file in file_entity_names
                ):
                    file_deps[target_file].add(dep_file)

    synthetic_placements: List[GroupPlacement] = []
    for scc in sccs:
        # Union of wanting-sets across this helper SCC.
        scc_wanting: Set[str] = set()
        for name in scc:
            scc_wanting.update(wanting.get(name, set()))

        # Sort candidates by topological depth (deepest / most-downstream first).
        # For a DAG the deepest wanting file is always cycle-free on the first try,
        # eliminating trial-and-error.  Depths are recomputed after each placement
        # because file_deps grows as helpers are extracted.
        topo_depth = _topo_depth(file_deps)
        candidates = sorted(scc_wanting, key=lambda t: topo_depth.get(t, 0))
        chosen: Optional[str] = None
        for candidate in candidates:
            trial_deps: Dict[str, Set[str]] = {
                f: set(deps) for f, deps in file_deps.items()
            }
            for wanting_file in scc_wanting:
                if wanting_file != candidate:
                    trial_deps[wanting_file].add(candidate)
            for helper_name in scc:
                src = entity_source_map.get(helper_name, "")
                for ref_name in _collect_name_loads(src):
                    dep_file = name_to_target_file.get(ref_name)
                    if (
                        dep_file
                        and dep_file != candidate
                        and dep_file in file_entity_names
                    ):
                        trial_deps[candidate].add(dep_file)
            if not any(len(s) > 1 for s in find_sccs(trial_deps)):
                chosen = candidate
                break

        if chosen is None:
            continue  # No cycle-free placement — leave helpers in original file.

        # Apply the chosen placement: update file_deps for subsequent SCC decisions.
        for wanting_file in scc_wanting:
            if wanting_file != chosen:
                file_deps[wanting_file].add(chosen)
        for helper_name in scc:
            src = entity_source_map.get(helper_name, "")
            for ref_name in _collect_name_loads(src):
                dep_file = name_to_target_file.get(ref_name)
                if dep_file and dep_file != chosen and dep_file in file_entity_names:
                    file_deps[chosen].add(dep_file)

        # Prepend extracted helpers so they appear before the functions that use them.
        file_entity_names[chosen] = list(scc) + file_entity_names[chosen]
        for entity_name in scc:
            migrated_names.add(entity_name)
            entity = entity_map[entity_name]
            for defined_name in entity.names_defined:
                name_to_target_file[defined_name] = chosen
        synthetic_placements.append(GroupPlacement(group=list(scc), target_file=chosen))
    return synthetic_placements
