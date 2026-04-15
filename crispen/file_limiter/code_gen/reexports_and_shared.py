from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import ast
from ..advisor import GroupPlacement
from ..classifier import ClassifiedEntities
from ..dep_graph import find_sccs
from ..entity_parser import Entity, EntityKind
from .cross_file_deps import _relative_import_prefix, _target_module_name
from .import_analysis import _collect_name_loads, _import_derived_names
from .main_handling import _class_has_test_methods, _is_test_name


@dataclass
class SplitResult:
    """Output of :func:`generate_file_splits`."""

    new_files: Dict[str, str]  # {target_file: source_code}
    original_source: str  # updated original file source
    abort: bool  # True if generation failed / nothing to split
    abort_reason: str = ""  # human-readable explanation when abort=True
    entity_name_rewrites: Dict[str, Dict[str, str]] = field(
        default_factory=dict
    )  # {entity_name: {old_name: new_qualified_name}} per migrated entity
    actual_placements: List[GroupPlacement] = field(
        default_factory=list
    )  # final placements after conftest routing (for accurate output messages)


def _add_re_exports(
    source: str,
    placements: List[GroupPlacement],
    entity_map: Dict[str, Entity],
    entity_source_map: Dict[str, str],
    external_loads: Set[str] = frozenset(),
    abs_pkg: Optional[str] = None,
    relative_from: Optional[str] = None,
    is_test_file: bool = False,
    reexport_mode: str = "always",
) -> str:
    """Add ``from .module import name`` imports for migrated entities.

    *reexport_mode* controls when public (non-underscore) names get a
    re-export stub:

    * ``"always"``      — always re-export every public name (default; most
      conservative, preserves the full public API regardless of usage).
    * ``"application"`` — re-export public names in non-test files only.
    * ``"imported"``    — re-export a public name only when it appears in
      *external_loads* (imported from the original module by another file in
      the project) or is still referenced in the remaining *source*.

    Private names (starting with ``_``) are always re-exported when the
    remaining *source* still references them, or when they appear in
    *external_loads*, regardless of *reexport_mode*.

    When *relative_from* is set (e.g. ``"service/__init__.py"``), import
    prefixes are computed via :func:`_relative_import_prefix` so that
    re-exports from a package ``__init__.py`` reference sibling modules
    correctly (e.g. ``from .utils import Foo`` instead of
    ``from .service.utils import Foo``).

    Import-derived names (names introduced by ``import`` / ``from … import``
    statements inside a TOP_LEVEL entity) are never re-exported: they were
    kept in the original file by :func:`_remove_entity_lines` and cannot
    meaningfully be re-exported from a new module.

    Inserts after the last import line in *source*.  Returns *source* unchanged
    when there are no names to import.
    """
    still_loaded = _collect_name_loads(source)
    re_exports: Dict[str, List[str]] = {}
    # Names added solely for external re-export (not referenced in remaining source).
    # These need "# fmt: skip # noqa: F401, E501" to suppress flake8 false positives.
    noqa_names: Set[str] = set()
    for placement in placements:
        # Compute the import prefix for this placement's target file.
        if relative_from is not None:
            import_prefix = _relative_import_prefix(
                relative_from, placement.target_file
            )
        elif abs_pkg is not None:
            module = _target_module_name(placement.target_file)
            import_prefix = f"{abs_pkg}.{module}" if abs_pkg else module
        else:
            module = _target_module_name(placement.target_file)
            import_prefix = f".{module}"
        to_import: List[str] = []
        for entity_name in placement.group:
            if entity_name in entity_map:
                entity = entity_map[entity_name]
                defined = entity.names_defined
                if entity.kind == EntityKind.TOP_LEVEL:
                    skip = _import_derived_names(entity_source_map.get(entity_name, ""))
                    defined = [n for n in defined if n not in skip]
            else:
                defined = [entity_name]
            is_test_class = entity_name in entity_map and _class_has_test_methods(
                entity_source_map.get(entity_name, "")
            )
            for defined_name in defined:
                # Test-named symbols (Test* / test_*) are never re-exported at
                # module level: _inject_inline_test_imports_original injects
                # them inside function/class bodies to prevent pytest from
                # discovering the same test twice.
                if _is_test_name(defined_name):
                    continue
                # Unconditional public re-export: only when reexport_mode
                # permits it for this file type.
                reexport_unconditionally = (
                    not defined_name.startswith("_")
                    and not defined_name.startswith("test_")
                    and not is_test_class
                    and (
                        reexport_mode == "always"
                        or (reexport_mode == "application" and not is_test_file)
                    )
                )
                if (
                    reexport_unconditionally
                    or defined_name in still_loaded
                    or defined_name in external_loads
                ):
                    to_import.append(defined_name)
                    # Add noqa when the name is not referenced in the remaining
                    # source (pure re-export stub), OR when it is in external_loads
                    # — in the latter case a non-migrated entity may currently use
                    # the name, but if that entity is itself migrated in a later
                    # recursive split the stub would become unreferenced and
                    # _prune_unused_imports would silently drop it, breaking the
                    # external caller.  The noqa marker protects against that.
                    if (
                        defined_name not in still_loaded
                        or defined_name in external_loads
                    ):
                        noqa_names.add(defined_name)
        if to_import:
            re_exports.setdefault(import_prefix, []).extend(to_import)

    if not re_exports:
        return source

    # Build export statements.  When a name is only there for external re-export
    # (not referenced in the remaining source), add "# fmt: skip # noqa: F401, E501"
    # so flake8 does not flag it as an unused import and Black does not reformat
    # the line (which would break the noqa directive).  Split mixed imports into
    # two lines so that the noqa comment does not suppress warnings for used names.
    export_stmts: List[str] = []
    for prefix, names in sorted(re_exports.items()):
        sorted_names = sorted(names)
        used = [n for n in sorted_names if n not in noqa_names]
        noqa = [n for n in sorted_names if n in noqa_names]
        if used:
            export_stmts.append(f"from {prefix} import {', '.join(used)}\n")
        for name in noqa:
            export_stmts.append(
                f"from {prefix} import {name}  # fmt: skip # noqa: F401, E501\n"
            )

    # In test files, add a single explanatory comment before the first F401 import.
    if is_test_file and noqa_names:
        first_noqa = next(i for i, s in enumerate(export_stmts) if "# noqa: F401" in s)
        export_stmts.insert(
            first_noqa,
            "# Re-exported for backwards compatibility with external callers.\n",
        )

    lines = source.splitlines(keepends=True)
    last_import_line = 0
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_line = max(last_import_line, node.end_lineno)

    insert_after = last_import_line
    if insert_after == 0 and tree.body:
        first = tree.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            insert_after = first.end_lineno

    return "".join(lines[:insert_after] + export_stmts + lines[insert_after:])


def _topo_depth(graph: Dict[str, Set[str]]) -> Dict[str, int]:
    """Return topological depth for each node in a DAG.

    Depth 0 = leaf (no outgoing edges).  A node's depth is 1 + the maximum
    depth of its dependencies.  All dependency nodes must be keys in *graph*.
    On non-DAG inputs (cycles detected), returns 0 for every node as a safe
    fallback so that callers degrade to arbitrary candidate ordering.
    """
    if any(len(s) > 1 for s in find_sccs(graph)):
        return {node: 0 for node in graph}
    depths: Dict[str, int] = {}

    def dfs(node: str) -> int:
        if node in depths:
            return depths[node]
        depths[node] = 1 + max((dfs(dep) for dep in graph[node]), default=-1)
        return depths[node]

    for node in graph:
        dfs(node)
    return depths


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
