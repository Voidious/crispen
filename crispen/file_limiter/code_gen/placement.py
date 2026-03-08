from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from ..advisor import GroupPlacement
from ..classifier import ClassifiedEntities
from ..dep_graph import find_sccs
from ..entity_parser import Entity, EntityKind
from .import_utils import (
    _collect_name_loads,
    _import_derived_names,
    _relative_import_prefix,
    _target_module_name,
)
import ast
import re


@dataclass
class SplitResult:
    """Output of :func:`generate_file_splits`."""

    new_files: Dict[str, str]  # {target_file: source_code}
    original_source: str  # updated original file source
    abort: bool  # True if generation failed / nothing to split
    abort_reason: str = ""  # human-readable explanation when abort=True


def _class_has_test_methods(entity_src: str) -> bool:
    """Return True if *entity_src* defines a class with any ``test_`` methods.

    Used to suppress re-exports of test classes: pytest discovers test classes
    by scanning the filesystem, so re-exporting them from the original file
    causes every test inside to run twice.
    """
    try:
        tree = ast.parse(entity_src)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("test_"):
                        return True
    return False


def _is_test_name(name: str) -> bool:
    """Return True if *name* matches pytest's test-discovery patterns.

    Pytest collects classes named ``Test*`` and functions named ``test_*``.
    Importing such names at module level in a test file causes every test
    inside to be discovered — and run — a second time.
    """
    return name.startswith("Test") or name.startswith("test_")


def _add_re_exports(
    source: str,
    placements: List[GroupPlacement],
    entity_map: Dict[str, Entity],
    entity_source_map: Dict[str, str],
    external_loads: Set[str] = frozenset(),
    abs_pkg: Optional[str] = None,
    relative_from: Optional[str] = None,
) -> str:
    """Add ``from .module import name`` imports for migrated entities.

    Public names are always re-exported so external callers can still import
    them from the original module.  Private names (starting with ``_``) are
    re-imported when the remaining *source* still references them, or when
    they appear in *external_loads* (names imported from the original module
    by other files in the project).

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
                if (
                    (
                        not defined_name.startswith("_")
                        and not defined_name.startswith("test_")
                        and not is_test_class
                    )
                    or defined_name in still_loaded
                    or (defined_name.startswith("_") and defined_name in external_loads)
                ):
                    to_import.append(defined_name)
                    if defined_name not in still_loaded:
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

    lines = source.splitlines(keepends=True)
    last_import_line = 0
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_line = max(last_import_line, node.end_lineno)

    return "".join(lines[:last_import_line] + export_stmts + lines[last_import_line:])


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


def _is_pytest_fixture(entity_src: str) -> bool:
    """Return True if *entity_src* defines a function with a @pytest.fixture decorator.

    Handles all common forms: ``@fixture``, ``@fixture()``, ``@pytest.fixture``,
    and ``@pytest.fixture(scope=...)``.
    """
    try:
        tree = ast.parse(entity_src)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                # Unwrap call forms like @pytest.fixture(...) to get the base reference.
                ref = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(ref, ast.Name) and ref.id == "fixture":
                    return True
                if isinstance(ref, ast.Attribute) and ref.attr == "fixture":
                    return True
    return False


def _split_cross_imports_by_test(
    imports: List[str],
) -> Tuple[List[str], List[str]]:
    """Split cross-file import statements into (non_test, test_named) groups.

    Import statements that name pytest-discoverable symbols (``Test*`` or
    ``test_*``) are returned as inline imports so callers can inject them
    into function/class bodies rather than emitting them at module level.
    Mixed imports (some test, some non-test names) are split into two
    separate statements.
    """
    non_test: List[str] = []
    test_named: List[str] = []
    for imp in imports:
        m = re.match(r"^(from\s+\S+\s+import\s+)(.*)", imp)
        if not m:
            non_test.append(imp)
            continue
        prefix = m.group(1)
        names = [n.strip() for n in m.group(2).split(",")]
        t_names = sorted(n for n in names if _is_test_name(n))
        nt_names = sorted(n for n in names if not _is_test_name(n))
        if t_names:
            test_named.append(f"{prefix}{', '.join(t_names)}")
        if nt_names:
            non_test.append(f"{prefix}{', '.join(nt_names)}")
    return non_test, test_named


def _find_main_block_entity(
    entities: List[Entity],
    entity_source_map: Dict[str, str],
) -> Optional[str]:
    """Return the entity name of the ``if __name__ == '__main__':`` block.

    Returns ``None`` when no such block is present.
    """
    for entity in entities:
        if entity.kind != EntityKind.TOP_LEVEL:
            continue
        src = entity_source_map.get(entity.name, "")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq)
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "__main__"
            ):
                return entity.name
    return None


def _find_main_direct_callees(
    main_src: str, function_entity_names: Set[str]
) -> Set[str]:
    """Return function entity names called directly in the ``__main__`` block.

    Only names that appear in *function_entity_names* (i.e. are defined as
    top-level FUNCTION entities in the same file) are returned, so the
    caller can keep those functions sticky to the original file alongside
    the ``__main__`` block.
    """
    try:
        tree = ast.parse(main_src)
    except SyntaxError:
        return set()
    callees: Set[str] = set()
    for node in tree.body:
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"
        ):
            continue
        for subnode in ast.walk(node):
            if (
                isinstance(subnode, ast.Call)
                and isinstance(subnode.func, ast.Name)
                and subnode.func.id in function_entity_names
            ):
                callees.add(subnode.func.id)
    return callees


def _inject_inline_test_imports_original(
    source: str,
    migrated_test_symbols: Dict[str, str],
    abs_pkg: Optional[str],
    original_basename: str,
) -> str:
    """Inject inline imports for migrated test-named symbols into function/class bodies.

    After a split, test-named symbols (``Test*`` / ``test_*``) that were
    migrated to new files are not re-exported at module level (to avoid
    pytest double-discovery).  This function finds every top-level
    function or class in *source* that still references such symbols and
    injects the required ``from … import …`` statement at the top of
    each body, after any docstring.

    *migrated_test_symbols* maps each migrated test name to its target
    file (relative path).  *abs_pkg* and *original_basename* are used to
    build the correct import prefix (absolute for test files, relative
    otherwise).
    """
    if not migrated_test_symbols:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    # Maps 1-based line number → list of import lines to insert before it.
    insertions: Dict[int, List[str]] = {}

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body_names: Set[str] = set()
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Name) and isinstance(subnode.ctx, ast.Load):
                body_names.add(subnode.id)
        needed: Dict[str, List[str]] = {}
        for name in body_names:
            tfile = migrated_test_symbols.get(name)
            if tfile:
                needed.setdefault(tfile, []).append(name)
        if not needed:
            continue
        import_stmts: List[str] = []
        for tfile, names in sorted(needed.items()):
            if abs_pkg is not None:
                mod = _target_module_name(tfile)
                prefix = f"{abs_pkg}.{mod}" if abs_pkg else mod
            else:
                prefix = _relative_import_prefix(original_basename, tfile)
            import_stmts.append(f"from {prefix} import {', '.join(sorted(names))}")
        first_stmt = node.body[0]
        insert_line = first_stmt.lineno
        if (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
            and len(node.body) > 1
        ):
            insert_line = node.body[1].lineno
        body_line = lines[insert_line - 1]
        indent = body_line[: len(body_line) - len(body_line.lstrip())]
        insertions.setdefault(insert_line, [])
        insertions[insert_line] = [f"{indent}{s}\n" for s in import_stmts] + insertions[
            insert_line
        ]

    if not insertions:
        return source
    result: List[str] = []
    for i, line in enumerate(lines, 1):
        if i in insertions:
            result.extend(insertions[i])
        result.append(line)
    return "".join(result)


def _merge_conftest_sources(existing: str, new_content: str) -> str:
    """Merge *new_content* into an existing conftest.py without duplicating anything.

    When multiple file splits each contribute fixtures to the same conftest.py,
    naively appending produces duplicate import statements, duplicate function
    definitions, and E402 errors (imports after function definitions).

    This function avoids all three:
    - Duplicate import statements (same module + same names) are skipped.
    - Function/class definitions whose names already appear in *existing* are skipped.
    - Non-duplicate imports from *new_content* are inserted after the last existing
      import (before any existing function definitions), preventing E402.
    - Non-duplicate definitions are appended at the end.

    Falls back to simple concatenation when either source cannot be parsed.
    """
    try:
        existing_tree = ast.parse(existing)
        new_tree = ast.parse(new_content)
    except SyntaxError:
        return existing.rstrip() + "\n\n\n" + new_content

    existing_lines = existing.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    def _import_key(node: ast.stmt) -> str:
        if isinstance(node, ast.Import):
            return "I:" + ",".join(
                sorted(f"{a.name}:{a.asname or ''}" for a in node.names)
            )
        assert isinstance(node, ast.ImportFrom)
        dots = "." * (node.level or 0)
        mod = node.module or ""
        return (
            "F:"
            + dots
            + mod
            + ":"
            + ",".join(sorted(f"{a.name}:{a.asname or ''}" for a in node.names))
        )

    # What is already in existing?
    existing_import_keys: Set[str] = {
        _import_key(n)
        for n in existing_tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    }
    existing_defined_names: Set[str] = {
        n.name
        for n in existing_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    # Last import line in existing (0-indexed insertion point).
    last_import_lineno: int = 0
    for n in existing_tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            last_import_lineno = max(last_import_lineno, n.end_lineno)

    # Collect new, non-duplicate imports and definitions from new_content.
    imports_to_insert: List[str] = []
    defs_to_append: List[str] = []

    for node in new_tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if _import_key(node) not in existing_import_keys:
                src = "".join(new_lines[node.lineno - 1 : node.end_lineno]).rstrip()
                imports_to_insert.append(src)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name not in existing_defined_names:
                first_line = (
                    node.decorator_list[0].lineno
                    if node.decorator_list
                    else node.lineno
                )
                src = "".join(new_lines[first_line - 1 : node.end_lineno]).rstrip()
                defs_to_append.append(src)

    if not imports_to_insert and not defs_to_append:
        return existing

    result_lines = list(existing_lines)
    if imports_to_insert:
        # Insert new imports directly after the last existing import line.
        insert_at = last_import_lineno  # 0-indexed position after last import
        new_import_lines = [imp + "\n" for imp in imports_to_insert]
        result_lines = (
            result_lines[:insert_at] + new_import_lines + result_lines[insert_at:]
        )

    result = "".join(result_lines).rstrip()
    if defs_to_append:
        result = result + "\n\n\n" + "\n\n\n".join(defs_to_append) + "\n"
    else:
        result = result + "\n"
    return result
