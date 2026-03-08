from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
from ..advisor import GroupPlacement
from ..classifier import ClassifiedEntities
from ..dep_graph import find_sccs
from ..entity_parser import Entity, EntityKind, _parse_section_headers
import ast
import re


@dataclass
class ImportInfo:
    """A top-level import statement and the names it introduces."""

    names: List[str]  # names made available by this import
    source: str  # the import statement text (no trailing newline)
    is_future: bool  # True if `from __future__ import ...`


# Matches any line that is an import statement (plain or from-import).
_IMPORT_LINE_RE = re.compile(r"^(import\s+|from\s+\S.*\s+import\s+)")

# Matches a `from __future__ import …` line (with optional trailing newline).
_FUTURE_IMPORT_LINE_RE = re.compile(r"^from __future__ import .*\n?", re.MULTILINE)

# Matches the leading dots of a relative import (``from .foo`` or ``from ..``).
_REL_IMPORT_RE = re.compile(r"^from (\.+)", re.MULTILINE)

# Matches four or more consecutive newlines (= 3+ blank lines between entities).
_EXCESS_BLANK_RE = re.compile(r"\n{4,}")


def _normalize_blank_lines(source: str) -> str:
    """Collapse runs of 3+ blank lines to 2; ensure exactly one trailing newline.

    Removes blank-line artefacts produced by entity removal (original file)
    and entity-source stripping (new files).  PEP 8 / flake8 E303 allows at
    most two blank lines between top-level definitions.
    """
    source = _EXCESS_BLANK_RE.sub("\n\n\n", source)
    return source.rstrip("\n") + "\n"


def _strip_orphaned_section_headers(source: str) -> str:
    """Remove section header comment blocks with no substantive code after them.

    When entities are removed from the original file, section headers that
    labelled a group of functions may be left with nothing beneath them.
    This function detects both 3-line (``# ---...--- / # Label / # ---...---``)
    and single-line (``# --- Label ---``, ``# === LABEL ===``) patterns and
    removes any whose remaining content (non-blank, non-header lines) has
    been entirely stripped away.
    """
    lines = source.splitlines(keepends=True)
    headers = _parse_section_headers(lines)
    if not headers:
        return source

    # 1-indexed set of lines that belong to any header block.
    header_1idx: Set[int] = set()
    for start, end, _ in headers:
        header_1idx.update(range(start, end + 1))

    # A header is orphaned when no substantive line (non-blank and not part of
    # any header block) falls between it and the *next* header (or EOF).
    orphaned_0idx: Set[int] = set()
    for h_idx, (start_1, end_1, _) in enumerate(headers):
        # Scan only up to the start of the next header so that content beneath
        # a later header does not rescue an earlier, empty one.
        if h_idx + 1 < len(headers):
            scan_end_0 = headers[h_idx + 1][0] - 1  # 0-indexed exclusive
        else:
            scan_end_0 = len(lines)
        has_content = False
        for j0 in range(end_1, scan_end_0):  # 0-indexed, past the header block
            stripped = lines[j0].strip()
            if stripped and (j0 + 1) not in header_1idx:
                has_content = True
                break
        if not has_content:
            for i1 in range(start_1, end_1 + 1):
                orphaned_0idx.add(i1 - 1)  # convert to 0-indexed

    if not orphaned_0idx:
        return source
    return "".join(line for i, line in enumerate(lines) if i not in orphaned_0idx)


def _import_derived_names(source: str) -> Set[str]:
    """Return names introduced solely by import statements in *source*.

    These names live in the original file's namespace via its import
    statements and cannot be re-exported from a new module the way
    assignment-defined names can.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name)
    return names


def _collect_name_loads(source: str) -> Set[str]:
    """Return Name loads in *source* that are not shadowed by function parameters.

    For each function or async function, names that appear as parameters of that
    function are excluded from Name loads within its body.  This prevents generating
    spurious cross-file imports for names that are satisfied locally (e.g. pytest
    fixture names that appear as test function parameters).

    Decorators, argument default values, and return/argument annotations are
    always evaluated in the outer scope and are never excluded.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: Set[str] = set()

    def _walk(node: ast.AST, excluded: frozenset) -> None:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in excluded:
                names.add(node.id)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            own_params: frozenset = frozenset(
                a.arg
                for a in (
                    args.args
                    + args.posonlyargs
                    + args.kwonlyargs
                    + ([args.vararg] if args.vararg else [])
                    + ([args.kwarg] if args.kwarg else [])
                )
            )
            # Decorators are evaluated in the outer scope.
            for dec in node.decorator_list:
                _walk(dec, excluded)
            # Default values are evaluated in the outer scope.
            for default in args.defaults + args.kw_defaults:
                if default is not None:
                    _walk(default, excluded)
            # Annotations are in the outer scope (PEP 563 / regular annotations).
            for arg in args.args + args.posonlyargs + args.kwonlyargs:
                if arg.annotation:
                    _walk(arg.annotation, excluded)
            if args.vararg and args.vararg.annotation:
                _walk(args.vararg.annotation, excluded)
            if args.kwarg and args.kwarg.annotation:
                _walk(args.kwarg.annotation, excluded)
            if node.returns:
                _walk(node.returns, excluded)
            # Function body uses the combined excluded set.
            new_excluded = excluded | own_params
            for child in node.body:
                _walk(child, new_excluded)
            return
        for child in ast.iter_child_nodes(node):
            _walk(child, excluded)

    _walk(tree, frozenset())
    return names


def _extract_import_info(source: str) -> List[ImportInfo]:
    """Return :class:`ImportInfo` for each top-level import in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines(keepends=True)
    result: List[ImportInfo] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [
                alias.asname if alias.asname else alias.name.split(".")[0]
                for alias in node.names
            ]
            src = "".join(lines[node.lineno - 1 : node.end_lineno]).rstrip()
            result.append(ImportInfo(names=names, source=src, is_future=False))
        elif isinstance(node, ast.ImportFrom):
            names = [
                alias.asname if alias.asname else alias.name for alias in node.names
            ]
            # Reconstruct as a normalized single-line import so that
            # multi-line parenthesized imports (e.g. ``from X import (\n
            # Y,\n Z,\n)``) don't break _merge_from_imports, whose regex
            # only matches the first line.
            dots = "." * (node.level or 0)
            mod = node.module or ""
            alias_strs = [
                f"{a.name} as {a.asname}" if a.asname else a.name for a in node.names
            ]
            src = f"from {dots}{mod} import {', '.join(alias_strs)}"
            is_future = node.module == "__future__"
            result.append(ImportInfo(names=names, source=src, is_future=is_future))

    return result


def _find_needed_imports(
    entity_names: List[str],
    entity_source_map: Dict[str, str],
    import_infos: List[ImportInfo],
    all_entity_names: Set[str],
) -> List[str]:
    """Return import statements needed by the given entities.

    Always includes ``from __future__`` imports.  Other imports are included
    when any of the names they introduce appear in the entities' source.
    Duplicate import source strings are deduplicated.
    """
    referenced: Set[str] = set()
    for name in entity_names:
        src = entity_source_map.get(name, "")
        referenced |= _collect_name_loads(src)

    needed: List[str] = []
    seen: Set[str] = set()
    for info in import_infos:
        if info.source in seen:
            continue
        if info.is_future or any(n in referenced for n in info.names):
            needed.append(info.source)
            seen.add(info.source)

    return needed


def _bump_relative_imports(source: str, n: int = 1) -> str:
    """Increment the level of every relative import in *source* by *n*.

    Used when file content is moved directory levels deeper, e.g. when the
    source originally written for ``pkg/module.py`` becomes the content of
    ``pkg/module/__init__.py``, or when new files go into a subdirectory
    package instead of sitting next to the original file.

    With n=1: ``from .foo`` → ``from ..foo``, ``from ..bar`` → ``from ...bar``.
    With n=2: ``from .foo`` → ``from ...foo``, etc.
    Absolute imports are not affected.
    """
    for _ in range(n):
        source = _REL_IMPORT_RE.sub(lambda m: f"from .{m.group(1)}", source)
    return source


def _relative_import_prefix(from_file: str, to_file: str) -> str:
    """Return the Python relative-import prefix for *to_file* as seen from *from_file*.

    Both paths are relative to the same base directory (the original file's
    directory).  Examples::

        _relative_import_prefix("utils.py", "helpers.py")          → ".helpers"
        _relative_import_prefix("sub/a.py", "helpers/b.py")        → "..helpers.b"
        _relative_import_prefix("sub/a.py", "sub/b.py")            → ".b"
        _relative_import_prefix("a.py", "__init__.py")             → "."
        _relative_import_prefix("sub/a.py", "sub/__init__.py")     → "."
    """
    to_path = Path(to_file)
    from_parts = Path(from_file).parent.parts  # () for top-level files
    # __init__.py represents the package itself, not a submodule named "__init__".
    if to_path.stem == "__init__":
        to_module_parts = to_path.parent.parts
    else:
        to_module_parts = to_path.with_suffix("").parts  # ("helpers", "b")
    to_dir_parts = to_path.parent.parts  # ("helpers",)

    # Length of the common directory prefix between from_dir and to_dir.
    common_len = 0
    for fp, tp in zip(from_parts, to_dir_parts):
        if fp == tp:
            common_len += 1
        else:
            break

    levels_up = len(from_parts) - common_len
    module = ".".join(to_module_parts[common_len:])
    return "." * (levels_up + 1) + module


def _target_module_name(target_file: str) -> str:
    """Convert a relative target filename to a dotted module name.

    ``"utils.py"`` → ``"utils"``, ``"helpers/io.py"`` → ``"helpers.io"``,
    ``"pkg/__init__.py"`` → ``"pkg"`` (package, not ``"pkg.__init__"``).
    """
    path = Path(target_file)
    if path.stem == "__init__":
        parts = list(path.parent.parts)
    else:
        parts = list(path.with_suffix("").parts)
    return ".".join(parts)


def _find_cross_file_imports(
    entity_names: List[str],
    entity_source_map: Dict[str, str],
    name_to_target_file: Dict[str, str],
    current_target: str,
    abs_pkg: Optional[str] = None,
) -> List[str]:
    """Return ``from … import name`` statements for other-file dependencies.

    When an entity being moved to *current_target* references a name that is
    defined by another entity being moved to a different target file, the new
    file needs an explicit import for that name.

    When *abs_pkg* is ``None`` the import prefix is relative (e.g.
    ``from .constants import _CONST``).  When *abs_pkg* is set the import is
    absolute (e.g. ``from tests.constants import _CONST``), which is required
    for test files that pytest loads as top-level modules.
    """
    referenced: Set[str] = set()
    for name in entity_names:
        src = entity_source_map.get(name, "")
        referenced |= _collect_name_loads(src)
    from_files: Dict[str, List[str]] = {}  # source_file → names
    for ref_name in sorted(referenced):
        source_file = name_to_target_file.get(ref_name)
        if source_file and source_file != current_target:
            from_files.setdefault(source_file, []).append(ref_name)

    result = []
    for source_file, names in sorted(from_files.items()):
        if abs_pkg is not None:
            mod = _target_module_name(source_file)
            prefix = f"{abs_pkg}.{mod}" if abs_pkg else mod
        else:
            prefix = _relative_import_prefix(current_target, source_file)
        result.append(f"from {prefix} import {', '.join(sorted(names))}")
    return result


_FROM_IMPORT_RE = re.compile(r"^(from\s+\S+)\s+import\s+(.*)")


def _merge_from_imports(imports: List[str]) -> List[str]:
    """Merge ``from X import …`` lines that share the same module prefix.

    When multiple entities each contribute a ``from X import`` for the same
    module but with different name subsets, the naive per-entity approach
    produces duplicate imports such as::

        from .conversion import lua_to_python, python_to_lua
        from .conversion import lua_to_python_preserve_wrapped, python_to_lua

    This function collapses them into a single statement per prefix, with
    names sorted and deduplicated::

        from .conversion import lua_to_python, lua_to_python_preserve_wrapped, python_to_lua  # noqa: E501

    Plain ``import X`` statements are preserved unchanged and appended after
    the merged from-imports.
    """
    from_map: Dict[str, List[str]] = {}
    order: List[str] = []  # first-seen order of prefixes
    plain: List[str] = []
    for imp in imports:
        m = _FROM_IMPORT_RE.match(imp)
        if not m:
            plain.append(imp)
            continue
        prefix = m.group(1)
        names = [n.strip() for n in m.group(2).split(",") if n.strip()]
        if prefix not in from_map:
            from_map[prefix] = []
            order.append(prefix)
        from_map[prefix].extend(names)
    result = []
    for prefix in order:
        unique = sorted(dict.fromkeys(from_map[prefix]))
        result.append(f"{prefix} import {', '.join(unique)}")
    return result + plain


def _import_line_numbers(entity: Entity, entity_src: str) -> Set[int]:
    """Return absolute 1-based line numbers of import statements in *entity*.

    Used to preserve import lines in the original file when a TOP_LEVEL
    entity that mixes imports and assignments is migrated.
    """
    try:
        tree = ast.parse(entity_src)
    except SyntaxError:
        return set()
    result: Set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for rel_ln in range(node.lineno, node.end_lineno + 1):
                result.add(entity.start_line + rel_ln - 1)
    return result


def _remove_entity_lines(
    source: str,
    migrated_names: Set[str],
    entity_map: Dict[str, Entity],
    entity_source_map: Dict[str, str],
) -> str:
    """Return *source* with lines belonging to migrated entities removed.

    For TOP_LEVEL entities, import statement lines are preserved in the
    original file even when the entity is migrated: the remaining code may
    still reference those imported names, and stdlib/third-party names
    cannot be safely re-exported from a new module.
    """
    remove: Set[int] = set()
    preserve: Set[int] = set()
    for name in migrated_names:
        entity = entity_map.get(name)
        if entity is None:
            continue
        for ln in range(entity.start_line, entity.end_line + 1):
            remove.add(ln)
        if entity.kind == EntityKind.TOP_LEVEL:
            preserve |= _import_line_numbers(entity, entity_source_map.get(name, ""))

    lines = source.splitlines(keepends=True)
    return "".join(
        line for i, line in enumerate(lines, 1) if i not in remove or i in preserve
    )


def _find_project_root(path: Path) -> Optional[Path]:
    """Walk up from *path* to find the project root directory.

    Returns the first directory containing ``pyproject.toml``, ``setup.py``,
    ``setup.cfg``, or ``.git``.  Returns ``None`` when the filesystem root is
    reached without finding any of these markers.
    """
    markers = {"pyproject.toml", "setup.py", "setup.cfg", ".git"}
    current = path if path.is_dir() else path.parent
    while True:
        if any((current / m).exists() for m in markers):
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _module_path_from_file(project_root: Path, file_path: Path) -> Optional[str]:
    """Return the dotted Python module path of *file_path* relative to *project_root*.

    Returns ``None`` when *file_path* is not under *project_root*.
    """
    try:
        rel = file_path.relative_to(project_root)
    except ValueError:
        return None
    return ".".join(rel.with_suffix("").parts)


def _abs_package_for_dir(file_path: str) -> Optional[str]:
    """Return the dotted package path of the directory containing *file_path*.

    Used to generate absolute imports for test files so that pytest's default
    import mode (which loads test files as top-level modules, not package
    members) does not choke on ``from .module import …`` syntax.

    Returns an empty string for files sitting directly in the project root,
    ``None`` when the project root cannot be determined.
    """
    orig = Path(file_path).resolve()
    project_root = _find_project_root(orig.parent)
    if project_root is None:
        return None
    try:
        rel = orig.parent.relative_to(project_root)
    except ValueError:
        return None
    return ".".join(rel.parts)


def _collect_external_imported_names(original_path: str) -> Set[str]:
    """Return names imported from *original_path* by other Python files.

    Scans all Python files under the project root for ``from <module> import``
    statements targeting the module corresponding to *original_path*, and
    returns the union of all imported original names (before any ``as`` alias).

    Returns an empty set when *original_path* does not resolve to an existing
    file, the project root cannot be determined, or the path cannot be mapped
    to a module.  Both absolute and relative paths are accepted; relative paths
    are resolved against the current working directory (the repo root when
    crispen is invoked as ``git diff | crispen``).
    """
    orig = Path(original_path).resolve()
    if not orig.exists():
        return set()
    project_root = _find_project_root(orig.parent)
    if project_root is None:
        return set()
    # project_root is an ancestor of orig (derived by walking up from orig.parent),
    # so _module_path_from_file always returns a non-None string here.
    target_module = _module_path_from_file(project_root, orig)
    result: Set[str] = set()
    for py_file in project_root.rglob("*.py"):
        if py_file.resolve() == orig:
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except Exception:
            continue
        # Compute this file's dotted module path for relative-import resolution.
        file_module = _module_path_from_file(project_root, py_file)
        file_pkg_parts = file_module.split(".")[:-1] if file_module else []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 0:
                imported_from = node.module or ""
            else:
                # Relative import: go up (level - 1) packages from file_pkg_parts.
                up = node.level - 1
                if up > len(file_pkg_parts):
                    continue
                base = file_pkg_parts[: len(file_pkg_parts) - up]
                sub = node.module or ""
                imported_from = ".".join(base + ([sub] if sub else []))
            if imported_from != target_module:
                continue
            for alias in node.names:
                result.add(alias.name)
    return result


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


def _prune_inline_redundant_imports(source: str) -> str:
    """Remove function-body imports that duplicate module-level imports.

    When a function-local ``from x import y`` re-imports a name that is
    already provided by a top-level import, flake8 reports an F811
    redefinition warning.  This function removes such redundant inner imports
    (or narrows them when only some names are redundant).

    Returns *source* unchanged when it cannot be parsed or nothing needs
    pruning.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    # Names already available from top-level (module-level) imports.
    top_level_names: Set[str] = set()
    top_level_node_ids: Set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_node_ids.add(id(node))
            for alias in node.names:
                top_level_names.add(
                    alias.asname if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            top_level_node_ids.add(id(node))
            for alias in node.names:
                top_level_names.add(alias.asname if alias.asname else alias.name)

    if not top_level_names:
        return source

    # Find all import nodes that are NOT at module level.
    inner_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and id(node) not in top_level_node_ids
    ]

    if not inner_imports:
        return source

    lines = source.splitlines(keepends=True)
    # Maps 1-based line number → replacement line (None = remove that line).
    line_ops: Dict[int, Optional[str]] = {}

    for stmt in inner_imports:
        if isinstance(stmt, ast.Import):
            kept = [
                a
                for a in stmt.names
                if (a.asname if a.asname else a.name.split(".")[0])
                not in top_level_names
            ]
        else:
            kept = [
                a
                for a in stmt.names
                if (a.asname if a.asname else a.name) not in top_level_names
            ]

        if len(kept) == len(stmt.names):
            continue  # no redundancy — nothing to remove

        # Mark every line of this import for removal.
        for ln in range(stmt.lineno, stmt.end_lineno + 1):
            line_ops[ln] = None

        if kept:
            # Rebuild a narrowed import preserving original indentation.
            alias_strs = [
                f"{a.name} as {a.asname}" if a.asname else a.name for a in kept
            ]
            orig_line = lines[stmt.lineno - 1]
            indent = orig_line[: len(orig_line) - len(orig_line.lstrip())]
            if isinstance(stmt, ast.ImportFrom):
                dots = "." * (stmt.level or 0)
                mod = stmt.module or ""
                new_line = f"{indent}from {dots}{mod} import {', '.join(alias_strs)}\n"
            else:
                new_line = f"{indent}import {', '.join(alias_strs)}\n"
            line_ops[stmt.lineno] = new_line

    if not line_ops:
        return source

    result: List[str] = []
    for i, line in enumerate(lines, 1):
        if i in line_ops:
            repl = line_ops[i]
            if repl is not None:
                result.append(repl)
            # else: None → line is removed
        else:
            result.append(line)
    return "".join(result)


def _prune_unused_imports(source: str) -> str:
    """Remove or narrow unused imports in a generated file.

    ``from __future__`` and star imports are always preserved.  Multi-name
    imports are narrowed to only the names actually referenced in *source*
    rather than dropped wholesale.  Fully-unused imports are removed entirely.

    Returns *source* unchanged when it cannot be parsed or nothing needs
    pruning.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    used = _collect_name_loads(source)
    lines = source.splitlines(keepends=True)
    # Maps 1-based line number → replacement line (None = remove that line).
    replacements: Dict[int, Optional[str]] = {}

    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue

        # Always preserve __future__ imports.
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue

        # Always preserve star imports.
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            continue

        kept = [
            a
            for a in node.names
            if (a.asname if a.asname else a.name.split(".")[0]) in used
        ]

        if len(kept) == len(node.names):
            continue  # nothing to prune

        # Mark every line of this import for removal.
        for ln in range(node.lineno, node.end_lineno + 1):
            replacements[ln] = None

        if not kept:
            continue  # fully unused — all lines already removed

        # Rebuild a single-line import with only the kept aliases.
        alias_strs = [f"{a.name} as {a.asname}" if a.asname else a.name for a in kept]
        if isinstance(node, ast.ImportFrom):
            level_dots = "." * (node.level or 0)
            module = node.module or ""
            new_line = f"from {level_dots}{module} import {', '.join(alias_strs)}\n"
        else:
            new_line = f"import {', '.join(alias_strs)}\n"
        replacements[node.lineno] = new_line

    if not replacements:
        return source

    result: List[str] = []
    for i, line in enumerate(lines, 1):
        if i not in replacements:
            result.append(line)
        elif replacements[i] is not None:
            result.append(replacements[i])
        # else: line is removed — skip it
    return "".join(result)


def _strip_top_level_import_lines(src: str) -> str:
    """Return *src* with all top-level import statements removed.

    Uses AST to locate the exact line range of each import node, correctly
    handling multi-line imports.  Returns *src* unchanged when it cannot be
    parsed as Python.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    remove: Set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for ln in range(node.lineno, node.end_lineno + 1):
                remove.add(ln)
    if not remove:
        return src
    lines = src.splitlines(keepends=True)
    return "".join(line for i, line in enumerate(lines, 1) if i not in remove)


def _extract_module_docstring(source: str) -> Optional[str]:
    """Return the module-level docstring source text, or None if absent."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if not (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        return None
    node = tree.body[0]
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno]).rstrip()


def _strip_module_docstring(src: str) -> str:
    """Return *src* with the leading module-level docstring removed."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    if not (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        return src
    node = tree.body[0]
    remove = set(range(node.lineno, node.end_lineno + 1))
    lines = src.splitlines(keepends=True)
    return "".join(line for i, line in enumerate(lines, 1) if i not in remove)
