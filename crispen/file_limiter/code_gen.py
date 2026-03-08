"""Code generation for FileLimiter: build new files and update original source."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .advisor import FileLimiterPlan, GroupPlacement
from .classifier import ClassifiedEntities
from .dep_graph import find_sccs
from .entity_parser import Entity, EntityKind


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class ImportInfo:
    """A top-level import statement and the names it introduces."""

    names: List[str]  # names made available by this import
    source: str  # the import statement text (no trailing newline)
    is_future: bool  # True if `from __future__ import ...`


@dataclass
class SplitResult:
    """Output of :func:`generate_file_splits`."""

    new_files: Dict[str, str]  # {target_file: source_code}
    original_source: str  # updated original file source
    abort: bool  # True if generation failed / nothing to split
    abort_reason: str = ""  # human-readable explanation when abort=True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# __main__ handling
# ---------------------------------------------------------------------------


def _is_test_name(name: str) -> bool:
    """Return True if *name* matches pytest's test-discovery patterns.

    Pytest collects classes named ``Test*`` and functions named ``test_*``.
    Importing such names at module level in a test file causes every test
    inside to be discovered — and run — a second time.
    """
    return name.startswith("Test") or name.startswith("test_")


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


def _inject_inline_imports(entity_src: str, imports: List[str]) -> str:
    """Inject *imports* at the top of a function or class body in *entity_src*.

    The imports are inserted after any leading docstring.  Returns
    *entity_src* unchanged when it cannot be parsed or the top-level node
    is not a function or class (TOP_LEVEL entities have no body scope).
    """
    if not imports:
        return entity_src
    try:
        tree = ast.parse(entity_src)
    except SyntaxError:
        return entity_src
    if not tree.body:
        return entity_src
    top = tree.body[0]
    if not isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return entity_src
    first_stmt = top.body[0]
    insert_line = first_stmt.lineno
    if (
        isinstance(first_stmt, ast.Expr)
        and isinstance(first_stmt.value, ast.Constant)
        and isinstance(first_stmt.value.value, str)
        and len(top.body) > 1
    ):
        insert_line = top.body[1].lineno
    lines = entity_src.splitlines(keepends=True)
    body_line = lines[insert_line - 1]
    indent = body_line[: len(body_line) - len(body_line.lstrip())]
    import_lines = [f"{indent}{imp}\n" for imp in imports]
    return "".join(lines[: insert_line - 1] + import_lines + lines[insert_line - 1 :])


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


# ---------------------------------------------------------------------------
# Conftest merging
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_file_splits(
    classified: ClassifiedEntities,
    plan: FileLimiterPlan,
    post_source: str,
    original_path: str,
    subdir_name: Optional[str] = None,
    pytest_conftest: bool = False,
    has_main: bool = False,
) -> SplitResult:
    """Generate new file contents and the updated original source.

    When *plan* is aborted or has no placements, returns :class:`SplitResult`
    with the original source unchanged (``abort`` mirrors ``plan.abort``).

    When *subdir_name* is set (e.g. ``"service"``), the file is being split
    into a package subdirectory.  The "original" file is treated as
    ``service/__init__.py`` for dependency-graph and import-prefix purposes,
    so cross-file imports within the package use correct relative paths.

    When *pytest_conftest* is True, any entity decorated with
    ``@pytest.fixture`` (or ``@fixture``) is redirected to ``conftest.py``
    instead of the LLM-assigned target file.  pytest auto-discovers fixtures
    from ``conftest.py``, so no re-export import is added to the original
    file — eliminating both the F401 and F811 flake8 warnings that arise
    when a fixture name is used as a test function parameter.
    """
    if plan.abort:
        return SplitResult(
            new_files={},
            original_source=post_source,
            abort=True,
            abort_reason=plan.abort_reason,
        )

    if not plan.placements:
        return SplitResult(new_files={}, original_source=post_source, abort=False)

    # Detect shebang on line 1 so it can be stripped from new files and
    # preserved (or restored) at the top of the original.
    shebang: Optional[str] = None
    if post_source.startswith("#!"):
        nl = post_source.find("\n")
        shebang = post_source[: nl + 1] if nl != -1 else post_source + "\n"

    lines = post_source.splitlines(keepends=True)
    entity_map = {e.name: e for e in classified.entities}

    # Build entity source map (name → stripped source string).
    entity_source_map: Dict[str, str] = {}
    for entity in classified.entities:
        entity_source_map[entity.name] = "".join(
            lines[entity.start_line - 1 : entity.end_line]
        ).rstrip()

    # All entity-defined names (used to limit import matching scope).
    all_entity_names: Set[str] = {
        name for e in classified.entities for name in e.names_defined
    }

    # Extract import info from post-refactor source.
    import_infos = _extract_import_info(post_source)

    # Placements whose target_file matches the original filename would create a
    # self-referential import (e.g. `from .duplicate_extractor import Foo` inside
    # duplicate_extractor.py).  Drop them — entities stay in the original file.
    # In subdir-split mode the "original" is the package __init__.py; use that
    # name throughout so dependency-graph edges and import prefixes are correct.
    original_basename = (
        f"{subdir_name}/__init__.py"
        if subdir_name and not has_main
        else Path(original_path).name
    )
    is_test_file = Path(original_path).name.startswith("test_")
    # For test-file subdir splits the original test file stays on disk (runner.py
    # does not redirect it to __init__.py), so non-migrated names still live in
    # the original file (e.g. "test_runner.py"), not in the package __init__.py.
    non_migrated_home = (
        Path(original_path).name
        if (subdir_name and is_test_file)
        else original_basename
    )
    # Identify the __main__ block and any functions it calls directly.
    # These stay in the original file unconditionally: the __main__ block
    # is an entry point the user expects to keep working, and its direct
    # callees must live in the same file to avoid module-level test-class
    # imports that would cause pytest double-discovery.
    main_entity = _find_main_block_entity(classified.entities, entity_source_map)
    main_sticky: Set[str] = set()
    if main_entity is not None:
        main_sticky.add(main_entity)
        function_entity_names = {
            e.name for e in classified.entities if e.kind == EntityKind.FUNCTION
        }
        main_sticky.update(
            _find_main_direct_callees(
                entity_source_map.get(main_entity, ""), function_entity_names
            )
        )

    valid_placements = [
        p
        for p in plan.placements
        if p.target_file != original_basename
        and not any(name in main_sticky for name in p.group)
    ]

    # --- Pytest conftest routing ---
    # When enabled, entities decorated with @pytest.fixture are redirected to
    # conftest.py instead of the LLM-assigned target file.  pytest discovers
    # fixtures from conftest.py automatically, so no re-export import is added
    # to the original file — eliminating the F401/F811 flake8 false positives.
    fixture_entity_names: Set[str] = set()
    if pytest_conftest:
        conftest_group: List[str] = []
        new_valid: List[GroupPlacement] = []
        for p in valid_placements:
            non_fixture: List[str] = []
            for name in p.group:
                src = entity_source_map.get(name, "")
                if src and _is_pytest_fixture(src):
                    fixture_entity_names.add(name)
                    conftest_group.append(name)
                else:
                    non_fixture.append(name)
            if non_fixture:
                new_valid.append(
                    GroupPlacement(group=non_fixture, target_file=p.target_file)
                )
        if conftest_group:
            new_valid.append(
                GroupPlacement(group=conftest_group, target_file="conftest.py")
            )
        valid_placements = new_valid

    # Group placements by target file (preserving order for topo sort).
    file_entity_names: Dict[str, List[str]] = {}
    for placement in valid_placements:
        file_entity_names.setdefault(placement.target_file, []).extend(placement.group)

    # All migrated entity names.
    migrated_names: Set[str] = {name for p in valid_placements for name in p.group}

    # Build name → target-file map for cross-file import detection.
    # Exclude import-derived names (_find_needed_imports handles those).
    import_defined_names = {name for info in import_infos for name in info.names}
    name_to_target_file: Dict[str, str] = {}
    for target_file, ent_names in file_entity_names.items():
        for ent_name in ent_names:
            entity = entity_map.get(ent_name)
            if entity:
                for defined_name in entity.names_defined:
                    if defined_name not in import_defined_names:
                        name_to_target_file[defined_name] = target_file

    # Also map names from non-migrated entities to the original file so that
    # split files can import helpers (e.g. _run) that stayed behind.
    for entity in classified.entities:
        if entity.name not in migrated_names:
            for defined_name in entity.names_defined:
                if defined_name not in import_defined_names:
                    name_to_target_file.setdefault(defined_name, non_migrated_home)

    # Extract non-migrated FUNCTION/CLASS entities referenced by migrated ones
    # into the new files that use them, breaking O→F→O import cycles.
    synthetic_placements = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        non_migrated_home,
    )

    # Collect names that external files (outside the module being split) import
    # from the original file.  Private symbols in this set must get a re-export
    # proxy even though they are no longer referenced by the remaining source.
    external_loads = _collect_external_imported_names(original_path)

    # Detect circular imports.  Cycles can pass through the original file:
    # a new file that imports a non-migrated name from the original can form a
    # chain back to the original via the re-exports the original adds.
    # Model the original as an explicit node in the dependency graph.
    #
    # Original's outgoing edges: it will re-export a migrated name when the
    # name is public (no _/test_ prefix), referenced by a non-migrated entity,
    # or imported by an external file (external_loads).
    #
    # In a test-file subdir split, non_migrated_home ("test_runner.py") differs
    # from original_basename ("runner/__init__.py").  Re-exports are injected
    # into the original test file, so it—not the __init__.py—gains outgoing
    # import edges and must be a separate node in the dependency graph.
    non_migrated_loads: Set[str] = set()
    for ent_name, src in entity_source_map.items():
        if ent_name not in migrated_names:
            non_migrated_loads |= _collect_name_loads(src)

    reexport_home = non_migrated_home
    all_dep_nodes = set(file_entity_names.keys()) | {original_basename, reexport_home}
    file_deps: Dict[str, Set[str]] = {node: set() for node in all_dep_nodes}
    for target_file, ent_names in file_entity_names.items():
        for ent_name in ent_names:
            src = entity_source_map.get(ent_name, "")
            for ref_name in _collect_name_loads(src):
                dep_file = name_to_target_file.get(ref_name)
                if dep_file and dep_file != target_file and dep_file in file_deps:
                    file_deps[target_file].add(dep_file)
    for placement in valid_placements + synthetic_placements:
        for ent_name in placement.group:
            entity = entity_map.get(ent_name)
            if entity:
                for defined_name in entity.names_defined:
                    if (
                        (
                            not defined_name.startswith("_")
                            and not defined_name.startswith("test_")
                        )
                        or defined_name in non_migrated_loads
                        or defined_name in external_loads
                    ):
                        file_deps[reexport_home].add(placement.target_file)
                        break
    if any(len(scc) > 1 for scc in find_sccs(file_deps)):
        return SplitResult(
            new_files={},
            original_source=post_source,
            abort=True,
            abort_reason="proposed split would create circular file imports",
        )

    # Use absolute imports when the original file is a test file.  Pytest's
    # default import mode loads test files as top-level modules (not package
    # members), so relative imports like `from .helpers import foo` would
    # raise ImportError at collection time.
    abs_pkg: Optional[str] = None
    if Path(original_path).name.startswith("test_"):
        abs_pkg = _abs_package_for_dir(original_path)

    # In subdir-split mode, new files live inside a package subdirectory and
    # can use relative imports for cross-file references within that package.
    abs_pkg_for_new_files: Optional[str] = None if subdir_name else abs_pkg

    # Generate new file contents.
    new_files: Dict[str, str] = {}
    for target_file, ent_names in file_entity_names.items():
        needed = _find_needed_imports(
            ent_names, entity_source_map, import_infos, all_entity_names
        )
        if subdir_name is not None:
            depth = len(Path(target_file).parts) - 1
            needed = [_bump_relative_imports(s, depth) for s in needed]
        entity_srcs = []
        top_cross: List[str] = []
        seen_top_cross: Set[str] = set()
        for _ent_name in ent_names:
            _src = entity_source_map.get(_ent_name)
            if _src is None:
                continue
            _entity = entity_map.get(_ent_name)
            if _entity and _entity.kind == EntityKind.TOP_LEVEL:
                # Imports are emitted separately by _find_needed_imports; strip
                # them from the entity body to prevent duplicate import stmts.
                _src = _strip_top_level_import_lines(_src)
                if subdir_name is not None:
                    # In subdir-split mode the module docstring belongs in
                    # __init__.py rather than in one of the child modules.
                    _src = _strip_module_docstring(_src)
            else:
                _src = _FUTURE_IMPORT_LINE_RE.sub("", _src)
            # Strip shebang from any entity that begins on line 1 of the
            # original source — it must not appear in generated new files.
            if shebang and _entity and _entity.start_line == 1:
                nl = _src.find("\n")
                _src = _src[nl + 1 :] if nl != -1 else ""
            _src = _src.rstrip()
            # Compute cross-file imports for this entity and split off any
            # test-named symbols (Test* / test_*) to be injected inline.
            entity_cross = _find_cross_file_imports(
                [_ent_name],
                entity_source_map,
                name_to_target_file,
                target_file,
                abs_pkg=abs_pkg_for_new_files,
            )
            ent_top_cross, ent_test_imports = _split_cross_imports_by_test(entity_cross)
            for imp in ent_top_cross:
                if imp not in seen_top_cross:
                    seen_top_cross.add(imp)
                    top_cross.append(imp)
            if ent_test_imports and _entity and _entity.kind != EntityKind.TOP_LEVEL:
                _src = _inject_inline_imports(_src, ent_test_imports)
            else:
                # TOP_LEVEL entity: no body scope, fall back to module level.
                for imp in ent_test_imports:
                    if imp not in seen_top_cross:
                        seen_top_cross.add(imp)
                        top_cross.append(imp)
            entity_srcs.append(_src)
        entity_srcs = [s for s in entity_srcs if s]
        parts: List[str] = []
        all_imports = _merge_from_imports(needed + top_cross)
        if all_imports:
            parts.append("\n".join(all_imports))
        parts.extend(entity_srcs)
        pruned = _prune_unused_imports("\n\n\n".join(parts) + "\n")
        new_files[target_file] = _prune_inline_redundant_imports(pruned)

    # If an existing conftest.py is present on disk, merge intelligently so
    # that duplicate imports and fixture definitions are not repeated (which
    # would cause flake8 F811/E402 errors when multiple splits write to the
    # same conftest.py file).
    if "conftest.py" in new_files:
        existing_conftest = Path(original_path).parent / "conftest.py"
        if existing_conftest.exists():
            prior = existing_conftest.read_text(encoding="utf-8")
            new_files["conftest.py"] = _merge_conftest_sources(
                prior, new_files["conftest.py"]
            )

    # Build updated original source.
    updated = _remove_entity_lines(
        post_source, migrated_names, entity_map, entity_source_map
    )
    updated = _prune_unused_imports(updated)
    # For non-test subdir splits, re-exports from the __init__.py use relative
    # import prefixes computed from inside the package (e.g. ".utils" not
    # ".service.utils").  For test files the original keeps existing abs_pkg
    # behaviour so pytest can find the re-exported symbols.
    # In a non-test subdir split the updated source becomes subdir/__init__.py,
    # which sits one directory level deeper than the original file.  Any
    # relative imports it still contains (e.g. ``from .. import llm_client``
    # or ``from .base import Foo``) therefore need one extra dot so they keep
    # pointing at the same modules.  Re-exports added by _add_re_exports below
    # are already computed from the __init__.py's perspective and are correct.
    if subdir_name is not None and not is_test_file and not has_main:
        updated = _bump_relative_imports(updated)
    if subdir_name is not None:
        # If the original file had a module docstring and it was migrated away,
        # place it in subdir/__init__.py in both cases: for non-test splits
        # the docstring is prepended to `updated` which runner.py redirects to
        # __init__.py; for test splits it is written directly to __init__.py
        # (runner.py does not redirect `updated` for test files).
        _module_doc = _extract_module_docstring(post_source)
        if _module_doc and not _extract_module_docstring(updated):
            if is_test_file:
                new_files[f"{subdir_name}/__init__.py"] = _module_doc + "\n"
            else:
                updated = _module_doc + "\n\n" + updated
    relative_from: Optional[str] = (
        f"{subdir_name}/__init__.py"
        if (subdir_name and not is_test_file and not has_main)
        else None
    )
    # Exclude conftest.py from re-exports: fixtures there are auto-discovered
    # by pytest and must not be imported back into the original test file.
    re_export_placements = [
        p
        for p in valid_placements + synthetic_placements
        if p.target_file != "conftest.py"
    ]
    updated = _add_re_exports(
        updated,
        re_export_placements,
        entity_map,
        entity_source_map,
        external_loads=external_loads,
        abs_pkg=abs_pkg,
        relative_from=relative_from,
    )

    # For non-migrated entities that reference test-named symbols now living
    # in new files: _add_re_exports intentionally skips re-exporting them
    # (to avoid double-discovery), so inject those imports inline instead.
    migrated_test_symbols = {
        name: tfile
        for name, tfile in name_to_target_file.items()
        if tfile != original_basename and _is_test_name(name)
    }
    updated = _inject_inline_test_imports_original(
        updated, migrated_test_symbols, abs_pkg, original_basename
    )

    # Restore shebang at line 1 of the original.  It may have been removed
    # by _remove_entity_lines if the entity owning line 1 was migrated.
    if shebang and not updated.startswith("#!"):
        updated = shebang + updated

    # Normalize blank lines: collapse 3+ consecutive blank lines to 2 and
    # ensure exactly one trailing newline in every generated file.
    new_files = {f: _normalize_blank_lines(s) for f, s in new_files.items()}
    updated = _normalize_blank_lines(updated)

    return SplitResult(new_files=new_files, original_source=updated, abort=False)
