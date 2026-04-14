from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import ast
from ..dep_graph import find_sccs
from .source_utils import _REL_IMPORT_RE
from .import_analysis import _collect_name_loads, _collect_quoted_annotation_names


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


def _module_import_stmt(
    current_target: str,
    source_file: str,
    abs_pkg: Optional[str],
) -> Tuple[str, str]:
    """Return ``(import_statement, local_name)`` for a module-level import.

    Produces ``from . import conversion`` instead of
    ``from .conversion import SAFE_MODE`` so callers can reference
    ``conversion.SAFE_MODE`` for a live lookup rather than a value snapshot.
    This preserves the original single-file behaviour where module globals are
    looked up dynamically rather than captured at import time.
    """
    local_name = _target_module_name(source_file).split(".")[-1]
    if abs_pkg is not None:
        mod = _target_module_name(source_file)
        # Use "import full.module.path as local_name" for absolute contexts.
        # This avoids "from pkg import test_module" patterns that are
        # misidentified as test-name imports by _split_cross_imports_by_test.
        full_mod = f"{abs_pkg}.{mod}" if abs_pkg else mod
        stmt = (
            f"import {full_mod} as {local_name}"
            if full_mod != local_name
            else f"import {local_name}"
        )
    else:
        prefix = _relative_import_prefix(current_target, source_file)
        # prefix looks like ".conversion", "..test_svc", or "..helpers.io".
        # Decompose into leading dots + module path, then extract the last
        # segment as local_name and the rest as the parent package prefix.
        #   ".conversion"  → dots="..",   path="conversion" → "from . import conversion"
        #   "..test_svc"   → dots="..",   path="test_svc"   → "from .. import test_svc"
        #   "..helpers.io" → dots="..",   path="helpers.io" → "from ..helpers import io"
        dot_end = 0
        while dot_end < len(prefix) and prefix[dot_end] == ".":
            dot_end += 1
        dots = prefix[:dot_end]
        path = prefix[dot_end:]
        last_dot = path.rfind(".")
        if last_dot == -1:
            parent = dots or "."
        else:
            parent = dots + path[:last_dot]
        stmt = f"from {parent} import {local_name}"
    return stmt, local_name


def _find_cross_file_imports(
    entity_names: List[str],
    entity_source_map: Dict[str, str],
    name_to_target_file: Dict[str, str],
    current_target: str,
    abs_pkg: Optional[str] = None,
    top_level_var_names: Optional[Set[str]] = None,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Return ``(from_imports, module_imports, name_rewrites)`` for other-file
    dependencies.

    When an entity being moved to *current_target* references a name that is
    defined by another entity being moved to a different target file, the new
    file needs an explicit import for that name.

    *from_imports* are ``from .module import Name`` statements for
    function/class references.  These may be subject to test-name inline
    injection by the caller (to avoid pytest collecting imported test functions
    as duplicate tests).

    *module_imports* are ``from . import module`` (or ``import pkg.module as
    module``) statements for names defined by ``TOP_LEVEL`` entities
    (module-level variables such as ``SAFE_MODE = True``).  These must always
    be placed at module level — never injected inline — because they are
    required by decorator expressions that are evaluated before any function
    body runs.  The returned *name_rewrites* dict maps each such bare name
    (e.g. ``"SAFE_MODE"``) to its qualified form (e.g.
    ``"conversion.SAFE_MODE"``); callers must rewrite the entity source
    accordingly.

    When *abs_pkg* is ``None`` the import prefix is relative (e.g.
    ``from .constants import _CONST``).  When *abs_pkg* is set the import is
    absolute (e.g. ``from tests.constants import _CONST``), which is required
    for test files that pytest loads as top-level modules.
    """
    referenced: Set[str] = set()
    for name in entity_names:
        src = entity_source_map.get(name, "")
        referenced |= _collect_name_loads(src)
    from_files: Dict[str, List[str]] = {}  # source_file → regular names
    mod_files: Dict[str, List[str]] = {}  # source_file → top-level var names
    for ref_name in sorted(referenced):
        source_file = name_to_target_file.get(ref_name)
        if source_file and source_file != current_target:
            if top_level_var_names and ref_name in top_level_var_names:
                mod_files.setdefault(source_file, []).append(ref_name)
            else:
                from_files.setdefault(source_file, []).append(ref_name)

    from_result: List[str] = []
    mod_result: List[str] = []
    rewrites: Dict[str, str] = {}
    for source_file, names in sorted(from_files.items()):
        if abs_pkg is not None:
            mod = _target_module_name(source_file)
            prefix = f"{abs_pkg}.{mod}" if abs_pkg else mod
        else:
            prefix = _relative_import_prefix(current_target, source_file)
        from_result.append(f"from {prefix} import {', '.join(sorted(names))}")
    for source_file, names in sorted(mod_files.items()):
        stmt, local_name = _module_import_stmt(current_target, source_file, abs_pkg)
        mod_result.append(stmt)
        for name in names:
            rewrites[name] = f"{local_name}.{name}"
    return from_result, mod_result, rewrites


def _find_cross_file_type_checking_imports(
    entity_names: List[str],
    entity_source_map: Dict[str, str],
    name_to_target_file: Dict[str, str],
    current_target: str,
    abs_pkg: Optional[str] = None,
    top_level_var_names: Optional[Set[str]] = None,
) -> List[str]:
    """Return cross-file imports for names only referenced in quoted annotations.

    When an entity uses a name only inside a quoted type annotation (e.g.
    ``Optional["_LLMAccumulator"]``) and that name is defined in another new
    file produced by the same split, a ``from .other import Name`` statement
    is generated here.  These should be placed under ``if TYPE_CHECKING:``
    because they are not needed at runtime.

    Names that also appear in regular (non-annotation) loads are excluded —
    they already get a normal cross-file import from
    ``_find_cross_file_imports``.  Top-level variable names (which require
    module-alias imports) are also skipped here.
    """
    runtime_referenced: Set[str] = set()
    quoted_referenced: Set[str] = set()
    for name in entity_names:
        src = entity_source_map.get(name, "")
        runtime_referenced |= _collect_name_loads(src)
        quoted_referenced |= _collect_quoted_annotation_names(src)

    annotation_only = quoted_referenced - runtime_referenced
    if not annotation_only:
        return []

    tc_files: Dict[str, List[str]] = {}
    for ref_name in sorted(annotation_only):
        source_file = name_to_target_file.get(ref_name)
        if source_file and source_file != current_target:
            # Top-level var names need module-alias imports, not handled here.
            if top_level_var_names and ref_name in top_level_var_names:
                continue
            tc_files.setdefault(source_file, []).append(ref_name)

    result: List[str] = []
    for source_file, names in sorted(tc_files.items()):
        if abs_pkg is not None:
            mod = _target_module_name(source_file)
            prefix = f"{abs_pkg}.{mod}" if abs_pkg else mod
        else:
            prefix = _relative_import_prefix(current_target, source_file)
        result.append(f"from {prefix} import {', '.join(sorted(names))}")
    return result


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
    # __init__.py defines the package itself; external callers import from the
    # package path (e.g. "pkg.sub"), not "pkg.sub.__init__".
    if orig.name == "__init__.py":
        dot = target_module.rfind(".")
        if dot == -1:
            return set()  # bare __init__.py at project root; no external callers
        target_module = target_module[:dot]
    result: Set[str] = set()
    for py_file in project_root.rglob("*.py"):
        if py_file.resolve() == orig:
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py_file))
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
