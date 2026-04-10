from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import ast
import re
from ..entity_parser import Entity, EntityKind
from .analysis import _collect_name_loads, _collect_quoted_annotation_names
from .source_utils import _REL_IMPORT_RE


@dataclass
class ImportInfo:
    """A top-level import statement and the names it introduces."""

    names: List[str]  # names made available by this import
    source: str  # the import statement text (no trailing newline)
    is_future: bool  # True if `from __future__ import ...`
    is_type_checking: bool = False  # True if inside `if TYPE_CHECKING:` block


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


def _inject_module_level_imports(source: str, imports: List[str]) -> str:
    """Insert *imports* after the last existing import line in *source*.

    Uses the same insertion logic as :func:`_add_re_exports` so that module
    imports for reassigned TOP_LEVEL variables land in the same position as
    other imports added to the original file.
    """
    if not imports:
        return source
    lines = source.splitlines(keepends=True)
    last_import_line = 0
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "\n".join(sorted(imports)) + "\n\n" + source
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
    import_lines = [imp + "\n" for imp in sorted(imports)]
    return "".join(lines[:insert_after] + import_lines + lines[insert_after:])


def _extract_import_info(source: str) -> List[ImportInfo]:
    """Return :class:`ImportInfo` for each top-level import in *source*.

    Also includes imports found inside module-level ``if TYPE_CHECKING:``
    blocks, marked with ``is_type_checking=True``.  These are used by
    :func:`_find_type_checking_needed_imports` to distribute forward-reference
    imports to the correct sub-files after a split.
    """
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
        elif (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for child in node.body:
                if isinstance(child, ast.Import):
                    tc_names = [
                        alias.asname if alias.asname else alias.name.split(".")[0]
                        for alias in child.names
                    ]
                    tc_src = "".join(
                        lines[child.lineno - 1 : child.end_lineno]
                    ).rstrip()
                    result.append(
                        ImportInfo(
                            names=tc_names,
                            source=tc_src,
                            is_future=False,
                            is_type_checking=True,
                        )
                    )
                elif isinstance(child, ast.ImportFrom):
                    tc_names = [
                        alias.asname if alias.asname else alias.name
                        for alias in child.names
                    ]
                    tc_dots = "." * (child.level or 0)
                    tc_mod = child.module or ""
                    tc_alias_strs = [
                        f"{a.name} as {a.asname}" if a.asname else a.name
                        for a in child.names
                    ]
                    tc_src = f"from {tc_dots}{tc_mod} import {', '.join(tc_alias_strs)}"
                    result.append(
                        ImportInfo(
                            names=tc_names,
                            source=tc_src,
                            is_future=False,
                            is_type_checking=True,
                        )
                    )

    return result


def _inject_type_checking_imports(source: str, imports: List[str]) -> str:
    """Add *imports* under a module-level ``if TYPE_CHECKING:`` guard in *source*.

    If a TYPE_CHECKING block already exists, new imports are appended to it
    (skipping any already present).  Otherwise a new block is inserted after
    the last top-level import statement, along with ``from typing import
    TYPE_CHECKING`` when that name is not already imported.
    """
    if not imports:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    # Determine which imports are not already in an existing TC block.
    existing_tc = {i.source for i in _extract_import_info(source) if i.is_type_checking}
    new_imports = [imp for imp in imports if imp not in existing_tc]
    if not new_imports:
        return source

    lines = source.splitlines(keepends=True)

    # Append to an existing TYPE_CHECKING block if one is present.
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            insert_line = node.end_lineno
            new_lines = ["    " + imp + "\n" for imp in sorted(new_imports)]
            return "".join(lines[:insert_line] + new_lines + lines[insert_line:])

    # No existing block: insert one after the last top-level import.
    last_import_line = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_line = max(last_import_line, node.end_lineno)
    insert_after = last_import_line

    tc_already_imported = any(
        isinstance(n, ast.ImportFrom)
        and n.module == "typing"
        and any((a.asname or a.name) == "TYPE_CHECKING" for a in n.names)
        for n in tree.body
    )
    new_lines = []
    if not tc_already_imported:
        new_lines.append("from typing import TYPE_CHECKING\n")
    new_lines.append("if TYPE_CHECKING:\n")
    for imp in sorted(new_imports):
        new_lines.append("    " + imp + "\n")
    new_lines.append("\n")
    return "".join(lines[:insert_after] + new_lines + lines[insert_after:])


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
        if info.is_type_checking:
            continue  # handled by _find_type_checking_needed_imports
        if info.is_future or any(n in referenced for n in info.names):
            needed.append(info.source)
            seen.add(info.source)

    return needed


def _narrow_import_source(import_src: str, keep_names: Set[str]) -> str:
    """Return a copy of *import_src* keeping only the exposed names in *keep_names*.

    For ``from X import A, B, C`` with ``keep_names={A}``, returns
    ``from X import A``.  Non-ImportFrom statements are returned unchanged.
    """
    try:
        node = ast.parse(import_src).body[0]
    except (SyntaxError, IndexError):
        return import_src
    if not isinstance(node, ast.ImportFrom):
        return import_src
    dots = "." * (node.level or 0)
    mod = node.module or ""
    alias_strs = [
        f"{a.name} as {a.asname}" if a.asname else a.name
        for a in node.names
        if (a.asname or a.name) in keep_names
    ]
    if not alias_strs:
        return import_src
    return f"from {dots}{mod} import {', '.join(alias_strs)}"


def _find_type_checking_needed_imports(
    entity_names: List[str],
    entity_source_map: Dict[str, str],
    import_infos: List[ImportInfo],
) -> List[str]:
    """Return import statements needed only for quoted type annotations.

    These should be placed under ``if TYPE_CHECKING:`` because the names are
    only referenced inside string-valued annotations (forward references) and
    are not needed at runtime.  Names that appear in regular (non-annotation)
    loads are excluded via ``annotation_only = quoted - runtime``, which
    guarantees that any name emitted here will be pruned from regular imports
    by ``_prune_unused_imports`` — so no duplicate imports can arise.
    ``__future__`` imports are always excluded since they are handled by
    ``_find_needed_imports``.
    """
    runtime: Set[str] = set()
    quoted: Set[str] = set()
    for name in entity_names:
        src = entity_source_map.get(name, "")
        runtime |= _collect_name_loads(src)
        quoted |= _collect_quoted_annotation_names(src)

    annotation_only = quoted - runtime
    if not annotation_only:
        return []

    needed: List[str] = []
    seen: Set[str] = set()
    for info in import_infos:
        if info.source in seen:
            continue
        if info.is_future:
            continue
        tc_names = {n for n in info.names if n in annotation_only}
        if not tc_names:
            continue
        # Narrow the import to only the names actually needed for type checking,
        # avoiding unused-import warnings for names from multi-name imports that
        # are not referenced in this file.
        src = (
            info.source
            if len(tc_names) == len(info.names)
            else _narrow_import_source(info.source, tc_names)
        )
        if src in seen:
            continue
        needed.append(src)
        seen.add(src)
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


def _strip_top_level_import_lines(src: str) -> str:
    """Return *src* with all top-level import statements removed.

    Also removes module-level ``if TYPE_CHECKING:`` blocks, since their
    imports are now redistributed to each sub-file via the import-info
    system and emitting the block verbatim would produce the wrong relative
    import path and/or an unused import in the wrong sub-file.

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
        elif (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for ln in range(node.lineno, node.end_lineno + 1):
                remove.add(ln)
    if not remove:
        return src
    lines = src.splitlines(keepends=True)
    return "".join(line for i, line in enumerate(lines, 1) if i not in remove)
