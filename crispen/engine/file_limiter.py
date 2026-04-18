from pathlib import Path
from typing import Dict, List, Optional, Set, TYPE_CHECKING, Tuple
import ast
from ..stats import RunStats
from ..patch_rewriter import _FLContext


if TYPE_CHECKING:
    from ..file_limiter.runner import FileLimiterResult


_PROJECT_MARKERS = frozenset({"pyproject.toml", "setup.py", "setup.cfg", ".git"})


def _collect_top_level_names(source: str) -> Set[str]:
    """Return all names defined or imported at the module top level of *source*.

    Covers functions, classes, module-level variable assignments, and all
    import styles.  Returns an empty set when *source* cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname if alias.asname else alias.name)
    return names


def _collect_assignment_names(source: str) -> Set[str]:
    """Return names from top-level variable assignments in *source*.

    Covers ``X = …``, ``X: T = …``, and ``X += …`` where the target is a
    plain ``ast.Name``.  Functions, classes, and imports are excluded (they
    are handled elsewhere).  Returns an empty set on parse failure.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def _collect_imported_names(source: str) -> Set[str]:
    """Return names imported at the top level of *source*.

    Handles ``import X``, ``import X as Y``, ``from X import Y``, and
    ``from X import Y as Z``.  Star imports (``from X import *``) are skipped.
    Returns an empty set when *source* cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname if alias.asname else alias.name)
    return names


def _collect_code_referenced_names(source: str) -> Set[str]:
    """Return names referenced in code as *Load* expressions.

    Walks the AST looking for ``ast.Name`` nodes with ``Load`` context —
    actual uses of a name in code (calls, attribute targets, right-hand-side
    expressions).  Pure re-export stubs (``from .sub import X  # noqa: F401``)
    produce no such nodes because the import alias itself is an ``ast.alias``,
    not an ``ast.Name``.  Returns an empty set on parse failure.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _module_path_for_file(file_path: str) -> Optional[str]:
    """Return the dotted Python module path of *file_path*.

    Walks up from the file's directory to find the project root (the first
    ancestor containing pyproject.toml, setup.py, setup.cfg, or .git), then
    computes the dotted path relative to that root.

    ``__init__.py`` files are mapped to their package name (e.g.
    ``mypkg/__init__.py`` → ``mypkg``) so that patch paths always use the
    public package namespace rather than the internal ``.__init__`` segment.

    Returns ``None`` when the project root cannot be determined or when the
    resolved path is not under the project root.
    """
    abs_path = Path(file_path).resolve()
    current = abs_path.parent
    while True:
        if any((current / m).exists() for m in _PROJECT_MARKERS):
            rel = abs_path.relative_to(current)
            module = ".".join(rel.with_suffix("").parts)
            if module.endswith(".__init__"):
                module = module[:-9]
            return module
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _redirect_inline_module_imports(
    source: str,
    old_mod: str,
    name_to_new_mod: Dict[str, str],
) -> str:
    """Replace ``from old_mod import names`` statements with new module paths.

    Scans all ``from <old_mod> import …`` statements at every level
    (module-level and inside function/class bodies).  Each imported name is
    redirected to the module given by *name_to_new_mod*; names absent from the
    map are kept pointing at *old_mod*.

    Returns *source* unchanged when there are no matching imports or when
    *source* cannot be parsed as Python.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines(keepends=True)
    replacements: Dict[Tuple[int, int], str] = {}

    def _scan(stmts: list) -> None:
        for node in stmts:
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module == old_mod
            ):
                names = [alias.name for alias in node.names]
                new_mod_to_names: Dict[str, List[str]] = {}
                kept: List[str] = []
                for n in names:
                    dest = name_to_new_mod.get(n)
                    if dest:
                        new_mod_to_names.setdefault(dest, []).append(n)
                    else:
                        kept.append(n)
                if not new_mod_to_names:
                    continue  # No names moved; leave unchanged.
                raw_line = lines[node.lineno - 1]
                indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
                parts: List[str] = []
                if kept:
                    parts.append(f"{indent}from {old_mod} import {', '.join(kept)}")
                for dest_mod, dest_names in sorted(new_mod_to_names.items()):
                    joined = ", ".join(sorted(dest_names))
                    parts.append(f"{indent}from {dest_mod} import {joined}")
                replacements[(node.lineno, node.end_lineno)] = "\n".join(parts)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                _scan(node.body)

    _scan(tree.body)
    if not replacements:
        return source

    result = list(lines)
    for (start, end), text in sorted(replacements.items(), reverse=True):
        last_line = result[end - 1]
        trailing = "\n" if last_line.endswith("\n") else ""
        result[start - 1 : end] = [text + trailing]
    return "".join(result)


def _patch_inline_imports_after_test_deletion(
    deleted_path: str,
    deleted_dir: Path,
    new_files: Dict[str, str],
    per_file: Dict,
    fl_new_file_final: Dict[str, Optional[str]],
) -> None:
    """Redirect inline imports pointing to a deleted test module.

    When a test file is deleted after a recursive subdir split (because all
    entities migrated away, leaving an empty ``original_source``), any parent
    file that received injected inline imports during the *first* split still
    references the now-gone module.  This function rewrites those stale imports
    to point directly at the new sub-file locations.

    Updates both ``per_file`` states (written to disk later by the write loop)
    and ``fl_new_file_final`` entries (already on disk; re-written immediately).
    """
    old_mod = _module_path_for_file(deleted_path)
    if old_mod is None:
        return

    # Build name → new_module by parsing top-level definitions in new_files.
    name_to_new_mod: Dict[str, str] = {}
    for rel_path, content in new_files.items():
        new_abs = (deleted_dir / rel_path).resolve()
        new_mod = _module_path_for_file(str(new_abs))
        if new_mod is None:
            continue
        try:
            sub_tree = ast.parse(content)
        except SyntaxError:
            continue
        for node in sub_tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name_to_new_mod[node.name] = new_mod

    if not name_to_new_mod:
        return

    for state in per_file.values():
        updated = _redirect_inline_module_imports(
            state["source"], old_mod, name_to_new_mod
        )
        if updated != state["source"]:
            state["source"] = updated

    for path, content in list(fl_new_file_final.items()):
        if not content:
            continue
        updated = _redirect_inline_module_imports(content, old_mod, name_to_new_mod)
        if updated != content:
            fl_new_file_final[path] = updated
            Path(path).write_text(updated, encoding="utf-8")


def _build_patch_map(
    filepath: str,
    fl_result: "FileLimiterResult",
    original_dir: Path,
    pre_split_source: str = "",
) -> Dict[str, str]:
    """Build old_dotted_path.EntityName → new_dotted_path.EntityName map.

    Uses "basic" mode logic: for each entity, maps to the single new file that
    imports it (the "caller") rather than its definition file.  If multiple new
    files import the entity (forking), the entity is skipped.  Import aliases
    present in *pre_split_source* that appear in exactly one new file are also
    included.

    Returns an empty dict when the module path cannot be determined or when no
    entities were moved.
    """
    old_module = _module_path_for_file(filepath)
    if old_module is None:
        return {}

    # Build import index: name → list of new-file rel_paths that import it.
    # Build usage index: name → set of new-file rel_paths that reference it in
    # code (ast.Name Load nodes — actual calls or expressions, not re-exports).
    import_index: Dict[str, List[str]] = {}
    usage_index: Dict[str, Set[str]] = {}
    for rel_path, src in fl_result.new_files.items():
        if not src:
            continue
        for name in _collect_imported_names(src):
            import_index.setdefault(name, []).append(rel_path)
        for name in _collect_code_referenced_names(src):
            usage_index.setdefault(name, set()).add(rel_path)

    patch_map: Dict[str, str] = {}
    for entity_name, def_rel_target in fl_result.entity_to_target.items():
        # Callers = new files that both import this entity and reference it in
        # code, excluding its definer.  Pure re-export stubs import the name
        # but produce no ast.Name Load nodes, so they are naturally excluded.
        callers = [
            p
            for p in import_index.get(entity_name, [])
            if p != def_rel_target and p in usage_index.get(entity_name, set())
        ]
        if len(callers) > 1:
            continue  # forking: multiple callers, skip
        target_rel = callers[0] if callers else def_rel_target
        new_module = _module_path_for_file(str(original_dir / target_rel))
        if new_module is None:
            continue
        patch_map[f"{old_module}.{entity_name}"] = f"{new_module}.{entity_name}"

    # Import aliases: names imported by the original file that appear in
    # exactly one new sub-file (and are not already moved entities).
    # Only count files that actually reference the alias in code, not stubs.
    for alias_name in _collect_imported_names(pre_split_source):
        if alias_name in fl_result.entity_to_target:
            continue  # already handled above
        importers = [
            p
            for p in import_index.get(alias_name, [])
            if p in usage_index.get(alias_name, set())
        ]
        if len(importers) != 1:
            continue  # zero or multiple: skip
        new_module = _module_path_for_file(str(original_dir / importers[0]))
        if new_module is None:
            continue
        patch_map[f"{old_module}.{alias_name}"] = f"{new_module}.{alias_name}"

    # Variable assignments: names assigned at the module level that were present
    # in the original file and are not already tracked as entities or import
    # aliases.  Uses the same caller logic as named entities: 0 callers → only
    # used in its defining file; 1 caller → migrated and consumed by exactly one
    # other file; 2+ → forking.  Only names from the original file are considered
    # to avoid spurious entries for helper variables introduced by code generation.
    # Names defined in multiple new files are skipped (ambiguous origin).
    orig_assignments = _collect_assignment_names(pre_split_source)
    def_index: Dict[str, List[str]] = {}
    for rel_path, src in fl_result.new_files.items():
        if not src:
            continue
        for name in _collect_assignment_names(src):
            if name in orig_assignments and name not in fl_result.entity_to_target:
                def_index.setdefault(name, []).append(rel_path)

    for assign_name, def_rel_targets in def_index.items():
        if len(def_rel_targets) != 1:
            continue  # defined in multiple files → ambiguous
        old_path = f"{old_module}.{assign_name}"
        if old_path in patch_map:
            continue  # already handled by entity or import-alias section
        def_rel_target = def_rel_targets[0]
        callers = [
            p
            for p in import_index.get(assign_name, [])
            if p != def_rel_target and p in usage_index.get(assign_name, set())
        ]
        if len(callers) > 1:
            continue  # forking: multiple consumers, skip
        target_rel = callers[0] if callers else def_rel_target
        new_module = _module_path_for_file(str(original_dir / target_rel))
        if new_module is None:
            continue
        patch_map[old_path] = f"{new_module}.{assign_name}"

    return patch_map


def _add_fl_context(
    fl_all_contexts: List["_FLContext"],
    filepath: str,
    pre_split_src: str,
    fl_result: "FileLimiterResult",
    combined_patch_map: Dict[str, str],
) -> None:
    """Append an _FLContext to *fl_all_contexts* for "rewrite" patch mode.

    Computes the forking old paths (entities in entity_to_target that basic
    mode skipped because they appeared in multiple callers) and builds the
    new module path map for all sub-files.  Does nothing when no forking
    entities exist or when the module path cannot be determined.

    When no forking entities exist but TOP_LEVEL blocks (_block_N) were
    moved, also scans the new target files to find names that came from
    those blocks (module-level vars, constants, imported aliases) but are
    not individually tracked in entity_to_target.  Each such name is added
    as a specific old path (``old_module.name``) so the LLM can find any
    ``with patch(old_module.name)`` calls without matching already-updated
    paths like ``old_module.sub.name`` that basic mode already rewrote.

    Import aliases from the original file that basic mode skipped (forked
    into multiple new sub-files) are also added so the LLM rewrite step
    can determine the correct per-function patch target.
    """
    old_mod = _module_path_for_file(filepath)
    if old_mod is None:
        return
    forking_old_paths = {
        f"{old_mod}.{name}"
        for name in fl_result.entity_to_target
        if f"{old_mod}.{name}" not in combined_patch_map
    }
    # Also collect names from moved _block_N entities that are NOT individually
    # tracked (i.e., not in entity_to_target).  These are block-internal names
    # (vars, constants, imported aliases) that basic mode never maps, regardless
    # of whether forking entities were also found above.
    all_entity_names = set(fl_result.entity_to_target)
    for entity_name, target_rel in fl_result.entity_to_target.items():
        if not entity_name.startswith("_block_"):
            continue
        new_src = fl_result.new_files.get(target_rel, "")
        for name in _collect_top_level_names(new_src):
            old_path = f"{old_mod}.{name}"
            if name not in all_entity_names and old_path not in combined_patch_map:
                forking_old_paths.add(old_path)
    # Also add import aliases from the original file that basic mode skipped
    # because they appeared in multiple new sub-files (forking).  These
    # aliases are absent from combined_patch_map but may still appear as
    # @patch string targets in test files — the LLM rewrite step can resolve
    # the correct sub-module for each test function individually.
    for alias_name in _collect_imported_names(pre_split_src):
        if alias_name in all_entity_names:
            continue
        old_path = f"{old_mod}.{alias_name}"
        if old_path not in combined_patch_map:
            forking_old_paths.add(old_path)
    if not forking_old_paths:
        return
    orig_dir = Path(filepath).parent
    new_mod_paths = {
        rel: _module_path_for_file(str(orig_dir / rel)) or rel
        for rel in fl_result.new_files
    }
    # For non-test subdir splits the original file stays on disk unchanged and
    # fl_result.original_source is the pre-split source (runner.py restores it
    # at line 704 so the original file is left untouched).  The post-split
    # module state lives in new_files["{subdir_name}/__init__.py"].  Use that
    # as modified_source so _build_rename_guard_sets and the BFS terminal
    # builder both see the correct set of names still present in the module.
    init_key = f"{fl_result.subdir_name}/__init__.py" if fl_result.subdir_name else None
    if init_key and init_key in fl_result.new_files:
        modified_src = fl_result.new_files[init_key] or fl_result.original_source or ""
    else:
        modified_src = fl_result.original_source or ""
    fl_all_contexts.append(
        _FLContext(
            filepath=filepath,
            old_module=old_mod,
            original_source=pre_split_src,
            modified_source=modified_src,
            new_files=dict(fl_result.new_files),
            new_module_paths=new_mod_paths,
            entity_to_target=dict(fl_result.entity_to_target),
            forking_old_paths=forking_old_paths,
        )
    )


def _categorize_into_stats(stats: RunStats, msg: str) -> None:
    """Increment the appropriate counter in *stats* for a raw change message."""
    if msg.startswith("IfNotElse:"):
        stats.if_not_else += 1
    elif msg.startswith("TupleDataclass:"):
        stats.tuple_to_dataclass += 1
    elif msg.startswith("DuplicateExtractor:") and "with call to" in msg:
        stats.duplicate_matched += 1
    elif msg.startswith("DuplicateExtractor:"):
        stats.duplicate_extracted += 1
    elif msg.startswith("split "):
        stats.function_split += 1
