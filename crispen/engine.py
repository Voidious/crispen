"""Load files, apply refactors, verify, and write back."""

import ast
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Generator, List, NamedTuple, Optional, Set, Tuple

from .stats import RunStats

import libcst as cst
from libcst.metadata import FullRepoManager, MetadataWrapper, QualifiedNameProvider

from .config import CrispenConfig, format_header, load_config
from .errors import CrispenAPIError
from .file_limiter.runner import run_file_limiter
from .refactors.caller_updater import CallerUpdater
from .refactors.duplicate_extractor import DuplicateExtractor
from .refactors.function_splitter import FunctionSplitter
from .refactors.if_not_else import IfNotElse
from .refactors.tuple_dataclass import TransformInfo, TupleDataclass

# Single-file refactors applied in order before TupleDataclass.
_REFACTORS = [IfNotElse, DuplicateExtractor, FunctionSplitter]

# Refactor keys that invoke LLM calls (used to decide whether to print config).
_LLM_REFACTOR_KEYS = frozenset(
    {"duplicate_extractor", "function_splitter", "tuple_dataclass", "file_limiter"}
)

# Canonical snake_case name for each refactor class (used by _should_run).
_REFACTOR_KEY: Dict[type, str] = {
    IfNotElse: "if_not_else",
    DuplicateExtractor: "duplicate_extractor",
    FunctionSplitter: "function_splitter",
}


def _should_run(name: str, config: CrispenConfig) -> bool:
    """Return True if the named refactor should run given the config.

    When ``config.enabled_refactors`` is non-empty only names in that list run.
    Otherwise names in ``config.disabled_refactors`` are skipped.
    """
    if config.enabled_refactors:
        return name in config.enabled_refactors
    return name not in config.disabled_refactors


# Directory names excluded from the outside-caller scan (e.g. virtual environments).
_EXCLUDED_DIR_NAMES = frozenset(
    {".venv", "venv", "env", ".tox", "__pycache__", "node_modules"}
)

# Total wall-clock budget for all files in _find_outside_callers (seconds).
_SCOPE_ANALYSIS_TIMEOUT = 10


# ---------------------------------------------------------------------------
# update_diff_file_callers helpers
# ---------------------------------------------------------------------------


def _has_callers_outside_ranges(
    source: str, func_name: str, ranges: List[Tuple[int, int]]
) -> bool:
    """Return True if func_name is called at any line outside the given ranges."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == func_name
        ):
            line = node.lineno
            if not any(start <= line <= end for start, end in ranges):
                return True
    return False


def _blocked_private_scopes(source: str, ranges: List[Tuple[int, int]]) -> Set[str]:
    """Return names of private functions that have callers outside the diff ranges."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    blocked: Set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("_")
        ):
            line = node.lineno
            if not any(start <= line <= end for start, end in ranges):
                blocked.add(node.func.id)
    return blocked


# ---------------------------------------------------------------------------
# FileLimiter: inline-import redirect helpers
# ---------------------------------------------------------------------------

_PROJECT_MARKERS = frozenset({"pyproject.toml", "setup.py", "setup.cfg", ".git"})


def _module_path_for_file(file_path: str) -> Optional[str]:
    """Return the dotted Python module path of *file_path*.

    Walks up from the file's directory to find the project root (the first
    ancestor containing pyproject.toml, setup.py, setup.cfg, or .git), then
    computes the dotted path relative to that root.

    Returns ``None`` when the project root cannot be determined or when the
    resolved path is not under the project root.
    """
    abs_path = Path(file_path).resolve()
    current = abs_path.parent
    while True:
        if any((current / m).exists() for m in _PROJECT_MARKERS):
            rel = abs_path.relative_to(current)
            return ".".join(rel.with_suffix("").parts)
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


# ---------------------------------------------------------------------------
# Repo-root helpers
# ---------------------------------------------------------------------------


def _find_repo_root(changed: Dict[str, List]) -> Optional[str]:
    """Find git repo root by searching parent directories for .git."""
    for filepath in changed.keys():
        p = Path(filepath).resolve().parent
        while p != p.parent:
            if (p / ".git").is_dir():
                return str(p)
            p = p.parent
    return None


def _file_to_module(repo_root: str, filepath: str) -> str:
    """Convert an absolute file path to a dotted Python module name."""
    path = Path(filepath).resolve().relative_to(Path(repo_root).resolve())
    module = str(path.with_suffix("")).replace(os.sep, ".")
    if module.endswith(".__init__"):
        module = module[:-9]
    return module


def _compute_qname(repo_root: str, filepath: str, func_name: str) -> str:
    """Compute the qualified name of a function defined in filepath."""
    return f"{_file_to_module(repo_root, filepath)}.{func_name}"


# ---------------------------------------------------------------------------
# __init__.py alias resolution
# ---------------------------------------------------------------------------


def _build_alias_map(repo_root: str, canonical_qnames: Set[str]) -> Dict[str, str]:
    """Map alias qualified names → canonical qualified names.

    Handles explicit re-exports like ``from .service import get_user`` in
    ``pkg/__init__.py``, which creates the alias ``pkg.get_user`` for the
    canonical name ``pkg.service.get_user``.
    """
    alias_map: Dict[str, str] = {q: q for q in canonical_qnames}

    for init_path in Path(repo_root).rglob("__init__.py"):
        pkg_parts = list(init_path.relative_to(repo_root).parts[:-1])
        pkg_qname = ".".join(pkg_parts)

        try:
            source = init_path.read_text(encoding="utf-8")
            tree = cst.parse_module(source)
        except Exception:
            continue

        for stmt in tree.body:
            if not isinstance(stmt, cst.SimpleStatementLine):
                continue
            for s in stmt.body:
                if not isinstance(s, cst.ImportFrom):
                    continue
                if isinstance(s.names, cst.ImportStar) or not isinstance(
                    s.names, (list, tuple)
                ):
                    continue
                for al in s.names:
                    if not isinstance(al, cst.ImportAlias) or not isinstance(
                        al.name, cst.Name
                    ):
                        continue  # pragma: no cover
                    func_name = al.name.value
                    alias_qname = f"{pkg_qname}.{func_name}" if pkg_qname else func_name
                    # Map this alias to a canonical qname if unambiguous
                    matches = [
                        c for c in canonical_qnames if c.split(".")[-1] == func_name
                    ]
                    if len(matches) == 1:
                        alias_map[alias_qname] = matches[0]

    return alias_map


# ---------------------------------------------------------------------------
# Outside-caller detection using FullRepoManager
# ---------------------------------------------------------------------------


class _CallerFinder(cst.CSTVisitor):
    """Visit a file and record which target qualified names are called."""

    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(self, target_qnames: Set[str]) -> None:
        self.target_qnames = target_qnames
        self.found: Set[str] = set()

    def visit_Call(self, node: cst.Call) -> None:
        qnames = self.get_metadata(QualifiedNameProvider, node.func, set())
        for qn in qnames:
            if qn.name in self.target_qnames:
                self.found.add(qn.name)


def _visit_with_timeout(wrapper, finder, timeout: float) -> bool:
    """Run wrapper.visit(finder) in a daemon thread with a wall-clock timeout.

    Returns True if the call completed within *timeout* seconds, False if it
    timed out (libcst scope analysis can hang on large files).
    """
    done = threading.Event()

    def _target():
        try:
            wrapper.visit(finder)
        finally:
            done.set()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return done.wait(timeout=timeout)


def _find_outside_callers(
    repo_root: str,
    target_qnames: Set[str],
    diff_files: Set[str],
) -> Set[str]:
    """Return the subset of *target_qnames* called in files outside *diff_files*."""
    if not target_qnames:
        return set()

    repo_root_path = Path(repo_root)
    outside_py = [
        p
        for p in repo_root_path.rglob("*.py")
        if str(p.resolve()) not in diff_files
        and not any(
            part in _EXCLUDED_DIR_NAMES
            for part in p.relative_to(repo_root_path).parts[:-1]
        )
    ]
    if not outside_py:
        return set()

    rel_paths = [str(p.relative_to(repo_root)) for p in outside_py]

    try:
        manager = FullRepoManager(repo_root, rel_paths, {QualifiedNameProvider})
    except Exception:
        # Can't build the manager → conservatively block all transforms.
        return set(target_qnames)

    found_outside: Set[str] = set()
    deadline = time.monotonic() + _SCOPE_ANALYSIS_TIMEOUT
    for rel_path in rel_paths:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Total budget exhausted: conservatively block all remaining.
            found_outside.update(target_qnames)
            break
        try:
            wrapper = manager.get_metadata_wrapper_for_path(rel_path)
            finder = _CallerFinder(target_qnames)
            if not _visit_with_timeout(wrapper, finder, remaining):
                # This file timed out: conservatively block all transforms.
                found_outside.update(target_qnames)
                break
            found_outside.update(finder.found)
        except Exception:
            continue

    return found_outside


# ---------------------------------------------------------------------------
# TupleDataclass helper (used in both passes)
# ---------------------------------------------------------------------------


class _ApplyResult(NamedTuple):
    """Return type of _apply_tuple_dataclass."""

    source: str
    msgs: List[str]
    td: Optional[TupleDataclass]


def _apply_tuple_dataclass(
    filepath: str,
    ranges: List[Tuple[int, int]],
    source: str,
    verbose: bool,
    approved_public_funcs: Set[str],
    min_size: int = 4,
    blocked_scopes: Optional[Set[str]] = None,
) -> "_ApplyResult":
    """Run TupleDataclass on *source*. Returns (new_source, messages, transformer)."""
    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        return _ApplyResult(
            source, [f"SKIP {filepath} (TupleDataclass): parse error: {exc}"], None
        )

    wrapper = MetadataWrapper(tree)
    try:
        td = TupleDataclass(
            ranges,
            source=source,
            verbose=verbose,
            approved_public_funcs=approved_public_funcs,
            min_size=min_size,
            blocked_scopes=blocked_scopes,
        )
        new_tree = wrapper.visit(td)
    except CrispenAPIError:
        raise
    except Exception as exc:
        return _ApplyResult(
            source,
            [f"SKIP {filepath} (TupleDataclass): transform error: {exc}"],
            None,
        )

    new_source = td.get_rewritten_source() or new_tree.code
    if new_source == source:
        return _ApplyResult(source, [], td)

    try:
        compile(new_source, filepath, "exec")
    except SyntaxError as exc:  # pragma: no cover
        return _ApplyResult(
            source,
            [f"SKIP {filepath} (TupleDataclass): output not valid Python: {exc}"],
            td,
        )

    msgs = [f"{filepath}: {m}" for m in td.get_changes()]
    return _ApplyResult(new_source, msgs, td)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


def run_engine(
    changed: Dict[str, List[Tuple[int, int]]],
    verbose: bool = True,
    _repo_root: Optional[str] = None,
    config: Optional[CrispenConfig] = None,
    stats: Optional[RunStats] = None,
) -> Generator[str, None, None]:
    """Apply all refactors to changed files and yield summary messages."""
    if config is None:
        config = load_config()
    _stats = stats if stats is not None else RunStats()

    if changed and any(_should_run(k, config) for k in _LLM_REFACTOR_KEYS):
        for line in format_header(config):
            print(line, file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ #
    # Phase 1 — single-file refactors + TupleDataclass (private only)     #
    # ------------------------------------------------------------------ #
    per_file: Dict[str, dict] = {}

    for filepath, ranges in changed.items():
        path = Path(filepath)
        if not path.exists():
            yield f"SKIP {filepath}: file not found"
            continue

        original_source = path.read_text(encoding="utf-8")
        current_source = original_source
        file_msgs: List[str] = []
        had_parse_error = False

        for RefactorClass in _REFACTORS:
            key = _REFACTOR_KEY.get(RefactorClass)
            if key is not None and not _should_run(key, config):
                continue
            try:
                current_tree = cst.parse_module(current_source)
            except cst.ParserSyntaxError as exc:
                file_msgs.append(
                    f"SKIP {filepath} ({RefactorClass.name()}): parse error: {exc}"
                )
                had_parse_error = True
                break

            wrapper = MetadataWrapper(current_tree)
            try:
                if RefactorClass is DuplicateExtractor:
                    transformer = DuplicateExtractor(
                        ranges,
                        source=current_source,
                        verbose=verbose,
                        min_weight=config.min_duplicate_weight,
                        max_seq_len=config.max_duplicate_seq_len,
                        model=config.model,
                        helper_docstrings=config.helper_docstrings,
                        provider=config.provider,
                        extraction_retries=config.extraction_retries,
                        llm_verify_retries=config.llm_verify_retries,
                        base_url=config.base_url,
                        tool_choice=config.tool_choice,
                        api_timeout=config.api_timeout,
                        match_functions=_should_run("match_function", config),
                        timing=config.timing,
                        current_file=filepath,
                    )
                elif RefactorClass is FunctionSplitter:
                    transformer = FunctionSplitter(
                        ranges,
                        source=current_source,
                        verbose=verbose,
                        max_lines=config.max_function_length,
                        model=config.model,
                        provider=config.provider,
                        helper_docstrings=config.helper_docstrings,
                        base_url=config.base_url,
                        tool_choice=config.tool_choice,
                        api_timeout=config.api_timeout,
                        current_file=filepath,
                    )
                else:
                    transformer = RefactorClass(
                        ranges, source=current_source, verbose=verbose
                    )
                transformer.current_file = filepath
                transformer.timing = config.timing
                new_tree = wrapper.visit(transformer)
            except CrispenAPIError:
                raise
            except Exception as exc:
                name = RefactorClass.name()
                file_msgs.append(f"SKIP {filepath} ({name}): transform error: {exc}")
                continue

            rewritten = transformer.get_rewritten_source()
            new_source = rewritten if rewritten is not None else new_tree.code
            if new_source == current_source:
                continue

            try:
                compile(new_source, filepath, "exec")
            except SyntaxError as exc:  # pragma: no cover
                name = RefactorClass.name()
                file_msgs.append(
                    f"SKIP {filepath} ({name}): output not valid Python: {exc}"
                )
                continue

            for msg in transformer.get_changes():
                file_msgs.append(f"{filepath}: {msg}")
                _categorize_into_stats(_stats, msg)
            _stats.merge(transformer.stats)
            current_source = new_source

        # Apply TupleDataclass — private functions only in this pass.
        candidates: Dict[str, TransformInfo] = {}
        if not had_parse_error and _should_run("tuple_dataclass", config):
            blocked: Set[str] = set()
            if not config.update_diff_file_callers:
                blocked = _blocked_private_scopes(current_source, ranges)
            new_source, msgs, td = _apply_tuple_dataclass(
                filepath,
                ranges,
                current_source,
                verbose,
                approved_public_funcs=set(),
                min_size=config.min_tuple_size,
                blocked_scopes=blocked,
            )
            current_source = new_source
            file_msgs.extend(msgs)
            if td is not None:
                for m in td.get_changes():
                    _categorize_into_stats(_stats, m)
                candidates = td.get_candidate_public_transforms()
                # Run CallerUpdater for private function callers in this file.
                private_transforms = td.get_private_transforms()
                if private_transforms:
                    try:
                        cu_tree = cst.parse_module(current_source)
                        cu_wrapper = MetadataWrapper(cu_tree)
                        cu = CallerUpdater(
                            ranges,
                            transforms={},
                            local_transforms=private_transforms,
                            source=current_source,
                            verbose=verbose,
                        )
                        cu_new_source = cu_wrapper.visit(cu).code
                    except Exception:
                        cu_new_source = current_source
                    if cu_new_source != current_source:
                        try:
                            compile(cu_new_source, filepath, "exec")
                        except SyntaxError:  # pragma: no cover
                            pass
                        else:
                            for msg in cu.get_changes():
                                file_msgs.append(f"{filepath}: {msg}")
                                _categorize_into_stats(_stats, msg)
                            current_source = cu_new_source

        per_file[filepath] = {
            "original": original_source,
            "source": current_source,
            "msgs": file_msgs,
            "candidates": candidates,
            "ranges": ranges,
        }

    # ------------------------------------------------------------------ #
    # Phase 2 — cross-file public-function transforms + caller updates    #
    # ------------------------------------------------------------------ #
    repo_root = _repo_root if _repo_root is not None else _find_repo_root(changed)

    if repo_root and per_file:
        # Collect all public-function candidates with their qualified names.
        all_candidates: Dict[str, Tuple[TransformInfo, str]] = {}
        for filepath, state in per_file.items():
            for func_name, info in state["candidates"].items():
                try:
                    qname = _compute_qname(repo_root, filepath, func_name)
                    all_candidates[qname] = (info, filepath)
                except ValueError:
                    pass  # file not under repo_root

        if all_candidates:
            canonical_qnames = set(all_candidates.keys())
            alias_map = _build_alias_map(repo_root, canonical_qnames)
            all_qnames = set(alias_map.keys())  # canonical + __init__ aliases

            diff_files = {str(Path(f).resolve()) for f in per_file}
            outside_callers = _find_outside_callers(repo_root, all_qnames, diff_files)

            # Any alias with an outside caller blocks its canonical transform.
            outside_canonical = {
                alias_map[q] for q in outside_callers if q in alias_map
            }

            # When update_diff_file_callers is disabled, also block functions
            # that have callers within diff files but outside the diff ranges.
            if not config.update_diff_file_callers:
                for qname in list(canonical_qnames - outside_canonical):
                    info, _ = all_candidates[qname]
                    for caller_state in per_file.values():
                        if _has_callers_outside_ranges(
                            caller_state["source"],
                            info.func_name,
                            caller_state["ranges"],
                        ):
                            outside_canonical.add(qname)
                            break

            approved_canonical = canonical_qnames - outside_canonical

            for qname in canonical_qnames - approved_canonical:
                info, filepath = all_candidates[qname]
                yield (
                    f"SKIP {filepath}: {info.func_name}:"
                    f" callers exist outside the diff"
                )

            if approved_canonical:
                # Build the transforms dict for CallerUpdater (all names → info).
                approved_transforms: Dict[str, TransformInfo] = {}
                approved_by_file: Dict[str, Set[str]] = {}

                for qname in approved_canonical:
                    info, filepath = all_candidates[qname]
                    approved_transforms[qname] = info
                    approved_by_file.setdefault(filepath, set()).add(info.func_name)

                for alias, canonical in alias_map.items():
                    if canonical in approved_canonical:
                        approved_transforms[alias] = all_candidates[canonical][0]

                # Second TupleDataclass pass — approved public functions only.
                for filepath, funcs in approved_by_file.items():
                    state = per_file[filepath]
                    new_source, msgs, td2 = _apply_tuple_dataclass(
                        filepath,
                        state["ranges"],
                        state["source"],
                        verbose,
                        approved_public_funcs=funcs,
                        min_size=config.min_tuple_size,
                    )
                    state["source"] = new_source
                    state["msgs"].extend(msgs)
                    if td2 is not None:
                        for m in td2.get_changes():
                            _categorize_into_stats(_stats, m)

                # CallerUpdater pass — all diff files.
                for filepath, state in per_file.items():
                    try:
                        file_module = _file_to_module(repo_root, filepath)
                    except ValueError:
                        continue

                    try:
                        current_tree = cst.parse_module(state["source"])
                    except cst.ParserSyntaxError:
                        continue

                    wrapper = MetadataWrapper(current_tree)
                    try:
                        cu = CallerUpdater(
                            state["ranges"],
                            approved_transforms,
                            file_module=file_module,
                            source=state["source"],
                            verbose=verbose,
                        )
                        new_tree = wrapper.visit(cu)
                    except Exception:
                        continue

                    new_source = new_tree.code
                    if new_source == state["source"]:
                        continue

                    try:
                        compile(new_source, filepath, "exec")
                    except SyntaxError:  # pragma: no cover
                        continue

                    for msg in cu.get_changes():
                        state["msgs"].append(f"{filepath}: {msg}")
                        _categorize_into_stats(_stats, msg)
                    state["source"] = new_source

    # ------------------------------------------------------------------ #
    # Phase 3 — FileLimiter: split files exceeding max_file_lines        #
    # ------------------------------------------------------------------ #
    if config.max_file_lines > 0 and _should_run("file_limiter", config):
        # Pending queue for recursive FileLimiter processing: (filepath, source)
        # pairs for newly-created files that are still over the limit.
        _fl_recursive: List[Tuple[str, str]] = []
        # Track the final content of each new file created by FileLimiter so
        # lines_added/deleted counts reflect the net result, not interim states.
        _fl_new_file_final: Dict[str, Optional[str]] = {}
        # Deduplicate verified functions/classes across recursive passes so
        # entities migrated more than once are not counted multiple times.
        _fl_verified_func_names: Set[str] = set()
        _fl_verified_class_names: Set[str] = set()
        _fl_verified_entity_lines: Dict[str, int] = {}

        for filepath, state in per_file.items():
            if len(state["source"].splitlines()) <= config.max_file_lines:
                continue

            try:
                fl_result = run_file_limiter(
                    filepath=filepath,
                    original_source=state["original"],
                    post_source=state["source"],
                    diff_ranges=state["ranges"],
                    config=config,
                    verbose=verbose,
                    timing=config.timing,
                )
            except CrispenAPIError:
                raise

            _stats.file_limiter_llm_calls += fl_result.llm_calls
            if fl_result.llm_elapsed > 0 or fl_result.llm_input_tokens > 0:
                _stats.record_llm_call(
                    fl_result.llm_elapsed,
                    fl_result.llm_input_tokens,
                    fl_result.llm_output_tokens,
                    "file_limiter",
                    "file_limiter",
                    filepath,
                )
            _fl_verified_func_names |= fl_result.verified_function_names
            _fl_verified_class_names |= fl_result.verified_class_names
            _fl_verified_entity_lines.update(fl_result.verified_entity_line_counts)

            if fl_result.messages:
                state["msgs"].extend(fl_result.messages)

            if fl_result.abort or not fl_result.new_files:
                continue

            original_dir = Path(filepath).parent
            for rel_path, new_source in fl_result.new_files.items():
                new_path = original_dir / rel_path
                new_path.parent.mkdir(parents=True, exist_ok=True)
                if new_path.parent != original_dir:
                    init_py = new_path.parent / "__init__.py"
                    if not init_py.exists():
                        init_py.write_text("", encoding="utf-8")
                new_path.write_text(new_source, encoding="utf-8")
                _stats.files_edited.append(str(new_path))
                _stats.file_limiter_edits += 1
                _fl_new_file_final[str(new_path)] = new_source
                if (
                    config.file_limiter_recursive
                    and len(new_source.splitlines()) > config.max_file_lines
                ):
                    _fl_recursive.append((str(new_path), new_source))

            state["source"] = fl_result.original_source

            # For non-test whole-file subdir splits (without __main__), delete
            # the original file now that service/__init__.py takes its place as
            # the public entry point.  state["source"] was reset to
            # state["original"] above, so the final write loop will see no diff
            # and skip the (deleted) file.  Count the original lines as deleted
            # so stats stay accurate.
            # When has_main is True the original file is kept on disk as the
            # runnable script entry point; the engine's write loop will update
            # it with the re-export stubs from fl_result.original_source.
            if (
                fl_result.subdir_name is not None
                and not Path(filepath).name.startswith("test_")
                and not fl_result.has_main
            ):
                Path(filepath).unlink()
                _stats.count_lines_changed(state["original"], "")

        # Recursive pass: process any newly-created files that are still over
        # the limit.  Each iteration may enqueue further files; the loop ends
        # when no oversized new files remain.
        _recursive_msgs: List[str] = []
        while _fl_recursive:
            r_path, r_source = _fl_recursive.pop(0)
            n_lines = len(r_source.splitlines())
            try:
                r_result = run_file_limiter(
                    filepath=r_path,
                    original_source="",
                    post_source=r_source,
                    diff_ranges=[(1, n_lines)],
                    config=config,
                    verbose=verbose,
                )
            except CrispenAPIError:
                raise

            _stats.file_limiter_llm_calls += r_result.llm_calls
            if r_result.llm_elapsed > 0 or r_result.llm_input_tokens > 0:
                _stats.record_llm_call(
                    r_result.llm_elapsed,
                    r_result.llm_input_tokens,
                    r_result.llm_output_tokens,
                    "file_limiter",
                    "file_limiter",
                    r_path,
                )
            _fl_verified_func_names |= r_result.verified_function_names
            _fl_verified_class_names |= r_result.verified_class_names
            _fl_verified_entity_lines.update(r_result.verified_entity_line_counts)

            _recursive_msgs.extend(r_result.messages)

            if r_result.abort or not r_result.new_files:
                continue

            r_dir = Path(r_path).parent
            for rel_path, new_source in r_result.new_files.items():
                new_path = r_dir / rel_path
                new_path.parent.mkdir(parents=True, exist_ok=True)
                if new_path.parent != r_dir:
                    init_py = new_path.parent / "__init__.py"
                    if not init_py.exists():
                        init_py.write_text("", encoding="utf-8")
                new_path.write_text(new_source, encoding="utf-8")
                _stats.files_edited.append(str(new_path))
                _stats.file_limiter_edits += 1
                _fl_new_file_final[str(new_path)] = new_source
                if len(new_source.splitlines()) > config.max_file_lines:
                    _fl_recursive.append((str(new_path), new_source))

            # Subdir split of a recursively-processed file: delete the file
            # that was replaced by a package __init__.py.  Handle before the
            # rewrite check so we don't write-then-delete (and double-count lines).
            # Skip deletion when has_main is True (original kept as entry point).
            if (
                r_result.subdir_name is not None
                and not Path(r_path).name.startswith("test_")
                and not r_result.has_main
            ):
                Path(r_path).unlink()
                _fl_new_file_final.pop(str(r_path), None)
            elif r_result.original_source != r_source:
                if r_result.original_source:
                    Path(r_path).write_text(r_result.original_source, encoding="utf-8")
                    _fl_new_file_final[str(r_path)] = r_result.original_source
                elif Path(r_path).name == "__init__.py":
                    # Keep __init__.py even when empty — it defines the package.
                    Path(r_path).write_text("", encoding="utf-8")
                    _fl_new_file_final[str(r_path)] = ""
                else:
                    # Before deleting a test file, redirect any inline imports
                    # in parent or sibling files that point to the old module.
                    if Path(r_path).name.startswith("test_"):
                        _patch_inline_imports_after_test_deletion(
                            r_path,
                            r_dir,
                            r_result.new_files,
                            per_file,
                            _fl_new_file_final,
                        )
                    Path(r_path).unlink()
                    _fl_new_file_final.pop(str(r_path), None)

        for path, content in _fl_new_file_final.items():
            _stats.count_lines_changed("", content)
        _stats.file_limiter_functions_verified = len(_fl_verified_func_names)
        _stats.file_limiter_classes_verified = len(_fl_verified_class_names)
        _stats.file_limiter_lines_verified = sum(_fl_verified_entity_lines.values())
        yield from _recursive_msgs

    # ------------------------------------------------------------------ #
    # Write modified files and yield all messages                         #
    # ------------------------------------------------------------------ #
    for filepath, state in per_file.items():
        if state["source"] != state["original"]:
            if state["source"]:
                Path(filepath).write_text(state["source"], encoding="utf-8")
            elif Path(filepath).name == "__init__.py":
                # Keep __init__.py even when empty — it defines the package.
                Path(filepath).write_text("", encoding="utf-8")
            elif Path(filepath).exists():
                Path(filepath).unlink()
            _stats.files_edited.append(filepath)
            _stats.count_lines_changed(state["original"], state["source"])
        yield from state["msgs"]
