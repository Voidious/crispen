"""Load files, apply refactors, verify, and write back."""

import ast
import os
import threading
import time
from pathlib import Path
from typing import Dict, Generator, List, NamedTuple, Optional, Set, Tuple

from .stats import RunStats

import libcst as cst
from libcst.metadata import FullRepoManager, MetadataWrapper, QualifiedNameProvider

from .config import CrispenConfig, load_config
from .errors import CrispenAPIError
from .file_limiter.runner import run_file_limiter
from .refactors.caller_updater import CallerUpdater
from .refactors.duplicate_extractor import DuplicateExtractor
from .refactors.function_splitter import FunctionSplitter
from .refactors.if_not_else import IfNotElse
from .refactors.tuple_dataclass import TransformInfo, TupleDataclass

# Single-file refactors applied in order before TupleDataclass.
_REFACTORS = [IfNotElse, DuplicateExtractor, FunctionSplitter]

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
                    )
                else:
                    transformer = RefactorClass(
                        ranges, source=current_source, verbose=verbose
                    )
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
                )
            except CrispenAPIError:
                raise

            if fl_result.messages:
                state["msgs"].extend(fl_result.messages)

            if fl_result.abort or not fl_result.new_files:
                continue

            original_dir = Path(filepath).parent
            for rel_path, new_source in fl_result.new_files.items():
                new_path = original_dir / rel_path
                new_path.parent.mkdir(parents=True, exist_ok=True)
                new_path.write_text(new_source, encoding="utf-8")
                _stats.files_edited.append(str(new_path))

            state["source"] = fl_result.original_source

    # ------------------------------------------------------------------ #
    # Write modified files and yield all messages                         #
    # ------------------------------------------------------------------ #
    for filepath, state in per_file.items():
        if state["source"] != state["original"]:
            Path(filepath).write_text(state["source"], encoding="utf-8")
            _stats.files_edited.append(filepath)
            _stats.count_lines_changed(state["original"], state["source"])
        yield from state["msgs"]
