"""Load files, apply refactors, verify, and write back."""

import sys
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple
from ..stats import RunStats
import libcst as cst
from libcst.metadata import MetadataWrapper
from ..config import CrispenConfig, format_header, load_config
from ..errors import CrispenAPIError
from ..file_limiter.runner import run_file_limiter
from ..patch_rewriter import (
    _FLContext,
    RewriteAccumulator,
    apply_patch_callgraph,
    apply_patch_rewrite,
)
from ..patch_updater import apply_patch_strings
from ..refactors.caller_updater import CallerUpdater
from ..refactors.duplicate_extractor import DuplicateExtractor
from ..refactors.function_splitter import FunctionSplitter
from ..refactors.tuple_dataclass import TransformInfo
from .callers import _blocked_private_scopes  # fmt: skip # noqa: F401, E501
from .callers import _build_alias_map  # fmt: skip # noqa: F401, E501
from .callers import _compute_qname  # fmt: skip # noqa: F401, E501
from .callers import _file_to_module  # fmt: skip # noqa: F401, E501
from .callers import _find_outside_callers  # fmt: skip # noqa: F401, E501
from .callers import _find_repo_root  # fmt: skip # noqa: F401, E501
from .callers import _has_callers_outside_ranges  # fmt: skip # noqa: F401, E501
from .callers import _visit_with_timeout  # fmt: skip # noqa: F401, E501
from .core import _REFACTORS, _REFACTOR_KEY
from .core import _EXCLUDED_DIR_NAMES  # fmt: skip # noqa: F401, E501
from .core import _LLM_REFACTOR_KEYS  # fmt: skip # noqa: F401, E501
from .core import _apply_tuple_dataclass  # fmt: skip # noqa: F401, E501
from .core import _categorize_into_stats  # fmt: skip # noqa: F401, E501
from .core import _should_run  # fmt: skip # noqa: F401, E501
from .filelimiter import _add_fl_context  # fmt: skip # noqa: F401, E501
from .filelimiter import _build_patch_map  # fmt: skip # noqa: F401, E501
from .filelimiter import _collect_assignment_names  # fmt: skip # noqa: F401, E501
from .filelimiter import _collect_code_referenced_names  # fmt: skip # noqa: F401, E501
from .filelimiter import _collect_imported_names  # fmt: skip # noqa: F401, E501
from .filelimiter import _collect_top_level_names  # fmt: skip # noqa: F401, E501
from .filelimiter import _module_path_for_file  # fmt: skip # noqa: F401, E501
from .filelimiter import _patch_inline_imports_after_test_deletion  # fmt: skip # noqa: F401, E501
from .filelimiter import _redirect_inline_module_imports  # fmt: skip # noqa: F401, E501


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
                        rate_limit_retries=config.rate_limit_retries,
                        rate_limit_backoff=config.rate_limit_backoff,
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
                        rate_limit_retries=config.rate_limit_retries,
                        rate_limit_backoff=config.rate_limit_backoff,
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
    combined_patch_map: Dict[str, str] = {}
    _fl_all_contexts: List[_FLContext] = []
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

            pre_split_src = state["source"]
            state["source"] = fl_result.original_source

            if fl_result.entity_to_target:
                combined_patch_map.update(
                    _build_patch_map(
                        filepath, fl_result, Path(filepath).parent, pre_split_src
                    )
                )
                if config.file_limiter_patch_update in ("basic", "rewrite"):
                    _add_fl_context(
                        _fl_all_contexts,
                        filepath,
                        pre_split_src,
                        fl_result,
                        combined_patch_map,
                    )

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

            if r_result.entity_to_target and not r_result.abort:
                combined_patch_map.update(
                    _build_patch_map(r_path, r_result, Path(r_path).parent, r_source)
                )
                if config.file_limiter_patch_update in ("basic", "rewrite"):
                    _add_fl_context(
                        _fl_all_contexts,
                        r_path,
                        r_source,
                        r_result,
                        combined_patch_map,
                    )

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

    # Flatten transitive chains in combined_patch_map.  When recursive splits
    # run, round 1 may produce A→B and round 2 may produce B→C.  Without
    # flattening, apply_patch_strings (single-pass) would leave consumers of
    # A pointing at the intermediate path B instead of the final path C.
    if combined_patch_map:
        changed = True
        while changed:
            changed = False
            for k in list(combined_patch_map):
                v = combined_patch_map[k]
                if v in combined_patch_map and combined_patch_map[v] != v:
                    combined_patch_map[k] = combined_patch_map[v]
                    changed = True

    # ------------------------------------------------------------------ #
    # Phase 4 — Update @patch strings after FileLimiter entity moves     #
    # ------------------------------------------------------------------ #
    _patch_acc = RewriteAccumulator()
    if (
        config.file_limiter_patch_update in ("basic", "rewrite")
        and combined_patch_map
        and repo_root
    ):
        _stats.patch_single_candidate += len(combined_patch_map)
        # Update per_file sources still in memory (not yet written to disk).
        for filepath, state in per_file.items():
            new_src = apply_patch_strings(state["source"], combined_patch_map)
            if new_src != state["source"]:
                state["source"] = new_src
                state["msgs"].append(
                    f"{filepath}: patch_update: updated @patch strings"
                )
                _stats.patch_update_edits += 1
        # Scan every other *.py file in the repo and update on disk.
        per_file_abs = {str(Path(f).resolve()) for f in per_file}
        repo_root_path = Path(repo_root)
        for py_file in sorted(repo_root_path.rglob("*.py")):
            if str(py_file.resolve()) in per_file_abs:
                continue
            if any(
                part in _EXCLUDED_DIR_NAMES
                for part in py_file.relative_to(repo_root_path).parts[:-1]
            ):
                continue
            try:
                src = py_file.read_text(encoding="utf-8")
            except OSError:
                continue
            new_src = apply_patch_strings(src, combined_patch_map)
            if new_src != src:
                py_file.write_text(new_src, encoding="utf-8")
                _stats.patch_update_edits += 1
                yield f"{py_file}: patch_update: updated @patch strings"

    _cg_candidates: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    if config.file_limiter_patch_update in ("basic", "rewrite") and _fl_all_contexts:
        for _cg_msg in apply_patch_callgraph(
            _fl_all_contexts,
            per_file,
            repo_root,
            verbose=verbose,
            candidates_out=_cg_candidates,
            config=config,
            _acc=_patch_acc,
        ):
            _stats.patch_update_edits += 1
            yield _cg_msg

    if config.file_limiter_patch_update == "rewrite" and _fl_all_contexts:
        yield from apply_patch_rewrite(
            _fl_all_contexts,
            per_file,
            repo_root,
            config,
            verbose=verbose,
            _acc=_patch_acc,
            cg_candidates=_cg_candidates or None,
        )
        _stats.patch_rewrite_llm_calls += _patch_acc.calls
        if _patch_acc.elapsed > 0 or _patch_acc.input_tokens > 0:
            _stats.record_llm_call(
                _patch_acc.elapsed,
                _patch_acc.input_tokens,
                _patch_acc.output_tokens,
                "file_limiter",
                "patch_rewriter",
                "",
            )
        _stats.patch_update_edits += _patch_acc.files_updated

    _stats.patch_cg_resolved += _patch_acc.cg_resolved
    _stats.patch_llm_no_change += _patch_acc.no_change
    _stats.patch_llm_rename += _patch_acc.rename
    _stats.patch_llm_rewrite += _patch_acc.rewrite
    _stats.patch_edit_failures += _patch_acc.edit_failures

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
