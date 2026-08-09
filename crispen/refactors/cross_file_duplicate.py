"""Cross-file new-duplicate extraction.

Finds a duplicate code block repeated across 2+ files in the current diff,
extracts it into a shared helper placed at the common ancestor package of
every call site (see duplicate_extractor._cross_file_helper_target), and
rewrites every call site to use it.

Unlike the single-file DuplicateExtractor pass (a per-file CSTTransformer),
this edits N call-site files plus one helper file in a single pass — closer
in shape to FileLimiter's multi-file write + compile-validate + stats model
than to the Refactor base class, so it's a standalone engine phase rather
than a Refactor subclass.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Tuple

import libcst as cst
from libcst.metadata import MetadataWrapper

from .. import llm_client as _llm_client
from ..skip_comments import extract_comments, is_skipped
from ..stats import RunStats
from .duplicate_extractor import (
    _VERIFY_CHECKLIST,
    _VERIFY_TOOL,
    _ApiTimeout,
    _SeqInfo,
    _SequenceCollector,
    _apply_edits,
    _cross_file_helper_target,
    _extract_defined_names,
    _find_cross_file_duplicate_groups,
    _find_escaping_vars,
    _first_funcdef_idx,
    _group_ends_in_return,
    _has_call_to,
    _lift_and_dedup_imports,
    _llm_veto,
    _pyflakes_new_undefined_names,
    _pyflakes_strip_newly_unused_imports,
    _run_with_timeout,
    _strip_helper_docstring,
    _verify_extraction,
)

if TYPE_CHECKING:
    from ..config import CrispenConfig  # pragma: no cover

_CROSS_FILE_EXTRACT_TOOL: dict = {
    "name": "extract_cross_file_helper",
    "description": (
        "Extract duplicate code blocks found in different files into a "
        "single shared helper function"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "function_name": {
                "type": "string",
                "description": (
                    "A public, importable name (no leading underscore) — "
                    "this helper is called from other files."
                ),
            },
            "helper_source": {
                "type": "string",
                "description": "Complete source of the helper function",
            },
            "call_site_replacements": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Replacement source for each duplicate block, "
                    "in the same order as the input blocks. "
                    "Each replacement must preserve the original block's "
                    "leading indentation and end with a trailing newline. "
                    "Cover only the exact lines of the specified block — "
                    "do not include any code from before or after the block."
                ),
            },
        },
        "required": ["function_name", "helper_source", "call_site_replacements"],
    },
}


def _llm_extract_cross_file(
    client,
    group: List[_SeqInfo],
    file_sources: Dict[str, str],
    module_name: str,
    escaping_vars: frozenset = frozenset(),
    used_names: frozenset = frozenset(),
    model: str = "",
    helper_docstrings: bool = True,
    provider: str = "anthropic",
    veto_notes: str = "",
    prev_failures: Optional[List[str]] = None,
    prev_output: Optional[dict] = None,
    tool_choice_override: Optional[str] = None,
    _timing_out=None,
    rate_limit_retries: int = 6,
    rate_limit_backoff: float = 20.0,
) -> Optional[dict]:
    """Like duplicate_extractor._llm_extract, but for a group whose
    occurrences span multiple files.

    Placement is already decided mechanically (module_name, computed by
    _cross_file_helper_target) — the LLM only generates the helper body and
    call-site replacements, never a placement choice, and never an import
    statement (the import is added mechanically afterward, same as the
    repo-wide match-function pass in 0.8.0-a).
    """
    prev_failures = prev_failures or []
    block_entries = []
    for i, s in enumerate(group):
        src_lines = file_sources.get(s.filepath, "").splitlines(keepends=True)
        entry = (
            f"Block {i + 1} (file: {s.filepath}, scope: {s.scope}, "
            f"lines {s.start_line}-{s.end_line}):\n"
            f"```python\n{s.source.rstrip()}\n```"
        )
        next_idx = s.end_line
        if next_idx < len(src_lines):
            next_line = src_lines[next_idx].rstrip()
            if next_line.strip():
                entry += (
                    f"\nLine immediately after this block"
                    f" (must NOT appear in the replacement): `{next_line}`"
                )
        block_entries.append(entry)
    blocks_text = "\n\n".join(block_entries)
    escaping_note = ""
    if escaping_vars:
        vars_str = ", ".join(sorted(escaping_vars))
        escaping_note = (
            f"\n\nThe following variables are assigned within the duplicate block "
            f"and referenced by code that immediately follows the block at one or "
            f"more call sites: {vars_str}. The helper function must return these "
            f"variables. Every call site replacement that needs the returned "
            f"value(s) MUST begin with the capturing assignment or `return` — "
            f"check this individually for each call site, including ones later "
            f"in the list, not just the first. At call sites where the return "
            f"value is not needed, discard it."
        )
    return_note = ""
    if _group_ends_in_return(group):
        return_note = (
            "\n\nEach duplicate block's own last statement is `return <expr>` "
            "— the call site replacement for every occurrence must be "
            "`return <helper_call>(...)`, never a bare call that silently "
            "drops the return value."
        )
    used_names_note = ""
    if used_names:
        names_str = ", ".join(sorted(used_names))
        used_names_note = (
            f"\n\nThe following names are already defined in one of the involved "
            f"files or reserved by a previous extraction: {names_str}. "
            f"Do not use any of these names for the helper function."
        )
    docstring_note = (
        ""
        if helper_docstrings
        else "\n\nDo not include a docstring in the helper function."
    )
    veto_notes_note = ""
    if veto_notes:
        veto_notes_note = (
            f"\n\nNotes from code review (watch out for these pitfalls): "
            f"{veto_notes[:500]}"
        )
    failures_note = ""
    if prev_failures:
        failures_str = "\n".join(f"- {f}" for f in prev_failures)
        prior_helper = (prev_output or {}).get("helper_source", "")
        prior_repls = (prev_output or {}).get("call_site_replacements", [])
        repls_text = "\n".join(f"  [{i + 1}] {r!r}" for i, r in enumerate(prior_repls))
        failures_note = (
            f"\n\nThe previous extraction attempt produced:\n\n"
            f"helper_source:\n```python\n{prior_helper}```\n\n"
            f"call_site_replacements:\n{repls_text}\n\n"
            f"But failed these checks:\n{failures_str}\n\n"
            f"Before responding, check EVERY call site replacement individually "
            f"against these issues — a fix that only corrects the first call "
            f"site and leaves a later one with the same mistake will fail again."
        )
    prompt = (
        "Extract the following duplicate code blocks — found in different files — "
        f"into a single shared helper function.\n\nDuplicate blocks:\n{blocks_text}\n\n"
        f"The helper will be placed in the module `{module_name}` and imported into "
        "every call site automatically — do not add an import statement yourself, "
        "and do not choose a placement; just generate the helper itself as a plain "
        "module-level function with a public name (no leading underscore). "
        "Return complete, valid Python for the helper and each call site replacement. "
        "Each call site replacement must start with the same leading indentation as "
        "the block it replaces, end with a trailing newline, and cover only the exact "
        "lines of the duplicate block — stopping before the 'Line immediately after "
        "this block' marker shown above. Do not include any code from before or after "
        "the block. "
        "Double-check that only required parameters are passed to the helper — do not "
        "include an unused parameter, or one that is overwritten before being read. "
        "Be mindful of the code being removed from each call site: if variable "
        "assignments are moved into the helper, those variables may no longer be "
        "defined in the calling scope at that point. "
        "If the helper uses a sentinel return value to signal an error path (such as "
        "returning an empty collection), check for it at the call site with `==`, not "
        "`is` — `is` only gives correct results for singletons like `None`, `True`, "
        "and `False`, not for constructed objects like `set()`."
        f"{escaping_note}"
        f"{return_note}"
        f"{used_names_note}"
        f"{docstring_note}"
        f"{veto_notes_note}"
        f"{failures_note}"
    )
    result = _llm_client.call_with_tool(
        client,
        provider,
        model,
        5000,
        _CROSS_FILE_EXTRACT_TOOL,
        "extract_cross_file_helper",
        [{"role": "user", "content": prompt}],
        caller="DuplicateExtractor",
        tool_choice_override=tool_choice_override,
        rate_limit_retries=rate_limit_retries,
        rate_limit_backoff=rate_limit_backoff,
    )
    if _timing_out is not None:
        _timing_out.append(result)
    return result.tool_input


def _llm_verify_extraction_cross_file(
    client,
    group: List[_SeqInfo],
    helper_source: str,
    call_replacements: List[str],
    file_sources: Dict[str, str],
    model: str = "",
    provider: str = "anthropic",
    tool_choice_override: Optional[str] = None,
    _timing_out=None,
    rate_limit_retries: int = 6,
    rate_limit_backoff: float = 20.0,
) -> Tuple[bool, List[str]]:
    """Like duplicate_extractor._llm_verify_extraction, but builds one context
    window per file instead of a single window across all blocks — a single
    window wouldn't make sense when blocks come from files with unrelated
    line numbers."""
    blocks_text = "\n\n".join(
        f"Original block {i + 1} (file: {s.filepath}, scope: {s.scope}, "
        f"lines {s.start_line}-{s.end_line}):\n"
        f"```python\n{s.source.rstrip()}\n```"
        for i, s in enumerate(group)
    )
    replacements_text = "\n\n".join(
        f"Replacement for block {i + 1}:\n```python\n{r.rstrip()}\n```"
        for i, r in enumerate(call_replacements)
    )
    context_parts = []
    for filepath in dict.fromkeys(s.filepath for s in group):
        file_seqs = [s for s in group if s.filepath == filepath]
        src_lines = file_sources.get(filepath, "").splitlines(keepends=True)
        min_start = min(s.start_line for s in file_seqs)
        max_end = max(s.end_line for s in file_seqs)
        window_start = max(0, min_start - 30)
        window_end = min(len(src_lines), max_end + 100)
        snippet = "".join(src_lines[window_start:window_end])
        context_parts.append(
            f"{filepath} (lines {window_start + 1}–{window_end}):\n"
            f"```python\n{snippet}\n```"
        )
    context_text = "\n\n".join(context_parts)
    prompt = (
        "Verify that the following helper function extraction is semantically "
        "correct by tracing through the code carefully.\n\n"
        f"Original duplicate blocks:\n{blocks_text}\n\n"
        f"Extracted helper:\n```python\n{helper_source.rstrip()}\n```\n\n"
        f"Call site replacements:\n{replacements_text}\n\n"
        f"Source context around each duplicate block:\n{context_text}\n\n"
        f"{_VERIFY_CHECKLIST}"
    )
    result = _llm_client.call_with_tool(
        client,
        provider,
        model,
        4096,
        _VERIFY_TOOL,
        "verify_extraction",
        [{"role": "user", "content": prompt}],
        caller="DuplicateExtractor",
        tool_choice_override=tool_choice_override,
        rate_limit_retries=rate_limit_retries,
        rate_limit_backoff=rate_limit_backoff,
    )
    if _timing_out is not None:
        _timing_out.append(result)
    if result.tool_input is None:
        return False, [
            "Verification response was truncated or empty — treating as unverified."
        ]
    return result.tool_input["is_correct"], result.tool_input.get("issues", [])


def run_cross_file_duplicate_extraction(
    per_file: Dict[str, dict],
    repo_root: Optional[str],
    config: "CrispenConfig",
    verbose: bool = True,
    stats: Optional[RunStats] = None,
) -> Generator[str, None, None]:
    """Find and extract duplicate blocks that span 2+ files in *per_file*.

    Reads/writes ``per_file[filepath]["source"]`` in place for call-site
    edits (picked up by run_engine's normal final write-back loop, same as
    every other phase); writes the new helper file directly to disk (it has
    no entry in *per_file*, matching how FileLimiter writes its own new
    files directly).
    """
    _stats = stats if stats is not None else RunStats()
    if repo_root is None or len(per_file) < 2:
        return

    all_sequences: List[_SeqInfo] = []
    changed_ranges_by_file: Dict[str, List[Tuple[int, int]]] = {}
    for filepath, state in per_file.items():
        source = state["source"]
        try:
            tree = cst.parse_module(source)
        except cst.ParserSyntaxError:
            continue
        source_lines = source.splitlines(keepends=True)
        collector = _SequenceCollector(
            source_lines,
            max_seq_len=config.max_duplicate_seq_len,
            min_weight=config.min_duplicate_weight,
        )
        MetadataWrapper(tree).visit(collector)
        comments = extract_comments(source)
        for seq in collector.sequences:
            if is_skipped(
                seq.start_line, "duplicate_extractor", source_lines, comments
            ):
                continue
            seq.filepath = filepath
            all_sequences.append(seq)
        changed_ranges_by_file[filepath] = state["ranges"]

    groups = _find_cross_file_duplicate_groups(all_sequences, changed_ranges_by_file)
    if not groups:
        return

    if verbose:
        print(
            f"crispen: DuplicateExtractor: found {len(groups)} cross-file "
            "duplicate group(s)",
            file=sys.stderr,
            flush=True,
        )

    api_key = _llm_client.get_api_key(config.provider, caller="DuplicateExtractor")
    client = _llm_client.make_client(
        config.provider, api_key, timeout=config.api_timeout, base_url=config.base_url
    )
    hard_timeout = config.api_timeout + 30

    for group in groups:
        file_paths = sorted({s.filepath for s in group})
        file_sources = {fp: per_file[fp]["source"] for fp in file_paths}
        file_lines = {
            fp: src.splitlines(keepends=True) for fp, src in file_sources.items()
        }

        escaping_vars: set = set()
        used_names: set = set()
        for fp in file_paths:
            fp_seqs = [s for s in group if s.filepath == fp]
            escaping_vars |= _find_escaping_vars(fp_seqs, file_lines[fp])
            used_names |= _extract_defined_names(file_sources[fp])
        escaping_vars_fs = frozenset(escaping_vars)

        target_file, dotted_module = _cross_file_helper_target(
            file_paths, repo_root, config.cross_file_helper_module
        )
        helper_file_existed = target_file.exists()
        existing_helper_content = (
            target_file.read_text(encoding="utf-8") if helper_file_existed else ""
        )
        if helper_file_existed:
            used_names |= _extract_defined_names(existing_helper_content)

        if verbose:
            ranges_str = ", ".join(
                f"{s.filepath}:{s.start_line}-{s.end_line}" for s in group
            )
            print(
                f"crispen: DuplicateExtractor: cross-file veto check — {ranges_str}",
                file=sys.stderr,
                flush=True,
            )
        _stats.llm_veto_calls += 1
        timing: list = []
        try:
            is_valid, reason, veto_notes = _run_with_timeout(
                _llm_veto,
                hard_timeout,
                client,
                group,
                config.model,
                config.provider,
                tool_choice_override=config.tool_choice,
                _timing_out=timing,
                rate_limit_retries=config.rate_limit_retries,
                rate_limit_backoff=config.rate_limit_backoff,
            )
            if timing:
                lr = timing[0]
                _stats.record_llm_call(
                    lr.elapsed,
                    lr.input_tokens,
                    lr.output_tokens,
                    "veto",
                    "duplicate_extractor",
                    file_paths[0],
                )
        except _ApiTimeout:
            print(
                "crispen: DuplicateExtractor: cross-file API call timed out, "
                "skipping group",
                file=sys.stderr,
                flush=True,
            )
            continue
        if verbose:
            status = "ACCEPTED" if is_valid else "VETOED"
            print(
                f"crispen: DuplicateExtractor:   → {status}: {reason}",
                file=sys.stderr,
                flush=True,
            )
        if not is_valid:
            _stats.llm_rejected += 1
            continue

        alg_retries_left = config.extraction_retries
        llm_verify_retries_left = config.llm_verify_retries
        prev_failures: List[str] = []
        prev_output: Optional[dict] = None
        accepted = False
        accepted_helper_source = ""
        accepted_func_name = ""
        accepted_trial_sources: Dict[str, str] = {}

        while True:
            _stats.llm_edit_calls += 1
            timing2: list = []
            try:
                extraction = _run_with_timeout(
                    _llm_extract_cross_file,
                    hard_timeout,
                    client,
                    group,
                    file_sources,
                    dotted_module,
                    escaping_vars_fs,
                    used_names=frozenset(used_names),
                    model=config.model,
                    helper_docstrings=config.helper_docstrings,
                    provider=config.provider,
                    veto_notes=veto_notes,
                    prev_failures=prev_failures,
                    prev_output=prev_output,
                    tool_choice_override=config.tool_choice,
                    _timing_out=timing2,
                    rate_limit_retries=config.rate_limit_retries,
                    rate_limit_backoff=config.rate_limit_backoff,
                )
                if timing2:
                    lr = timing2[0]
                    _stats.record_llm_call(
                        lr.elapsed,
                        lr.input_tokens,
                        lr.output_tokens,
                        "edit",
                        "duplicate_extractor",
                        file_paths[0],
                    )
            except _ApiTimeout:
                print(
                    "crispen: DuplicateExtractor: cross-file API call timed out, "
                    "skipping group",
                    file=sys.stderr,
                    flush=True,
                )
                break
            if extraction is None:
                break  # pragma: no cover

            helper_source = extraction["helper_source"]
            if not config.helper_docstrings:
                helper_source = _strip_helper_docstring(helper_source)
            call_replacements = extraction["call_site_replacements"]
            func_name = extraction["function_name"]

            failures: List[str] = []
            trial_sources: Dict[str, str] = {}

            if len(call_replacements) != len(group):
                failures.append(
                    "call_site_replacements must have exactly one entry per "
                    "duplicate block, in the same order as the input blocks"
                )
            elif not _verify_extraction(helper_source, call_replacements):
                failures.append(
                    "helper_source or a call_site_replacement is not valid "
                    "Python, a parameter is overwritten before being read, or "
                    "contains a mutable-literal identity check (`is set()` "
                    "etc., which is always False)"
                )
            else:
                edits_by_file: Dict[str, List[Tuple[int, int, str]]] = {}
                for seq, repl in zip(group, call_replacements):
                    edits_by_file.setdefault(seq.filepath, []).append(
                        (seq.start_line - 1, seq.end_line, repl)
                    )
                import_line = f"from {dotted_module} import {func_name}\n\n\n"
                for fp in file_paths:
                    src = file_sources[fp]
                    insert_idx = _first_funcdef_idx(src.splitlines(keepends=True))
                    fp_edits = edits_by_file.get(fp, []) + [
                        (insert_idx, insert_idx, import_line)
                    ]
                    combined = _lift_and_dedup_imports(_apply_edits(src, fp_edits))
                    combined = _pyflakes_strip_newly_unused_imports(src, combined)
                    try:
                        compile(combined, fp, "exec")
                    except SyntaxError:
                        failures.append(f"{fp}: result is not valid Python")
                        break
                    if not _has_call_to(func_name, combined):
                        failures.append(
                            f"{fp}: '{func_name}' is not called anywhere in the result"
                        )
                        break
                    undef = _pyflakes_new_undefined_names(src, combined)
                    if undef:
                        failures.append(
                            f"{fp}: undefined name(s) introduced by the edit: "
                            f"{', '.join(sorted(undef))}"
                        )
                        break
                    trial_sources[fp] = combined

                # The per-file checks above only look at the call-site files.
                # The helper itself also has to stand alone in its own new
                # file — e.g. it must not reference a private module-level
                # import or constant (like the original file's own
                # `_llm_client` or a `_SOME_TOOL` dict) that has no import
                # path from the target module. compile() alone can't catch
                # this: a bare `def f(): ...body...` referencing an undefined
                # name is syntactically valid, it only fails at *call* time.
                if not failures:
                    trial_helper_content = (
                        (
                            existing_helper_content.rstrip("\n") + "\n\n\n"
                            if existing_helper_content
                            else ""
                        )
                        + helper_source.rstrip("\n")
                        + "\n"
                    )
                    helper_undef = _pyflakes_new_undefined_names(
                        existing_helper_content, trial_helper_content
                    )
                    if helper_undef:
                        failures.append(
                            "helper module: undefined name(s) — the extracted "
                            "helper can't stand alone in its own file (it "
                            "likely references a private import or constant "
                            "from the original file): "
                            f"{', '.join(sorted(helper_undef))}"
                        )

            if failures:
                if alg_retries_left > 0:
                    alg_retries_left -= 1
                    prev_failures = failures
                    prev_output = extraction
                    if verbose:
                        print(
                            f"crispen: DuplicateExtractor:   → retrying"
                            f" extraction ({alg_retries_left} retries remaining"
                            f" after algorithmic failure)",
                            file=sys.stderr,
                            flush=True,
                        )
                    continue
                _stats.algorithmic_rejected += 1
                break

            _stats.llm_verify_calls += 1
            timing3: list = []
            try:
                verify_ok, verify_issues = _run_with_timeout(
                    _llm_verify_extraction_cross_file,
                    hard_timeout,
                    client,
                    group,
                    helper_source,
                    call_replacements,
                    file_sources,
                    config.model,
                    config.provider,
                    tool_choice_override=config.tool_choice,
                    _timing_out=timing3,
                    rate_limit_retries=config.rate_limit_retries,
                    rate_limit_backoff=config.rate_limit_backoff,
                )
                if timing3:
                    lr = timing3[0]
                    _stats.record_llm_call(
                        lr.elapsed,
                        lr.input_tokens,
                        lr.output_tokens,
                        "verify",
                        "duplicate_extractor",
                        file_paths[0],
                    )
            except _ApiTimeout:
                if verbose:
                    print(
                        "crispen: DuplicateExtractor:   → verify timed out,"
                        " accepting extraction",
                        file=sys.stderr,
                        flush=True,
                    )
                verify_ok, verify_issues = True, []

            if verbose:
                v_status = "ACCEPTED" if verify_ok else "REJECTED"
                print(
                    f"crispen: DuplicateExtractor:   → verify {v_status}",
                    file=sys.stderr,
                    flush=True,
                )
                if not verify_ok:
                    for issue in verify_issues:
                        print(
                            f"crispen: DuplicateExtractor:     issue: {issue}",
                            file=sys.stderr,
                            flush=True,
                        )

            if not verify_ok:
                if llm_verify_retries_left > 0:
                    llm_verify_retries_left -= 1
                    prev_failures = [
                        f"LLM verification issue: {i}" for i in verify_issues
                    ]
                    prev_output = extraction
                    if verbose:
                        print(
                            f"crispen: DuplicateExtractor:   → retrying extraction"
                            f" after verify rejection ({llm_verify_retries_left}"
                            f" retries remaining)",
                            file=sys.stderr,
                            flush=True,
                        )
                    continue
                _stats.llm_rejected += 1
                break

            accepted = True
            accepted_helper_source = helper_source
            accepted_func_name = func_name
            accepted_trial_sources = trial_sources
            break

        if not accepted:
            continue

        if existing_helper_content:
            helper_file_content = (
                existing_helper_content.rstrip("\n")
                + "\n\n\n"
                + accepted_helper_source.rstrip("\n")
                + "\n"
            )
        else:
            helper_file_content = accepted_helper_source.rstrip("\n") + "\n"
        try:
            compile(helper_file_content, str(target_file), "exec")
        except SyntaxError:
            if verbose:
                print(
                    "crispen: DuplicateExtractor: cross-file extraction FAILED — "
                    "helper file would not be valid Python after merging with "
                    "existing content, skipping group",
                    file=sys.stderr,
                    flush=True,
                )
            continue

        target_file.parent.mkdir(parents=True, exist_ok=True)
        init_py = target_file.parent / "__init__.py"
        if not init_py.exists():
            init_py.write_text("", encoding="utf-8")
        target_file.write_text(helper_file_content, encoding="utf-8")
        _stats.files_edited.append(str(target_file))
        _stats.count_lines_changed(existing_helper_content, helper_file_content)
        _stats.duplicate_extracted += 1

        for fp, combined in accepted_trial_sources.items():
            per_file[fp]["source"] = combined

        if verbose:
            print(
                f"crispen: DuplicateExtractor: extracted '{accepted_func_name}'",
                file=sys.stderr,
                flush=True,
            )
        yield (
            f"DuplicateExtractor: extracted '{accepted_func_name}' from "
            f"{len(group)} duplicate blocks across {len(file_paths)} files "
            f"into {dotted_module}"
        )
