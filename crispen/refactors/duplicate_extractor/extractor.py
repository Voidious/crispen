from __future__ import annotations
from typing import List, Optional, Tuple
import re
import sys
import textwrap
from libcst.metadata import MetadataWrapper
import libcst as cst
from ... import llm_client as _llm_client
from ..base import Refactor
from .collectors import (
    _FunctionCollector,
    _SequenceCollector,
    _build_function_body_fps,
    _collect_called_names,
    _find_duplicate_groups,
    _overlaps_diff,
)
from .editing import (
    _apply_edits,
    _build_helper_insertion,
    _find_insertion_point,
    _skip_class_docstring,
)
from .llm_integration import (
    _generate_no_arg_call,
    _llm_extract,
    _llm_generate_call,
    _llm_verify_extraction,
    _llm_veto,
    _llm_veto_func_match,
)
from .utils import (
    _ApiTimeout,
    _MAX_SEQ_LEN,
    _MIN_WEIGHT,
    _MODEL,
    _run_with_timeout,
    _strip_helper_docstring,
)
from .verification import (
    _collect_called_attr_names,
    _extract_defined_names,
    _find_escaping_vars,
    _has_call_to,
    _has_funcdef,
    _helper_imports_local_name,
    _lift_and_dedup_imports,
    _missing_free_vars,
    _names_in_edit_texts,
    _normalize_replacement_indentation,
    _pyflakes_new_undefined_names,
    _pyflakes_strip_unused_simple_assigns,
    _replacement_contains_return,
    _replacement_steals_post_block_line,
    _scope_end_line,
    _seq_ends_with_return,
    _strip_unused_call_assignments,
    _verify_extraction,
    _would_create_proxy_wrappers,
)


class DuplicateExtractor(Refactor):
    """Detect and extract duplicate code blocks into helper functions via LLM."""

    def __init__(
        self,
        changed_ranges: List[Tuple[int, int]],
        source: str = "",
        verbose: bool = True,
        min_weight: int = _MIN_WEIGHT,
        max_seq_len: int = _MAX_SEQ_LEN,
        model: str = _MODEL,
        helper_docstrings: bool = False,
        provider: str = "anthropic",
        extraction_retries: int = 1,
        llm_verify_retries: int = 1,
        base_url: Optional[str] = None,
        tool_choice: Optional[str] = None,
        api_timeout: float = 60.0,
        match_functions: bool = True,
        timing: str = "detailed",
        current_file: str = "",
        rate_limit_retries: int = 6,
        rate_limit_backoff: float = 20.0,
    ) -> None:
        super().__init__(changed_ranges, source=source, verbose=verbose)
        self.current_file = current_file
        self.timing = timing
        self._min_weight = min_weight
        self._base_max_seq_len = max_seq_len
        self._model = model
        self._helper_docstrings = helper_docstrings
        self._provider = provider
        self._extraction_retries = extraction_retries
        self._llm_verify_retries = llm_verify_retries
        self._base_url = base_url
        self._tool_choice = tool_choice
        self._api_timeout = api_timeout
        self._hard_timeout = api_timeout + 30
        self._match_functions = match_functions
        self._rate_limit_retries = rate_limit_retries
        self._rate_limit_backoff = rate_limit_backoff
        self._new_source: Optional[str] = None
        if source:
            self._analyze(source)

    def _analyze(self, source: str) -> None:
        # 1. Parse tree; early-return on syntax error.
        try:
            tree = cst.parse_module(source)
        except cst.ParserSyntaxError:
            return

        # 2. Source lines.
        source_lines = source.splitlines(keepends=True)

        # 3. Collect functions.
        func_collector = _FunctionCollector(source_lines)
        MetadataWrapper(tree).visit(func_collector)
        all_functions = func_collector.functions

        # 4-5. Build function body fingerprint map (only for called functions).
        called_names = _collect_called_names(source)
        func_body_fps = _build_function_body_fps(all_functions, called_names)

        # 6. Compute max sequence length to capture full function bodies.
        max_seq_len = max(
            max(f.body_stmt_count for f in all_functions) if all_functions else 0,
            self._base_max_seq_len,
        )

        # 7. Collect sequences.
        collector = _SequenceCollector(
            source_lines, max_seq_len=max_seq_len, min_weight=self._min_weight
        )
        MetadataWrapper(tree).visit(collector)

        # 8. Preliminary duplicate groups.
        groups = _find_duplicate_groups(collector.sequences, self.changed_ranges)

        # 9. Check whether any sequence can be replaced with an existing function.
        has_func_matches = (
            self._match_functions
            and func_body_fps
            and any(
                _overlaps_diff(seq, self.changed_ranges)
                and seq.fingerprint in func_body_fps
                and func_body_fps[seq.fingerprint].name != seq.scope
                for seq in collector.sequences
            )
        )

        # 10. Early exit — nothing to do.
        if not has_func_matches and not groups:
            return

        # 12. Create API client.
        api_key = _llm_client.get_api_key(self._provider, caller="DuplicateExtractor")
        client = _llm_client.make_client(
            self._provider, api_key, timeout=self._api_timeout, base_url=self._base_url
        )
        edits: List[Tuple[int, int, str]] = []
        pending_changes: List[str] = []
        # Extraction groups tracked separately so the final combined check can
        # drop any whose call-site edits were silently overridden by overlapping
        # edits from another group or the func-match pass.
        extraction_groups: List[Tuple[str, List[Tuple[int, int, str]], str]] = []
        matched_line_ranges: set = set()

        # 14. Function body match pass.
        if self._match_functions and func_body_fps:
            for seq in collector.sequences:
                if not _overlaps_diff(seq, self.changed_ranges):
                    continue
                if seq.fingerprint not in func_body_fps:
                    continue
                func = func_body_fps[seq.fingerprint]
                if func.name == seq.scope:
                    continue
                if self.verbose:
                    print(
                        f"crispen: DuplicateExtractor: func-match check — "
                        f"scope '{seq.scope}': lines {seq.start_line}-{seq.end_line}"
                        f" → '{func.name}'",
                        file=sys.stderr,
                        flush=True,
                    )
                self.stats.llm_veto_calls += 1
                timing: list = []
                try:
                    is_valid, reason, _veto_notes = _run_with_timeout(
                        _llm_veto_func_match,
                        self._hard_timeout,
                        client,
                        seq,
                        func,
                        source,
                        self._model,
                        self._provider,
                        tool_choice_override=self._tool_choice,
                        _timing_out=timing,
                        rate_limit_retries=self._rate_limit_retries,
                        rate_limit_backoff=self._rate_limit_backoff,
                    )
                    if timing:
                        lr = timing[0]
                        self.stats.record_llm_call(
                            lr.elapsed,
                            lr.input_tokens,
                            lr.output_tokens,
                            "veto",
                            "duplicate_extractor",
                            self.current_file,
                        )
                except _ApiTimeout:
                    print(
                        "crispen: DuplicateExtractor:   → func-match veto timed out",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                if self.verbose:
                    status = "ACCEPTED" if is_valid else "VETOED"
                    timing_suffix = ""
                    if self.timing == "detailed" and timing:
                        lr = timing[0]
                        timing_suffix = (
                            f" [{lr.elapsed:.2f}s,"
                            f" {lr.input_tokens:,} in / {lr.output_tokens:,} out]"
                        )
                    print(
                        f"crispen: DuplicateExtractor:   → {status}: {reason}"
                        f"{timing_suffix}",
                        file=sys.stderr,
                        flush=True,
                    )
                if not is_valid:
                    self.stats.llm_rejected += 1
                    continue
                timing2: list = []
                if func.scope == "<module>" and not func.params:
                    replacement = _generate_no_arg_call(seq, func)
                else:
                    self.stats.llm_edit_calls += 1
                    try:
                        replacement = _run_with_timeout(
                            _llm_generate_call,
                            self._hard_timeout,
                            client,
                            seq,
                            func,
                            source,
                            self._model,
                            self._provider,
                            tool_choice_override=self._tool_choice,
                            _timing_out=timing2,
                            rate_limit_retries=self._rate_limit_retries,
                            rate_limit_backoff=self._rate_limit_backoff,
                        )
                        if timing2:
                            lr = timing2[0]
                            self.stats.record_llm_call(
                                lr.elapsed,
                                lr.input_tokens,
                                lr.output_tokens,
                                "edit",
                                "duplicate_extractor",
                                self.current_file,
                            )
                    except _ApiTimeout:
                        print(
                            "crispen: DuplicateExtractor:"
                            "   → call generation timed out",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    if replacement is None:
                        continue  # pragma: no cover
                if not _verify_extraction(None, [replacement]):
                    continue
                if self.verbose:
                    timing_suffix = ""
                    if self.timing == "detailed" and timing2:
                        lr = timing2[0]
                        timing_suffix = (
                            f" [{lr.elapsed:.2f}s,"
                            f" {lr.input_tokens:,} in / {lr.output_tokens:,} out]"
                        )
                    print(
                        f"crispen: DuplicateExtractor:   → replacing '{seq.scope}'"
                        f" with '{func.name}()'{timing_suffix}",
                        file=sys.stderr,
                        flush=True,
                    )
                edits.append((seq.start_line - 1, seq.end_line, replacement))
                matched_line_ranges.add((seq.start_line, seq.end_line))
                pending_changes.append(
                    f"DuplicateExtractor: replaced '{seq.scope}' body"
                    f" with call to '{func.name}'"
                )

        # 15. Recompute duplicate groups excluding matched sequences.
        if matched_line_ranges:
            remaining = [
                s
                for s in collector.sequences
                if not any(
                    s.start_line <= r_end and s.end_line >= r_start
                    for r_start, r_end in matched_line_ranges
                )
            ]
            groups = _find_duplicate_groups(remaining, self.changed_ranges)

        # 16. Log group count.
        if groups and self.verbose:
            print(
                f"crispen: DuplicateExtractor: found {len(groups)} duplicate group(s)",
                file=sys.stderr,
                flush=True,
            )

        # 17. Duplicate group extraction pass.
        used_names = _extract_defined_names(source)
        for group in groups:
            # Compute escaping vars algorithmically before any LLM call so the
            # extraction prompt can instruct the LLM to return them.
            escaping_vars = frozenset(_find_escaping_vars(group, source_lines))

            # Skip groups that would leave a function as a trivial proxy wrapper
            # (i.e. the extracted block is the function's entire body).
            if _would_create_proxy_wrappers(group, all_functions):
                if self.verbose:
                    print(
                        "crispen: DuplicateExtractor: skipping group — "
                        "extraction would leave a trivial proxy wrapper",
                        file=sys.stderr,
                        flush=True,
                    )
                continue

            if self.verbose:
                ranges_str = ", ".join(
                    f"lines {s.start_line}-{s.end_line}" for s in group
                )
                print(
                    f"crispen: DuplicateExtractor: veto check — "
                    f"scope '{group[0].scope}': {ranges_str}",
                    file=sys.stderr,
                    flush=True,
                )
            self.stats.llm_veto_calls += 1
            timing3: list = []
            try:
                is_valid, reason, veto_notes = _run_with_timeout(
                    _llm_veto,
                    self._hard_timeout,
                    client,
                    group,
                    self._model,
                    self._provider,
                    tool_choice_override=self._tool_choice,
                    _timing_out=timing3,
                    rate_limit_retries=self._rate_limit_retries,
                    rate_limit_backoff=self._rate_limit_backoff,
                )
                if timing3:
                    lr = timing3[0]
                    self.stats.record_llm_call(
                        lr.elapsed,
                        lr.input_tokens,
                        lr.output_tokens,
                        "veto",
                        "duplicate_extractor",
                        self.current_file,
                    )
            except _ApiTimeout:
                print(
                    "crispen: DuplicateExtractor: API call timed out, skipping group",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if self.verbose:
                status = "ACCEPTED" if is_valid else "VETOED"
                timing_suffix = ""
                if self.timing == "detailed" and timing3:
                    lr = timing3[0]
                    timing_suffix = (
                        f" [{lr.elapsed:.2f}s,"
                        f" {lr.input_tokens:,} in / {lr.output_tokens:,} out]"
                    )
                print(
                    f"crispen: DuplicateExtractor:   → {status}: {reason}"
                    f"{timing_suffix}",
                    file=sys.stderr,
                    flush=True,
                )
            if not is_valid:
                self.stats.llm_rejected += 1
                continue

            # Extraction retry loop: attempt extraction up to
            # 1 + _extraction_retries times on algorithmic failure, and up to
            # 1 + _llm_verify_retries additional times on LLM verify failure.
            alg_retries_left = self._extraction_retries
            llm_verify_retries_left = self._llm_verify_retries
            prev_failures: List[str] = []
            prev_output: Optional[dict] = None

            while True:
                self.stats.llm_edit_calls += 1
                timing4: list = []
                try:
                    extraction = _run_with_timeout(
                        _llm_extract,
                        self._hard_timeout,
                        client,
                        group,
                        source,
                        escaping_vars,
                        used_names=frozenset(used_names),
                        model=self._model,
                        helper_docstrings=self._helper_docstrings,
                        provider=self._provider,
                        veto_notes=veto_notes,
                        prev_failures=prev_failures,
                        prev_output=prev_output,
                        tool_choice_override=self._tool_choice,
                        _timing_out=timing4,
                        rate_limit_retries=self._rate_limit_retries,
                        rate_limit_backoff=self._rate_limit_backoff,
                    )
                    if timing4:
                        lr = timing4[0]
                        self.stats.record_llm_call(
                            lr.elapsed,
                            lr.input_tokens,
                            lr.output_tokens,
                            "edit",
                            "duplicate_extractor",
                            self.current_file,
                        )
                except _ApiTimeout:
                    print(
                        "crispen: DuplicateExtractor: API call timed out,"
                        " skipping group",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                if extraction is None:
                    break  # pragma: no cover

                if self.verbose and self.timing == "detailed" and timing4:
                    lr = timing4[0]
                    print(
                        f"crispen: DuplicateExtractor:   → extraction"
                        f" [{lr.elapsed:.2f}s,"
                        f" {lr.input_tokens:,} in / {lr.output_tokens:,} out]",
                        file=sys.stderr,
                        flush=True,
                    )

                helper_source = extraction["helper_source"]
                if not self._helper_docstrings:
                    helper_source = _strip_helper_docstring(helper_source)
                call_replacements = extraction["call_site_replacements"]
                placement = extraction.get("placement", "module_level")
                # Auto-indent 0-indent helpers for staticmethod: placement.
                # The LLM sometimes writes a module-level def even when it
                # selects staticmethod:ClassName.  Inserting 0-indent code
                # inside the class body ends the class silently and makes all
                # subsequent methods nested inside the helper — valid syntax
                # but semantically broken, so compile() does not catch it.
                if placement.startswith("staticmethod:") and helper_source:
                    first_code = next(
                        (ln for ln in helper_source.splitlines() if ln.strip()), ""
                    )
                    if first_code and not first_code[0].isspace():
                        helper_source = textwrap.indent(helper_source, "    ")
                func_name = extraction["function_name"]

                # Helpers are always file-internal; enforce a leading underscore.
                if not func_name.startswith("_"):
                    _old_name = func_name
                    func_name = "_" + func_name
                    _rename_pat = re.compile(r"\b" + re.escape(_old_name) + r"\b")
                    helper_source = _rename_pat.sub(func_name, helper_source)
                    call_replacements = [
                        _rename_pat.sub(func_name, r) for r in call_replacements
                    ]

                _check_failed = False
                _failures: List[str] = []

                # Check 1: name collision
                # Pre-check: placement consistency with call-site class scopes.
                if placement.startswith("staticmethod:"):
                    group_class_scopes = {s.class_scope for s in group}
                    if len(group_class_scopes) != 1 or None in group_class_scopes:
                        _failures.append(
                            "staticmethod placement is invalid when call sites span "
                            "multiple classes or scopes; use module_level instead"
                        )
                        if self.verbose:
                            print(
                                "crispen: DuplicateExtractor: extraction FAILED — "
                                "staticmethod placement invalid for cross-class group",
                                file=sys.stderr,
                                flush=True,
                            )
                        _check_failed = True
                    elif placement.split(":", 1)[1] != next(iter(group_class_scopes)):
                        named_class = placement.split(":", 1)[1]
                        actual_class = next(iter(group_class_scopes))
                        _failures.append(
                            f"staticmethod names class '{named_class}' but all call "
                            f"sites are in '{actual_class}'; use "
                            f"'staticmethod:{actual_class}' instead"
                        )
                        if self.verbose:
                            print(
                                f"crispen: DuplicateExtractor: extraction FAILED — "
                                f"staticmethod names wrong class '{named_class}' "
                                f"(actual: '{actual_class}')",
                                file=sys.stderr,
                                flush=True,
                            )
                        _check_failed = True
                if placement == "module_level":
                    # Reject if any call site invokes the helper as an instance
                    # method (self.<func_name>(...)) — that is inconsistent with
                    # module-level placement and will fail at runtime.
                    _self_call_pat = re.compile(rf"\bself\.{re.escape(func_name)}\s*\(")
                    if any(_self_call_pat.search(r) for r in call_replacements):
                        group_class_scopes = {s.class_scope for s in group}
                        if (
                            len(group_class_scopes) == 1
                            and None not in group_class_scopes
                        ):
                            only_class = next(iter(group_class_scopes))
                            placement_hint = f"use 'staticmethod:{only_class}' instead"
                        else:
                            placement_hint = (
                                f"change call sites to call "
                                f"'{func_name}(...)' directly"
                            )
                        _failures.append(
                            f"module_level placement is inconsistent with call "
                            f"sites that invoke the helper as "
                            f"'self.{func_name}(...)'; {placement_hint}"
                        )
                        if self.verbose:
                            print(
                                f"crispen: DuplicateExtractor: extraction FAILED"
                                f" — module_level placement conflicts with "
                                f"self.{func_name}() call sites",
                                file=sys.stderr,
                                flush=True,
                            )
                        _check_failed = True
                if func_name in used_names:
                    _failures.append(
                        f"name collision: '{func_name}' is already defined,"
                        " choose a different name"
                    )
                    if self.verbose:
                        print(
                            f"crispen: DuplicateExtractor: extraction FAILED — "
                            f"name collision: '{func_name}' is already defined",
                            file=sys.stderr,
                            flush=True,
                        )
                    _check_failed = True

                # Check 2: call site count
                if not _check_failed and len(call_replacements) != len(group):
                    _failures.append(
                        f"wrong call_site_replacements count"
                        f" (expected {len(group)}, got {len(call_replacements)})"
                    )
                    if self.verbose:
                        print(
                            f"crispen: DuplicateExtractor: extraction FAILED — "
                            f"wrong call_site_replacements count "
                            f"(expected {len(group)}, got {len(call_replacements)})",
                            file=sys.stderr,
                            flush=True,
                        )
                        print(
                            f"crispen: DuplicateExtractor:   helper_source: "
                            f"{helper_source!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                        print(
                            f"crispen: DuplicateExtractor:   call_site_replacements: "
                            f"{call_replacements!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                    _check_failed = True

                if not _check_failed:
                    # Normalize each replacement's indentation to match its
                    # original block.  The LLM sometimes returns replacements at
                    # column 0; this re-indents them so the assembled edit is
                    # valid Python.
                    call_replacements = [
                        _normalize_replacement_indentation(seq, r)
                        for seq, r in zip(group, call_replacements)
                    ]

                    # Strip unused variable assignments from call-site
                    # replacements.  The LLM may assign return values that are
                    # never used after the block (e.g. when the helper returns a
                    # value only needed at some call sites), which would produce
                    # flake8 F841 warnings.
                    call_replacements = [
                        _strip_unused_call_assignments(
                            r,
                            source_lines[
                                seq.end_line : _scope_end_line(
                                    source_lines, seq.scope, seq.end_line
                                )
                            ],
                        )
                        for seq, r in zip(group, call_replacements)
                    ]

                    # Check 3: post-block line theft
                    if _replacement_steals_post_block_line(
                        group, call_replacements, source_lines
                    ):
                        _failures.append(
                            "replacement duplicates the line after the block"
                        )
                        if self.verbose:
                            print(
                                "crispen: DuplicateExtractor: extraction FAILED — "
                                "replacement duplicates the line after the block",
                                file=sys.stderr,
                                flush=True,
                            )
                        _check_failed = True

                # Check 4: syntax validation
                if not _check_failed and not _verify_extraction(
                    helper_source, call_replacements
                ):
                    _failures.append("invalid helper or replacement syntax")
                    if self.verbose:
                        print(
                            "crispen: DuplicateExtractor: extraction FAILED — "
                            "invalid helper or replacement syntax",
                            file=sys.stderr,
                            flush=True,
                        )
                        print(
                            f"crispen: DuplicateExtractor:   helper_source: "
                            f"{helper_source!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                        print(
                            f"crispen: DuplicateExtractor:   call_site_replacements: "
                            f"{call_replacements!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                    _check_failed = True

                # Check 5: return statement consistency
                if not _check_failed and any(
                    _seq_ends_with_return(seq)
                    and not _replacement_contains_return(repl)
                    for seq, repl in zip(group, call_replacements)
                ):
                    _failures.append("block ends with return but replacement omits it")
                    if self.verbose:
                        print(
                            "crispen: DuplicateExtractor: extraction FAILED — "
                            "block ends with return but replacement omits it",
                            file=sys.stderr,
                            flush=True,
                        )
                    _check_failed = True

                # Check 6: helper must not import local names
                if not _check_failed and _helper_imports_local_name(
                    helper_source, source
                ):
                    _failures.append(
                        "helper imports a name that is a parameter/local"
                        " in the original file"
                    )
                    if self.verbose:
                        print(
                            "crispen: DuplicateExtractor: extraction FAILED — "
                            "helper imports a name that is a parameter/local "
                            "in the original file",
                            file=sys.stderr,
                            flush=True,
                        )
                    _check_failed = True

                # Check 7: new attribute access
                if not _check_failed:
                    new_attrs = _collect_called_attr_names(
                        textwrap.dedent(helper_source)
                    ) - _collect_called_attr_names(source)
                    if new_attrs:
                        _failures.append(
                            f"helper introduces new attribute access(es) not in"
                            f" original: {', '.join(sorted(new_attrs))}"
                        )
                        if self.verbose:
                            print(
                                f"crispen: DuplicateExtractor: extraction FAILED — "
                                f"helper introduces new attribute access(es) not in"
                                f" original: {', '.join(sorted(new_attrs))}",
                                file=sys.stderr,
                                flush=True,
                            )
                        _check_failed = True

                # Check 8: free variable preservation
                if not _check_failed:
                    seq0 = group[0]
                    block_src = "".join(
                        source_lines[seq0.start_line - 1 : seq0.end_line]
                    )
                    missing = _missing_free_vars(
                        block_src, call_replacements, helper_source, source
                    )
                    if missing:
                        _failures.append(
                            f"free variable(s) from original block missing in"
                            f" replacement: {', '.join(sorted(missing))}"
                        )
                        if self.verbose:
                            print(
                                f"crispen: DuplicateExtractor: extraction FAILED — "
                                f"free variable(s) from original block missing in "
                                f"replacement: {', '.join(sorted(missing))}",
                                file=sys.stderr,
                                flush=True,
                            )
                        _check_failed = True

                # Build this group's edits (only if pre-edit checks passed).
                group_edits: List[Tuple[int, int, str]] = []
                candidate = ""
                if not _check_failed:
                    for seq, replacement in zip(group, call_replacements):
                        group_edits.append(
                            (seq.start_line - 1, seq.end_line, replacement)
                        )
                    first_seq = min(group, key=lambda s: s.start_line)
                    if placement.startswith("staticmethod:"):
                        # Insert inside the class body, after "class Foo:" and
                        # any class docstring (which must remain first).
                        scope = placement.split(":", 1)[1]
                        class_line = _find_insertion_point(source, scope)
                        insert_pos = _skip_class_docstring(source_lines, class_line + 1)
                    else:
                        scope = first_seq.scope
                        insert_pos = _find_insertion_point(source, scope)
                    group_edits.append(
                        _build_helper_insertion(
                            source_lines, insert_pos, helper_source, placement
                        )
                    )
                    # Compile the per-group candidate independently so one bad
                    # extraction doesn't discard valid ones for the same file.
                    candidate = _apply_edits(source, group_edits)

                    # Re-strip unused variable assignments using the assembled
                    # candidate's following lines.  The initial pass (above)
                    # used the original source, which can incorrectly retain an
                    # assignment when another call site's original block
                    # referenced the same name.  Re-running with candidate
                    # following lines also handles partial-tuple targets
                    # (``a, _ = helper()``) the same way the initial pass does.
                    cand_lines = candidate.splitlines(keepends=True)
                    restripped = []
                    for seq, repl in zip(group, call_replacements):
                        cs0 = seq.start_line - 1
                        offset = sum(
                            len(et.splitlines(keepends=True)) - (ee - es)
                            for (es, ee, et) in group_edits
                            if es < cs0
                        )
                        new_end = cs0 + offset + len(repl.splitlines(keepends=True))
                        scope_end = _scope_end_line(cand_lines, seq.scope, new_end)
                        restripped.append(
                            _strip_unused_call_assignments(
                                repl, cand_lines[new_end:scope_end]
                            )
                        )
                    if restripped != call_replacements:
                        call_replacements = restripped
                        group_edits = [
                            (seq.start_line - 1, seq.end_line, r)
                            for seq, r in zip(group, call_replacements)
                        ]
                        group_edits.append(
                            _build_helper_insertion(
                                source_lines, insert_pos, helper_source, placement
                            )
                        )
                        candidate = _apply_edits(source, group_edits)

                    # Check 9: assembled output is valid Python
                    try:
                        compile(candidate, "<rewritten>", "exec")
                    except SyntaxError as exc:
                        _failures.append(f"assembled edit not valid Python: {exc}")
                        if self.verbose:
                            print(
                                f"crispen: DuplicateExtractor: extraction FAILED — "
                                f"assembled edit not valid Python: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                            print(
                                f"crispen: DuplicateExtractor:   helper_source: "
                                f"{helper_source!r}",
                                file=sys.stderr,
                                flush=True,
                            )
                            print(
                                f"crispen: DuplicateExtractor:"
                                f"   call_site_replacements: "
                                f"{call_replacements!r}",
                                file=sys.stderr,
                                flush=True,
                            )
                        _check_failed = True

                    # Check 10: extracted function is actually called
                    if not _check_failed and not _has_call_to(func_name, candidate):
                        _failures.append(
                            f"'{func_name}' not called in candidate output"
                        )
                        if self.verbose:
                            print(
                                f"crispen: DuplicateExtractor: extraction FAILED — "
                                f"'{func_name}' not called in candidate output",
                                file=sys.stderr,
                                flush=True,
                            )
                        _check_failed = True

                    # Check 11: no new undefined names
                    if not _check_failed:
                        undef = _pyflakes_new_undefined_names(source, candidate)
                        if undef:
                            _failures.append(
                                f"undefined name(s) introduced by edit: "
                                f"{', '.join(sorted(undef))}"
                            )
                            if self.verbose:
                                print(
                                    f"crispen: DuplicateExtractor:"
                                    f" extraction FAILED — "
                                    f"undefined name(s) introduced by edit: "
                                    f"{', '.join(sorted(undef))}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            _check_failed = True

                # Retry decision for algorithmic failures
                if _check_failed:
                    if alg_retries_left > 0:
                        alg_retries_left -= 1
                        prev_failures = _failures
                        prev_output = extraction
                        if self.verbose:
                            print(
                                f"crispen: DuplicateExtractor:   → retrying"
                                f" extraction ({alg_retries_left} retries"
                                f" remaining after algorithmic failure)",
                                file=sys.stderr,
                                flush=True,
                            )
                        continue
                    self.stats.algorithmic_rejected += 1
                    break  # exhausted algorithmic retries — skip group

                # ---- LLM verification step ----
                self.stats.llm_verify_calls += 1
                timing5: list = []
                try:
                    verify_ok, verify_issues = _run_with_timeout(
                        _llm_verify_extraction,
                        self._hard_timeout,
                        client,
                        group,
                        helper_source,
                        call_replacements,
                        source,
                        self._model,
                        self._provider,
                        tool_choice_override=self._tool_choice,
                        _timing_out=timing5,
                        rate_limit_retries=self._rate_limit_retries,
                        rate_limit_backoff=self._rate_limit_backoff,
                    )
                    if timing5:
                        lr = timing5[0]
                        self.stats.record_llm_call(
                            lr.elapsed,
                            lr.input_tokens,
                            lr.output_tokens,
                            "verify",
                            "duplicate_extractor",
                            self.current_file,
                        )
                except _ApiTimeout:
                    if self.verbose:
                        print(
                            "crispen: DuplicateExtractor:   → verify timed out,"
                            " accepting extraction",
                            file=sys.stderr,
                            flush=True,
                        )
                    verify_ok, verify_issues = True, []

                if self.verbose:
                    v_status = "ACCEPTED" if verify_ok else "REJECTED"
                    timing_suffix = ""
                    if self.timing == "detailed" and timing5:
                        lr = timing5[0]
                        timing_suffix = (
                            f" [{lr.elapsed:.2f}s,"
                            f" {lr.input_tokens:,} in / {lr.output_tokens:,} out]"
                        )
                    print(
                        f"crispen: DuplicateExtractor:   → verify {v_status}"
                        f"{timing_suffix}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if not verify_ok:
                        for issue in verify_issues:
                            print(
                                f"crispen: DuplicateExtractor:" f"     issue: {issue}",
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
                        if self.verbose:
                            print(
                                f"crispen: DuplicateExtractor:   → retrying"
                                f" extraction after verify rejection"
                                f" ({llm_verify_retries_left} retries remaining)",
                                file=sys.stderr,
                                flush=True,
                            )
                        continue
                    self.stats.llm_rejected += 1
                    break  # exhausted LLM verify retries — skip group

                # ---- All checks passed: accept this extraction ----
                used_names.add(func_name)
                if self.verbose:
                    print(
                        f"crispen: DuplicateExtractor: extracting '{func_name}'",
                        file=sys.stderr,
                        flush=True,
                    )
                extraction_groups.append(
                    (
                        func_name,
                        group_edits,
                        f"DuplicateExtractor: extracted '{func_name}' "
                        f"from {len(group)} duplicate blocks",
                    )
                )
                break  # done with this group

        # 18. Combine all accepted edits, verify all extracted functions are
        # actually called in the combined output, then write.
        all_edits = list(edits)
        for _, g_edits, _ in extraction_groups:
            all_edits.extend(g_edits)

        if all_edits:
            combined = _apply_edits(source, all_edits)

            # Drop any extraction group whose extracted function is not called
            # in the combined output.  This happens when call-site edits are
            # silently skipped by the overlap detector because they conflict
            # with edits from another group or from the func-match pass.
            uncalled = {
                name
                for name, _, _ in extraction_groups
                if not _has_call_to(name, combined)
            }
            if uncalled:
                for name in sorted(uncalled):
                    if self.verbose:
                        print(
                            f"crispen: DuplicateExtractor: extraction DROPPED — "
                            f"'{name}' not called in combined output "
                            f"(call-site edits overridden by overlapping edits)",
                            file=sys.stderr,
                            flush=True,
                        )
                extraction_groups = [
                    (n, g, m) for n, g, m in extraction_groups if n not in uncalled
                ]
                all_edits = list(edits)
                for _, g_edits, _ in extraction_groups:
                    all_edits.extend(g_edits)
                combined = _apply_edits(source, all_edits)

            # Drop any extraction group whose helper function is not defined in
            # the combined output.  This happens when two groups insert helpers
            # before the same scope: _build_helper_insertion absorbs surrounding
            # blank lines into a replacement edit, so the second group's helper
            # insertion is silently skipped by the overlap detector — leaving a
            # call to the helper but no definition.
            undefined_helpers = {
                name
                for name, _, _ in extraction_groups
                if not _has_funcdef(name, combined)
            }
            if undefined_helpers:
                for name in sorted(undefined_helpers):
                    if self.verbose:
                        print(
                            f"crispen: DuplicateExtractor: extraction DROPPED — "
                            f"'{name}' not defined in combined output "
                            f"(helper insertion blocked by overlapping edit)",
                            file=sys.stderr,
                            flush=True,
                        )
                extraction_groups = [
                    (n, g, m)
                    for n, g, m in extraction_groups
                    if n not in undefined_helpers
                ]
                all_edits = list(edits)
                for _, g_edits, _ in extraction_groups:
                    all_edits.extend(g_edits)
                combined = _apply_edits(source, all_edits)

            all_pending = list(pending_changes)
            for _, _, msg in extraction_groups:
                all_pending.append(msg)

            if all_edits:
                _extracted_names = _names_in_edit_texts(extraction_groups)
                combined = _pyflakes_strip_unused_simple_assigns(
                    combined, _extracted_names
                )
                self._new_source = _lift_and_dedup_imports(combined)
                self.changes_made.extend(all_pending)

    def get_rewritten_source(self) -> Optional[str]:
        return self._new_source
