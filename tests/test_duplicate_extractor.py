"""Tests for duplicate_extractor: 100% branch coverage."""

import textwrap


# ---------------------------------------------------------------------------
# _node_weight
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _has_def
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _normalize_source
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _overlaps_diff
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _find_duplicate_groups
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _filter_maximal_groups
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _verify_extraction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _has_mutable_literal_is_check
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _collect_attribute_names
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _collect_called_attr_names
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _has_call_to
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _normalize_replacement_indentation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _has_param_overwritten_before_read
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _pyflakes_new_undefined_names
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _missing_free_vars
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _names_assigned_in
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _extract_defined_names
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _find_escaping_vars
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _apply_edits
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _find_insertion_point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _build_helper_insertion
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _collect_called_names
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _build_function_body_fps
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _SequenceCollector (integration via DuplicateExtractor internals)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _FunctionCollector unit tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — no source
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — no duplicates
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — missing API key
# ---------------------------------------------------------------------------

_DUP_SOURCE = textwrap.dedent(
    """\
    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def bar():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_DUP_RANGES = [(7, 9)]  # overlaps bar's body

# Source where foo's duplicate block assigns z, and foo uses z after the block.
# _has_escaping_vars should detect this and skip the extraction.
_ESC_SOURCE = textwrap.dedent(
    """\
    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        assert z == expected

    def bar():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_ESC_RANGES = [(8, 10)]  # overlaps bar's body


# ---------------------------------------------------------------------------
# DuplicateExtractor — API error in veto
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — parse error in source
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — veto rejects
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — wrong number of call site replacements
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — escaping variables passed to extraction prompt
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — assembled output not valid Python
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — pyflakes new-undefined-names check
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — _missing_free_vars check
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — verification fails
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — replacement steals post-block line
# ---------------------------------------------------------------------------

_POST_STEAL_SOURCE = textwrap.dedent(
    """\
    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        return z

    def bar():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        logger.info("done")
    """
)
_POST_STEAL_RANGES = [(8, 10)]  # overlaps bar's 3-statement block


# ---------------------------------------------------------------------------
# DuplicateExtractor — per-group call check
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — final combined call check
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — two groups, one dropped in combined check (line 1533)
# ---------------------------------------------------------------------------

# Source with two structurally distinct duplicate pairs so _find_duplicate_groups
# returns two separate groups.  The groups differ in argument count so that
# _ASTNormalizer produces different fingerprints for each group:
#   group 1 (foo/bar): 3-stmt bodies using 2-argument calls → fingerprint A
#   group 2 (baz/qux): 3-stmt bodies using 3-argument calls → fingerprint B
_TWO_PAIR_SOURCE = textwrap.dedent(
    """\
    import os

    def foo():
        x = compute(data, config)
        y = transform(x, scale)
        z = finalize(y, mode)

    def bar():
        x = compute(data, config)
        y = transform(x, scale)
        z = finalize(y, mode)

    def baz():
        a = process(item, key, idx)
        b = convert(a, fmt, enc)
        c = export(b, path, opts)

    def qux():
        a = process(item, key, idx)
        b = convert(a, fmt, enc)
        c = export(b, path, opts)
    """
)
_TWO_PAIR_RANGES = [(4, 21)]  # overlaps all duplicate sequences


# ---------------------------------------------------------------------------
# DuplicateExtractor — successful extraction at module level
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — staticmethod placement
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _llm_veto / _llm_extract: loop continues past non-matching content blocks
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# cli integration: CrispenAPIError → sys.exit(1)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _run_with_timeout: hard wall-clock timeout
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _analyze: veto timeout → group skipped
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _analyze: extract timeout → group skipped
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _llm_veto_func_match unit tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _generate_no_arg_call unit tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _llm_generate_call unit tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Function-match integration fixtures
# ---------------------------------------------------------------------------

# _setup() has no params; called by main() → in func_body_fps.
# foo.body fingerprint == _setup.body fingerprint.
# Diff range (2, 9) covers both _setup.body (2-4) AND foo.body (7-9).
# _setup.body hits the func.name==seq.scope True branch (skipped).
# foo.body hits the False branch and proceeds to veto → replace.
_FUNC_MATCH_SOURCE = textwrap.dedent(
    """\
    def _setup():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def main():
        _setup()
    """
)
_FUNC_MATCH_RANGES = [(2, 9)]  # covers _setup.body AND foo.body

# _process(val) has one param; called by main() → in func_body_fps.
# foo.body fingerprint == _process.body fingerprint (names normalized).
# Diff range covers foo.body only.
_FUNC_MATCH_PARAM_SOURCE = textwrap.dedent(
    """\
    def _process(val):
        y = transform(val)
        z = finalize(y)
        return z

    def foo():
        y = transform(data)
        z = finalize(y)
        return z

    def main():
        _process(data)
    """
)
_FUNC_MATCH_PARAM_RANGES = [(6, 9)]  # overlaps foo.body only

# Source with a function-match AND an independent duplicate group.
# bar/baz use an if-else structure so no sub-window of their bodies matches
# _setup's 3-chained-assignment fingerprint.
_FUNC_MATCH_THEN_DUP_SOURCE = textwrap.dedent(
    """\
    def _setup():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def bar():
        if condition:
            result = process(items)
        else:
            result = fallback(items)
        store(result)

    def baz():
        if condition:
            result = process(items)
        else:
            result = fallback(items)
        store(result)

    def main():
        _setup()
    """
)
_FUNC_MATCH_THEN_DUP_RANGES = [(2, 23)]  # covers foo, bar, baz bodies


# ---------------------------------------------------------------------------
# Function-match integration tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# match_functions=False
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor — name collision guard
# ---------------------------------------------------------------------------

# Source that already defines _helper AND has duplicate blocks.
_COLLISION_SOURCE = textwrap.dedent(
    """\
    def _helper(x):
        return x

    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def bar():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_COLLISION_RANGES = [(9, 11)]  # overlaps bar's body


# ---------------------------------------------------------------------------
# _strip_helper_docstring
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _SequenceCollector: min_weight parameter
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor: helper_docstrings config option
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DuplicateExtractor: model config option (passed to API)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _seq_ends_with_return
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _replacement_contains_return
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _replacement_steals_post_block_line
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _helper_imports_local_name
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Integration: block-ends-with-return guard
# ---------------------------------------------------------------------------

_RETURN_BLOCK_SOURCE = textwrap.dedent(
    """\
    def foo():
        x = compute(data)
        y = transform(x)
        return y

    def bar():
        x = compute(data)
        y = transform(x)
        return y
    """
)
_RETURN_BLOCK_RANGES = [(7, 9)]  # overlaps bar's body


# ---------------------------------------------------------------------------
# Integration: helper-imports-local-name guard
# ---------------------------------------------------------------------------

_PARAM_DUP_SOURCE = textwrap.dedent(
    """\
    def test_a(mock_client):
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def test_b(mock_client):
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_PARAM_DUP_RANGES = [(7, 9)]  # overlaps test_b's body


# ---------------------------------------------------------------------------
# New behaviour: veto notes, algorithmic retry, LLM verify step
# ---------------------------------------------------------------------------
