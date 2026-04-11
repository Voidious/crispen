from unittest.mock import MagicMock, patch
import textwrap
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _SeqInfo,
    _pyflakes_strip_unused_simple_assigns,
    _replace_unused_in_target,
    _replacement_contains_return,
    _replacement_steals_post_block_line,
    _seq_ends_with_return,
    _seq_source_contains_yield,
    _strip_helper_docstring,
    _strip_unused_call_assignments,
)
from .test_extractor_core import _make_extract_response, _make_veto_response
from .test_helper_building import _make_seq_with_source
from .test_collector_integration import _collect_sequences


def test_pyflakes_strip_unused_simple_assigns_removes_literal_init():
    # last_import_line = 0 becomes unused after extraction.
    source = textwrap.dedent(
        """\
        def foo(source):
            last_import_line = 0
            lines = source.splitlines()
            return lines
    """
    )
    result = _pyflakes_strip_unused_simple_assigns(source, {"last_import_line"})
    assert "last_import_line" not in result
    assert "lines = source.splitlines()" in result


def test_pyflakes_strip_unused_simple_assigns_keeps_call_rhs():
    # x = func() must NOT be stripped — it has side effects.
    source = textwrap.dedent(
        """\
        def foo():
            x = side_effect()
            return 1
    """
    )
    result = _pyflakes_strip_unused_simple_assigns(source, {"x"})
    assert "x = side_effect()" in result


def test_pyflakes_strip_unused_simple_assigns_no_change_when_used():
    source = textwrap.dedent(
        """\
        def foo(source):
            last_import_line = 0
            for line in source.splitlines():
                last_import_line += 1
            return last_import_line
    """
    )
    result = _pyflakes_strip_unused_simple_assigns(source, {"last_import_line"})
    assert result == source


def test_pyflakes_strip_unused_simple_assigns_fallback_on_empty_block():
    # If stripping would leave a block with no statements (syntax error),
    # the original source is returned unchanged.
    source = textwrap.dedent(
        """\
        def foo():
            x = 0
    """
    )
    # After stripping x = 0 the function body is empty — SyntaxError.
    result = _pyflakes_strip_unused_simple_assigns(source, {"x"})
    assert result == source


def test_pyflakes_strip_unused_simple_assigns_module_level_unchanged():
    # Module-level assignments are not flagged as UnusedVariable by pyflakes.
    source = "x = 0\n"
    result = _pyflakes_strip_unused_simple_assigns(source, {"x"})
    assert result == source


def test_pyflakes_strip_unused_simple_assigns_skips_unrelated_names():
    # A variable unused after extraction but NOT in allowed_names is preserved.
    source = textwrap.dedent(
        """\
        def foo(source):
            unrelated = 0
            lines = source.splitlines()
            return lines
    """
    )
    # "unrelated" is not in the allowed set → must not be removed.
    result = _pyflakes_strip_unused_simple_assigns(source, {"last_import_line"})
    assert "unrelated = 0" in result


def test_pyflakes_strip_unused_simple_assigns_empty_allowed():
    # Empty allowed_names means nothing can be stripped.
    source = textwrap.dedent(
        """\
        def foo(source):
            x = 0
            lines = source.splitlines()
            return lines
    """
    )
    result = _pyflakes_strip_unused_simple_assigns(source, set())
    assert result == source


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


def test_replacement_steals_post_block_skipped(monkeypatch):
    """Replacement whose last line matches the post-block line is rejected."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_do_work",
                    "placement": "module_level",
                    "helper_source": (
                        "def _do_work(data):\n"
                        "    x = compute(data)\n"
                        "    y = transform(x)\n"
                        "    z = finalize(y)\n"
                    ),
                    "call_site_replacements": [
                        "    _do_work(data)\n    return z\n",  # steals "return z"
                        "    _do_work(data)\n",
                    ],
                }
            ),
        ]
        de = DuplicateExtractor(
            _POST_STEAL_RANGES,
            source=_POST_STEAL_SOURCE,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None


def test_replacement_steals_post_block_skipped_verbose_false(monkeypatch):
    """verbose=False covers the False branch of the verbose guard."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_do_work",
                    "placement": "module_level",
                    "helper_source": (
                        "def _do_work(data):\n"
                        "    x = compute(data)\n"
                        "    y = transform(x)\n"
                        "    z = finalize(y)\n"
                    ),
                    "call_site_replacements": [
                        "    _do_work(data)\n    return z\n",  # steals "return z"
                        "    _do_work(data)\n",
                    ],
                }
            ),
        ]
        de = DuplicateExtractor(
            _POST_STEAL_RANGES,
            source=_POST_STEAL_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None


def test_strip_helper_docstring_with_docstring():
    source = 'def _helper(x):\n    """Strip me."""\n    return x\n'
    result = _strip_helper_docstring(source)
    assert '"""Strip me."""' not in result
    assert "return x" in result


def test_strip_helper_docstring_no_docstring():
    source = "def _helper(x):\n    return x\n"
    result = _strip_helper_docstring(source)
    assert result == source


def test_strip_helper_docstring_parse_error():
    bad = "def f(:\n    pass\n"
    result = _strip_helper_docstring(bad)
    assert result == bad


def test_strip_helper_docstring_non_function():
    source = "x = 1\n"
    result = _strip_helper_docstring(source)
    assert result == source


def test_strip_helper_docstring_docstring_only_body():
    # Function whose body is only a docstring — don't strip (would leave empty body).
    source = 'def _helper():\n    """Only doc."""\n'
    result = _strip_helper_docstring(source)
    assert '"""Only doc."""' in result


def test_replace_unused_in_target_name_used():
    import ast

    target = ast.parse("result = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "print(result)\n")
    assert all_r is False and any_r is False
    assert ast.unparse(new_t) == "result"


def test_replace_unused_in_target_name_unused():
    import ast

    target = ast.parse("result = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "return None\n")
    assert all_r is True and any_r is True
    assert ast.unparse(new_t) == "_"


def test_replace_unused_in_target_tuple_all_unused():
    import ast

    target = ast.parse("a, b = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "return None\n")
    assert all_r is True and any_r is True
    assert ast.unparse(new_t) == "(_, _)"


def test_replace_unused_in_target_tuple_some_unused():
    import ast

    target = ast.parse("a, b = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "print(a)\n")
    assert all_r is False and any_r is True
    assert ast.unparse(new_t) == "(a, _)"


def test_replace_unused_in_target_tuple_all_used():
    import ast

    target = ast.parse("a, b = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "print(a, b)\n")
    assert all_r is False and any_r is False


def test_replace_unused_in_target_attribute_treated_as_used():
    import ast

    target = ast.parse("self.x = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "return None\n")
    assert all_r is False and any_r is False


def test_strip_unused_call_assignments_removes_unused_single():
    # `result` never appears after the block → assignment stripped.
    replacement = "    result = _helper(x, y)\n"
    following = ["    do_something()\n", "    return z\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    _helper(x, y)\n"


def test_strip_unused_call_assignments_keeps_used_single():
    # `result` is referenced after the block → assignment kept.
    replacement = "    result = _helper(x, y)\n"
    following = ["    print(result)\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_removes_unused_tuple():
    # Both names unused after the block → assignment stripped entirely.
    replacement = "    a, b = _helper(x)\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    _helper(x)\n"


def test_strip_unused_call_assignments_partial_tuple_replaces_with_underscore():
    # One name used, one unused → replace unused with _.
    replacement = "    a, b = _helper(x)\n"
    following = ["    print(a)\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    (a, _) = _helper(x)\n"


def test_strip_unused_call_assignments_attribute_target_unchanged():
    # Target is an attribute (self.x = call()) → treated as used → left unchanged.
    replacement = "    self.result = _helper(x)\n"
    following = []
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_non_call_rhs_unchanged():
    # RHS is not a Call → leave unchanged.
    replacement = "    result = x + y\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_chained_all_unused_stripped():
    # Chained assignment where every name is unused → stripped to just the call.
    replacement = "    a = b = _helper(x)\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    _helper(x)\n"


def test_strip_unused_call_assignments_chained_some_used_unchanged():
    # Chained assignment where one name is used → left unchanged.
    replacement = "    a = b = _helper(x)\n"
    following = ["    print(a)\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_chained_no_names_unchanged():
    # Chained assignment whose targets yield no names (e.g. attributes) → unchanged.
    replacement = "    self.a = self.b = _helper(x)\n"
    following = []
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_no_assignment_unchanged():
    # Replacement is already just a call → returned as-is.
    replacement = "    _helper(x, y)\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_syntax_error_unchanged():
    # Unparseable replacement → returned unchanged.
    replacement = "    def (\n"
    following = []
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_multiline_replacement():
    # Multi-statement replacement: only the unused assignment is stripped.
    replacement = "    result = _helper(x)\n    do_other()\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    _helper(x)\n    do_other()\n"


def test_strip_unused_call_assignments_preserves_indentation():
    # Indentation of stripped replacement matches original block indent.
    replacement = "        result = _helper(x)\n"
    following = []
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "        _helper(x)\n"


def test_strip_unused_call_assignments_leading_blank_line():
    # Replacement with a blank leading line: indent is read from first content line.
    replacement = "\n    result = _helper(x)\n"
    following = []
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "\n    _helper(x)\n"


def test_strip_unused_call_assignments_await_unused_stripped():
    # `result = await _helper(x)` and `result` never used → strip assignment.
    replacement = "    result = await _helper(x)\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    await _helper(x)\n"


def test_strip_unused_call_assignments_await_used_kept():
    # `result = await _helper(x)` and `result` is used → keep assignment.
    replacement = "    result = await _helper(x)\n"
    following = ["    print(result)\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_await_tuple_unused_stripped():
    # `a, b = await _helper(x)` and neither name is used → strip assignment.
    replacement = "    a, b = await _helper(x)\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    await _helper(x)\n"


def test_strip_unused_call_assignments_await_non_call_unchanged():
    # `result = await some_awaitable` (not a call) → left unchanged.
    replacement = "    result = await some_awaitable\n"
    following = []
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_seq_ends_with_return_true():
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    return x\n"))
        is True
    )


def test_seq_ends_with_return_false_no_return():
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    y = 2\n")) is False
    )


def test_seq_ends_with_return_syntax_error():
    assert _seq_ends_with_return(_make_seq_with_source("    (\n")) is False


def test_seq_ends_with_return_empty_body():
    # Pure whitespace → ast.parse produces an empty module body.
    assert _seq_ends_with_return(_make_seq_with_source("   \n")) is False


def test_seq_ends_with_return_bare_return():
    # Bare `return` is equivalent to returning None — not flagged.
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    return\n")) is False
    )


def test_seq_ends_with_return_return_none():
    # Explicit `return None` is also equivalent to implicit None — not flagged.
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    return None\n"))
        is False
    )


def test_seq_source_contains_yield_async_with_yield():
    # The exact pattern that triggered the bug: async with ... as c: yield c
    src = "    async with Client(mcp) as c:\n        yield c\n"
    assert _seq_source_contains_yield(src) is True


def test_seq_source_contains_yield_plain_yield():
    assert _seq_source_contains_yield("    yield x\n") is True


def test_seq_source_contains_yield_from():
    assert _seq_source_contains_yield("    yield from something()\n") is True


def test_seq_source_contains_yield_no_yield():
    assert _seq_source_contains_yield("    x = 1\n    y = 2\n") is False


def test_seq_source_contains_yield_nested_funcdef_not_counted():
    # yield inside a nested def must NOT trigger the guard
    src = "    def inner():\n        yield 1\n"
    assert _seq_source_contains_yield(src) is False


def test_seq_source_contains_yield_syntax_error():
    assert _seq_source_contains_yield("    (\n") is False


def test_collector_skips_yield_sequences():
    # Sequences whose source contains yield should never be collected.
    source = textwrap.dedent(
        """\
        async def make_client():
            x = setup()
            async with Client(x) as c:
                yield c

        async def make_client2():
            x = setup()
            async with Client(x) as c:
                yield c
        """
    )
    seqs = _collect_sequences(source)
    for seq in seqs:
        assert not _seq_source_contains_yield(seq.source)


def test_replacement_contains_return_true():
    assert _replacement_contains_return("    return x\n") is True


def test_replacement_contains_return_false():
    assert _replacement_contains_return("    _helper()\n") is False


def test_replacement_contains_return_syntax_error():
    # Unclosed paren inside the wrapper → SyntaxError → False.
    assert _replacement_contains_return("    (\n") is False


def _make_steal_seq(end_line: int) -> _SeqInfo:
    return _SeqInfo(
        stmts=[], start_line=1, end_line=end_line, scope="f", source="", fingerprint=""
    )


def test_replacement_steals_post_block_at_eof():
    # Block is the last line of the file — no post-block line exists.
    source_lines = ["x = 1\n"]
    seq = _make_steal_seq(1)  # next_idx=1 >= len=1 → skip
    assert not _replacement_steals_post_block_line(
        [seq], ["y = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_blank_after():
    # Post-block line is blank but there is a non-blank line further down.
    # The check must scan past the blank to find the real post-block code.
    source_lines = ["x = 1\n", "\n", "y = 2\n"]
    seq = _make_steal_seq(1)  # next_idx=1 → "\n" → scan → next_idx=2 → "y = 2"
    assert _replacement_steals_post_block_line([seq], ["y = 2\n"], source_lines)


def test_replacement_steals_post_block_blank_after_no_match():
    # Blank after block, but replacement doesn't steal the non-blank post-block line.
    source_lines = ["x = 1\n", "\n", "y = 2\n"]
    seq = _make_steal_seq(1)
    assert not _replacement_steals_post_block_line(
        [seq], ["z = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_all_blank_after():
    # Only blank lines follow the block — no real post-block line to steal.
    source_lines = ["x = 1\n", "\n", "\n"]
    seq = _make_steal_seq(1)
    assert not _replacement_steals_post_block_line(
        [seq], ["z = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_no_match():
    # Replacement last line doesn't match post-block line.
    source_lines = ["x = 1\n", "y = 2\n"]
    seq = _make_steal_seq(1)  # next_idx=1 → "y = 2"
    assert not _replacement_steals_post_block_line(
        [seq], ["z = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_match():
    # Replacement last line matches post-block line → steal detected.
    source_lines = ["x = 1\n", "y = 2\n"]
    seq = _make_steal_seq(1)  # next_idx=1 → "y = 2"
    assert _replacement_steals_post_block_line(
        [seq], ["z = helper()\ny = 2\n"], source_lines
    )
