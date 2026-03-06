import textwrap
from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import (
    _build_helper_insertion,
    _SeqInfo,
    _apply_edits,
    _find_insertion_point,
    _normalize_replacement_indentation,
    _seq_ends_with_return,
    _replacement_contains_return,
    _replacement_steals_post_block_line,
    _helper_imports_local_name,
    DuplicateExtractor,
)
from .test_llm_integration_core import _make_extract_response, _make_veto_response


def _make_seq_with_source(source: str) -> _SeqInfo:
    return _SeqInfo(
        stmts=[], start_line=1, end_line=1, scope="f", source=source, fingerprint=""
    )


def test_normalize_indentation_already_correct():
    # Replacement already matches the block's indentation — unchanged.
    seq = _make_seq_with_source("    x = compute()\n    y = finalize(x)\n")
    replacement = "    result = helper()\n"
    assert (
        _normalize_replacement_indentation(seq, replacement)
        == "    result = helper()\n"
    )


def test_normalize_indentation_col0_to_indented():
    # Replacement at column 0 is re-indented to match the original block.
    seq = _make_seq_with_source("    x = compute()\n    y = finalize(x)\n")
    replacement = "result = helper()\n"
    assert (
        _normalize_replacement_indentation(seq, replacement)
        == "    result = helper()\n"
    )


def test_normalize_indentation_multiline():
    # Multi-line replacement at column 0 gets uniformly re-indented.
    seq = _make_seq_with_source("        x = a()\n        y = b(x)\n")
    replacement = "x = helper()\nif x is None:\n    x = default()\n"
    expected = (
        "        x = helper()\n        if x is None:\n            x = default()\n"
    )
    assert _normalize_replacement_indentation(seq, replacement) == expected


def test_normalize_indentation_module_level_block():
    # Module-level block (no indent) — replacement is just dedented.
    seq = _make_seq_with_source("x = compute()\ny = finalize(x)\n")
    replacement = "result = helper()\n"
    assert _normalize_replacement_indentation(seq, replacement) == "result = helper()\n"


def test_normalize_indentation_empty_source():
    # Empty source — no indentation can be inferred; replacement returned as-is.
    seq = _make_seq_with_source("")
    replacement = "result = helper()\n"
    assert _normalize_replacement_indentation(seq, replacement) == replacement


def test_apply_edits_no_edits():
    source = "a = 1\nb = 2\n"
    assert _apply_edits(source, []) == source


def test_apply_edits_replacement():
    source = "a = 1\nb = 2\nc = 3\n"
    # Replace line index 1 (b = 2) with new content
    result = _apply_edits(source, [(1, 2, "x = 99\n")])
    assert result == "a = 1\nx = 99\nc = 3\n"


def test_apply_edits_insertion():
    source = "a = 1\nb = 2\n"
    # Insert before line index 1 (b = 2)
    result = _apply_edits(source, [(1, 1, "INSERTED\n")])
    assert result == "a = 1\nINSERTED\nb = 2\n"


def test_apply_edits_overlapping_skipped():
    source = "a = 1\nb = 2\nc = 3\n"
    edits = [
        (0, 2, "FIRST\n"),
        (1, 3, "SECOND\n"),  # overlaps with first
    ]
    result = _apply_edits(source, edits)
    # Higher-start edit (SECOND) wins; FIRST overlaps and is skipped
    assert "SECOND" in result
    assert "FIRST" not in result


def test_apply_edits_no_trailing_newline_source():
    source = "a = 1"  # no trailing newline
    result = _apply_edits(source, [(0, 1, "b = 2\n")])
    assert result == "b = 2\n"


def test_apply_edits_no_trailing_newline_text():
    source = "a = 1\nb = 2\n"
    # Replacement text without trailing newline
    result = _apply_edits(source, [(0, 1, "x = 99")])
    assert result == "x = 99\nb = 2\n"


def test_find_insertion_point_module_with_imports():
    source = "import os\nfrom sys import argv\n\ndef foo():\n    pass\n"
    # Should insert after the last import (index 1), so return 2
    assert _find_insertion_point(source, "<module>") == 2


def test_find_insertion_point_module_no_imports():
    source = "a = 1\n"
    # No imports: last_import stays -1, returns 0
    assert _find_insertion_point(source, "<module>") == 0


def test_find_insertion_point_function_found():
    source = "import os\n\ndef target():\n    pass\n"
    # def target is at line index 2
    assert _find_insertion_point(source, "target") == 2


def test_find_insertion_point_function_not_found():
    source = "a = 1\n"
    # Falls back to 0
    assert _find_insertion_point(source, "missing_func") == 0


def test_find_insertion_point_class_method_inserts_before_class():
    # def bar is indented inside class Foo; helper must go before the class,
    # not inside it (which would end the class and turn _analyze into a nested func).
    source = "import os\n\nclass Foo:\n\n    def bar(self):\n        pass\n"
    # source_lines: ["import os", "", "class Foo:", "",
    #                "    def bar(self):", "        pass"]
    # "def bar" found at i=4 (indent=4).  Walk back:
    #   j=3 → blank → skip; j=2 → "class Foo:" indent=0 < 4 → return 2
    assert _find_insertion_point(source, "bar") == 2


def test_find_insertion_point_nested_function_no_class():
    # def inner is indented inside def outer (no enclosing class).
    # method_indent > 0, loop finds a non-class def at lower indent → break.
    # Falls through to decorator walk, which returns i (the line of def inner).
    source = "def outer():\n    def inner():\n        pass\n"
    # "def inner" found at i=1 (indent=4).  Walk back:
    #   j=0 → "def outer():" indent=0 < 4, not a class → break.
    # Falls through to return 1.
    assert _find_insertion_point(source, "inner") == 1


def test_find_insertion_point_nested_func_ignores_unrelated_class():
    # Regression: a nested function inside a module-level function must not
    # be confused with a class method just because an unrelated class appears
    # earlier in the file.  Before the fix the backward walk would skip past
    # the outer function (non-class, lower indent) and incorrectly match the
    # unrelated class, causing the helper to be inserted between the class's
    # decorator and its class statement.
    import textwrap as _textwrap

    source = _textwrap.dedent(
        """\
        @dataclass
        class _SplitTask:
            pass


        def _find_free_vars():
            x = 1
            def _collect_loads():
                pass
        """
    )
    # source_lines (0-based):
    #  0: "@dataclass\n"
    #  1: "class _SplitTask:\n"
    #  2: "    pass\n"
    #  3: "\n"
    #  4: "\n"
    #  5: "def _find_free_vars():\n"
    #  6: "    x = 1\n"
    #  7: "    def _collect_loads():\n"
    #  8: "        pass\n"
    # "def _collect_loads" found at i=7 (indent=4).  Walk back:
    #   j=6: "    x = 1" indent=4, not < 4 → continue
    #   j=5: "def _find_free_vars():" indent=0 < 4, NOT class → break
    # Falls through to decorator walk: j=6 ("    x = 1"), not a decorator
    # → break → return j+1 = 7.
    # The old (unfixed) code would have continued past j=5 and returned 1,
    # placing the helper between @dataclass and class _SplitTask:.
    result = _find_insertion_point(source, "_collect_loads")
    assert result != 1, "must not insert inside @dataclass/_SplitTask boundary"
    assert result == 7


def test_find_insertion_point_indented_func_at_file_start():
    # Edge case: the target def has method_indent > 0 but is at line 0 so the
    # backward-search loop range is empty.  Falls through to decorator walk
    # which also exits immediately (j=-1), returning 0.
    source = "    def inner():\n        pass\n"
    # "def inner" found at i=0 (indent=4).  range(-1, -1, -1) is empty → loop
    # body never runs → fall through to decorator walk → j = -1 → return 0.
    assert _find_insertion_point(source, "inner") == 0


def test_find_insertion_point_async_def():
    # Regression: helpers extracted from async functions were inserted at line 0
    # (before imports) because the pattern only matched 'def', not 'async def'.
    source = (
        "import pytest\n"  # 0
        "\n"  # 1
        "async def target(client):\n"  # 2
        "    pass\n"  # 3
    )
    assert _find_insertion_point(source, "target") == 2


def test_find_insertion_point_async_def_with_decorator():
    # async def with a preceding decorator: helper should land before the decorator.
    source = (
        "import pytest\n"  # 0
        "\n"  # 1
        "@pytest.mark.asyncio\n"  # 2
        "async def target(client):\n"  # 3
        "    pass\n"  # 4
    )
    assert _find_insertion_point(source, "target") == 2


def test_find_insertion_point_skips_over_decorators():
    # Helper must be inserted before the decorator block, not between the
    # decorators and the def they decorate.
    source = (
        "import os\n"  # 0
        "\n"  # 1
        "@decorator\n"  # 2
        "def target():\n"  # 3
        "    pass\n"  # 4
    )
    # Without the fix this would return 3 (the def line); with the fix it
    # should return 2 (the @decorator line).
    assert _find_insertion_point(source, "target") == 2


def test_find_insertion_point_skips_over_multiline_decorator():
    # Multi-line decorator: @patch(\n    "..."\n) above the def.
    source = (
        "import os\n"  # 0
        "\n"  # 1
        "@patch(\n"  # 2
        '    "some.module"\n'  # 3
        ")\n"  # 4
        "def target():\n"  # 5
        "    pass\n"  # 6
    )
    # Should return 2 (before the @patch line), not 5 (the def line).
    assert _find_insertion_point(source, "target") == 2


def test_build_helper_insertion_absorbs_blank_before_function():
    # Blank line between import and def is absorbed; 2 blank lines ensured.
    source = "import os\n\ndef foo():\n    pass\n"
    lines = source.splitlines(keepends=True)
    helper = "def _helper():\n    pass\n"
    start, end, text = _build_helper_insertion(lines, 2, helper, "module_level")
    # Blank line at index 1 is before insert_pos=2, so before_blanks=1 → start=1.
    assert start == 1
    assert end == 2  # no blanks after insert_pos (def foo starts there)
    assert text.startswith("\n\n")
    assert text.endswith("\n\n")
    assert "def _helper():" in text


def test_build_helper_insertion_no_surrounding_blanks():
    # No blanks to absorb → pure insertion with 2 blank lines each side.
    source = "import os\ndef foo():\n    pass\n"
    lines = source.splitlines(keepends=True)
    helper = "def _helper():\n    pass\n"
    start, end, text = _build_helper_insertion(lines, 1, helper, "module_level")
    assert start == 1
    assert end == 1  # pure insertion
    assert text.startswith("\n\n")
    assert text.endswith("\n\n")


def test_build_helper_insertion_staticmethod_uses_one_blank():
    # Staticmethod placement: 1 blank line before and after.
    source = "class Foo:\n    def bar(self):\n        pass\n"
    lines = source.splitlines(keepends=True)
    helper = "    @staticmethod\n    def _h():\n        pass\n"
    start, end, text = _build_helper_insertion(lines, 1, helper, "staticmethod:Foo")
    assert start == 1
    assert end == 1  # no blanks to absorb
    assert text.startswith("\n")
    assert not text.startswith("\n\n")
    assert text.endswith("\n\n")  # clean + 1 trailing blank = \n + \n


def test_build_helper_insertion_absorbs_blank_at_insert_pos():
    # When insert_pos itself is a blank line, after_blanks counts it.
    source = "import os\n\ndef foo():\n    pass\n"
    lines = source.splitlines(keepends=True)
    helper = "def _helper():\n    pass\n"
    # insert_pos=1 lands on the blank line: after_blanks=1, end=2.
    start, end, text = _build_helper_insertion(lines, 1, helper, "module_level")
    assert start == 1
    assert end == 2
    assert text.startswith("\n\n")
    assert text.endswith("\n\n")


def test_build_helper_insertion_strips_extra_newlines_from_helper():
    # If the LLM returns a helper with leading/trailing blank lines, they are stripped.
    source = "import os\ndef foo():\n    pass\n"
    lines = source.splitlines(keepends=True)
    helper = "\n\ndef _helper():\n    pass\n\n\n"
    start, end, text = _build_helper_insertion(lines, 1, helper, "module_level")
    assert text.startswith("\n\n")
    assert text.endswith("\n\n")
    assert "\n\n\n\ndef _helper" not in text  # no extra leading blanks inside text


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


def test_helper_imports_local_name_true():
    helper = "def _h():\n    import mock_client\n    mock_client.run()\n"
    original = "def test(mock_client):\n    mock_client.run()\n"
    assert _helper_imports_local_name(helper, original) is True


def test_helper_imports_local_name_already_imported_in_original():
    # mock_client is already a top-level import → not a local-only name.
    helper = "def _h():\n    import mock_client\n    mock_client.run()\n"
    original = "import mock_client\ndef test(x):\n    mock_client.run()\n"
    assert _helper_imports_local_name(helper, original) is False


def test_helper_imports_local_name_no_imports_in_helper():
    helper = "def _h():\n    pass\n"
    original = "def test(mock_client):\n    pass\n"
    assert _helper_imports_local_name(helper, original) is False


def test_helper_imports_local_name_syntax_error_helper():
    assert _helper_imports_local_name("def (:\n", "def test(x):\n    pass\n") is False


def test_helper_imports_local_name_syntax_error_original():
    assert _helper_imports_local_name("def _h():\n    import x\n", "(:\n") is False


def test_helper_imports_local_name_from_import_in_helper():
    # "from X import Y" in helper: the tracked name is "Y", not "X".
    # If "Y" is a param in the original, it is flagged.
    helper = "def _h():\n    from pkg import mock_client\n    mock_client.run()\n"
    original = "def test(mock_client):\n    mock_client.run()\n"
    assert _helper_imports_local_name(helper, original) is True


def test_helper_imports_local_name_from_import_in_original():
    # Top-level "from pkg import something" in the original covers the branch
    # in the orig_top_imports loop and prevents false-positive flagging.
    helper = "def _h():\n    import something\n    something.run()\n"
    original = "from pkg import something\ndef test(x):\n    something.run()\n"
    assert _helper_imports_local_name(helper, original) is False


def test_helper_imports_local_name_vararg():
    # Function with *args: vararg name tracked as potential local.
    helper = "def _h():\n    import args\n"
    original = "def test(*args):\n    pass\n"
    assert _helper_imports_local_name(helper, original) is True


def test_helper_imports_local_name_kwarg():
    # Function with **kwargs: kwarg name tracked as potential local.
    helper = "def _h():\n    import kwargs\n"
    original = "def test(**kwargs):\n    pass\n"
    assert _helper_imports_local_name(helper, original) is True


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


def _make_return_block_extract_response():
    return _make_extract_response(
        {
            "function_name": "_helper",
            "placement": "module_level",
            "helper_source": (
                "def _helper():\n"
                "    x = compute(data)\n"
                "    y = transform(x)\n"
                "    return y\n"
            ),
            # replacement drops the return — this is the bug being guarded
            "call_site_replacements": [
                "    _helper()\n",
                "    _helper()\n",
            ],
        }
    )


def test_block_ends_with_return_guard_skips(monkeypatch, capsys):
    """Extraction rejected when block ends with return but replacement omits it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_return_block_extract_response(),
        ]
        de = DuplicateExtractor(
            _RETURN_BLOCK_RANGES,
            source=_RETURN_BLOCK_SOURCE,
            extraction_retries=0,
            llm_verify_retries=0,
        )
    assert de._new_source is None
    assert "block ends with return but replacement omits it" in capsys.readouterr().err


def test_block_ends_with_return_guard_skips_silent(monkeypatch):
    """verbose=False: extraction rejected with no stderr output."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_return_block_extract_response(),
        ]
        de = DuplicateExtractor(
            _RETURN_BLOCK_RANGES,
            source=_RETURN_BLOCK_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )
    assert de._new_source is None


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


def _make_import_local_extract_response():
    return _make_extract_response(
        {
            "function_name": "_helper",
            "placement": "module_level",
            # helper imports mock_client instead of taking it as a parameter
            "helper_source": (
                "def _helper():\n"
                "    import mock_client\n"
                "    x = compute(data)\n"
                "    y = transform(x)\n"
                "    z = finalize(y)\n"
            ),
            "call_site_replacements": [
                "    _helper()\n",
                "    _helper()\n",
            ],
        }
    )


def test_helper_imports_local_guard_skips(monkeypatch, capsys):
    """Extraction rejected when helper imports a name that is a param in original."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_import_local_extract_response(),
        ]
        de = DuplicateExtractor(
            _PARAM_DUP_RANGES,
            source=_PARAM_DUP_SOURCE,
            extraction_retries=0,
            llm_verify_retries=0,
        )
    assert de._new_source is None
    assert "helper imports a name that is a parameter/local" in capsys.readouterr().err


def test_helper_imports_local_guard_skips_silent(monkeypatch):
    """verbose=False: extraction rejected with no stderr output."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_import_local_extract_response(),
        ]
        de = DuplicateExtractor(
            _PARAM_DUP_RANGES,
            source=_PARAM_DUP_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )
    assert de._new_source is None
