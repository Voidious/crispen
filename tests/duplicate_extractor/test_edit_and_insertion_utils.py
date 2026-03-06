import textwrap
import libcst as cst
from libcst.metadata import MetadataWrapper
from crispen.refactors.duplicate_extractor import (
    _build_helper_insertion,
    _FunctionCollector,
    _SeqInfo,
    _SequenceCollector,
    _apply_edits,
    _find_insertion_point,
    _find_escaping_vars,
    _normalize_replacement_indentation,
    _replacement_steals_post_block_line,
)


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


def _make_esc_seq(start: int, end: int) -> _SeqInfo:
    """Create a _SeqInfo for escaping-vars tests."""
    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope="foo",
        source="",
        fingerprint="",
    )


def test_find_escaping_vars_no_assignments():
    # Block has no assignments → skip (branch A), returns empty set.
    source_lines = [
        "def foo():\n",
        "    compute()\n",
        "    transform()\n",
        "    use_result()\n",
    ]
    seq = _make_esc_seq(2, 3)
    assert _find_escaping_vars([seq], source_lines) == set()


def test_find_escaping_vars_nothing_after_block():
    # Block is the last thing in scope → after_lines empty (branch D), returns set().
    source_lines = [
        "def foo():\n",
        "    x = compute()\n",
        "    y = transform(x)\n",
        "    z = finalize(y)\n",
    ]
    seq = _make_esc_seq(2, 4)
    assert _find_escaping_vars([seq], source_lines) == set()


def test_find_escaping_vars_escapes():
    # Block assigns z; z is used after the block → {"z"}.
    # Also covers: blank line (branch B) and lower-indent stop (branch C).
    source_lines = [
        "def foo():\n",
        "    x = compute()\n",
        "    y = transform(x)\n",
        "    z = finalize(y)\n",  # block ends line 4
        "\n",  # blank → branch B
        "    assert z == 42\n",  # same indent, uses z
        "\n",
        "def bar():\n",  # indent 0 < 4 → branch C (stop)
        "    pass\n",
    ]
    seq = _make_esc_seq(2, 4)
    assert _find_escaping_vars([seq], source_lines) == {"z"}


def test_find_escaping_vars_no_escape():
    # Block assigns x/y/z; none referenced after the block → set().
    source_lines = [
        "def foo():\n",
        "    x = compute()\n",
        "    y = transform(x)\n",
        "    z = finalize(y)\n",
        "    print('done')\n",  # uses 'print', not x/y/z
    ]
    seq = _make_esc_seq(2, 4)
    assert _find_escaping_vars([seq], source_lines) == set()


def test_find_escaping_vars_syntax_error_after():
    # After source is invalid Python → SyntaxError branch: continue, returns set().
    source_lines = [
        "def foo():\n",
        "    x = compute()\n",
        "    y = transform(x)\n",
        "    z = finalize(y)\n",
        "    def bar(x\n",  # unclosed paren at same indent
    ]
    seq = _make_esc_seq(2, 4)
    assert _find_escaping_vars([seq], source_lines) == set()


def test_find_escaping_vars_module_level_stops_at_def():
    # Module-level block (indent 0): a non-def/class line is included,
    # then a def line stops the scan (break via re.match).
    source_lines = [
        "x = compute()\n",
        "y = transform(x)\n",
        "z = finalize(y)\n",  # block ends line 3
        "CONSTANT = 42\n",  # module-level non-def → appended (False branch of re.match)
        "def foo(z):\n",  # module-level def → stop
        "    return z\n",
    ]
    seq = _make_esc_seq(1, 3)
    # CONSTANT is in after_lines; not in assigned → set().
    # z inside def foo(z) is not scanned (stopped before that def).
    assert _find_escaping_vars([seq], source_lines) == set()


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


def _collect_functions(source: str):
    tree = cst.parse_module(source)
    lines = source.splitlines(keepends=True)
    collector = _FunctionCollector(lines)
    MetadataWrapper(tree).visit(collector)
    return collector.functions


def test_function_collector_module_level():
    source = "def foo():\n    pass\n"
    funcs = _collect_functions(source)
    assert len(funcs) == 1
    assert funcs[0].name == "foo"
    assert funcs[0].scope == "<module>"
    assert funcs[0].body_stmt_count == 1
    assert funcs[0].params == []


def test_function_collector_class_level():
    source = "class C:\n    def method(self):\n        pass\n"
    funcs = _collect_functions(source)
    assert len(funcs) == 1
    assert funcs[0].name == "method"
    assert funcs[0].scope == "C"
    assert funcs[0].body_stmt_count == 1
    assert funcs[0].params == ["self"]


def test_function_collector_skips_nested():
    source = "def outer():\n    def inner():\n        pass\n"
    funcs = _collect_functions(source)
    assert len(funcs) == 1
    assert funcs[0].name == "outer"
    assert funcs[0].body_stmt_count == 1
    assert funcs[0].params == []


def test_function_collector_collects_body_source():
    source = "def foo():\n    x = 1\n    y = 2\n"
    funcs = _collect_functions(source)
    assert len(funcs) == 1
    assert "x = 1" in funcs[0].body_source


def test_function_collector_collects_stmt_count():
    source = "def foo():\n    pass\n"
    funcs = _collect_functions(source)
    assert funcs[0].body_stmt_count == 1


def test_function_collector_collects_params():
    source = "def f(x, y):\n    pass\n"
    funcs = _collect_functions(source)
    assert funcs[0].params == ["x", "y"]


def test_function_collector_no_params():
    source = "def f():\n    pass\n"
    funcs = _collect_functions(source)
    assert funcs[0].params == []


def test_sequence_collector_class_scope():
    """_SequenceCollector sets class_scope for sequences inside class methods."""

    source = textwrap.dedent(
        """\
        x = 1
        y = 2
        z = 3

        class MyClass:
            def method(self):
                a = 1
                b = 2
                c = 3
        """
    )
    lines = source.splitlines(keepends=True)
    tree = cst.parse_module(source)
    collector = _SequenceCollector(lines, max_seq_len=8)
    MetadataWrapper(tree).visit(collector)

    module_seqs = [s for s in collector.sequences if s.class_scope is None]
    class_seqs = [s for s in collector.sequences if s.class_scope == "MyClass"]
    assert module_seqs, "expected module-level sequences with class_scope=None"
    assert class_seqs, "expected class-method sequences with class_scope='MyClass'"


def _make_seq_info(start: int, end: int, src: str = "") -> _SeqInfo:
    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope="foo",
        source=src,
        fingerprint="",
    )


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
