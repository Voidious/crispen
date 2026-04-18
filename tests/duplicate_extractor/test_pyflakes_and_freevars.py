import textwrap
from crispen.refactors.duplicate_extractor import (
    _apply_edits,
    _is_pure_literal,
    _missing_free_vars,
    _names_assigned_in,
    _names_in_edit_texts,
    _pyflakes_new_undefined_names,
    _pyflakes_strip_unused_simple_assigns,
    _replace_unused_in_target,
    _strip_unused_call_assignments,
)


def test_pyflakes_new_undefined_names_returns_empty_when_no_new_issues():
    # Names undefined in both original and candidate → no NEW issues.
    original = "def foo():\n    return bar()\n"
    candidate = "def _h():\n    pass\n\ndef foo():\n    return bar()\n"
    assert _pyflakes_new_undefined_names(original, candidate) == set()


def test_pyflakes_new_undefined_names_detects_introduced_name():
    # candidate introduces a reference to an unassigned name not in original.
    original = "def foo():\n    x = 1\n    return x\n"
    # candidate removes the assignment, leaving x undefined at the call site
    candidate = "def _h():\n    x = 1\n\ndef foo():\n    _h(x)\n    return x\n"
    assert "x" in _pyflakes_new_undefined_names(original, candidate)


def test_is_pure_literal_constant():
    import ast

    assert _is_pure_literal(ast.parse("0", mode="eval").body)
    assert _is_pure_literal(ast.parse('"s"', mode="eval").body)
    assert _is_pure_literal(ast.parse("None", mode="eval").body)
    assert _is_pure_literal(ast.parse("True", mode="eval").body)


def test_is_pure_literal_containers():
    import ast

    assert _is_pure_literal(ast.parse("[]", mode="eval").body)
    assert _is_pure_literal(ast.parse("(1, 2)", mode="eval").body)
    assert _is_pure_literal(ast.parse("{1: 2}", mode="eval").body)
    assert _is_pure_literal(ast.parse("{1, 2}", mode="eval").body)


def test_is_pure_literal_call_is_false():
    import ast

    assert not _is_pure_literal(ast.parse("func()", mode="eval").body)


def test_is_pure_literal_name_is_false():
    import ast

    assert not _is_pure_literal(ast.parse("x", mode="eval").body)


def test_is_pure_literal_nested_call_is_false():
    import ast

    assert not _is_pure_literal(ast.parse("[func()]", mode="eval").body)


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


def test_names_in_edit_texts_collects_from_all_edits():
    groups = [
        (
            "_helper",
            [
                (1, 3, "def _helper(last_import_line):\n    return last_import_line\n"),
                (5, 6, "result = _helper(x)\n"),
            ],
            "msg",
        )
    ]
    names = _names_in_edit_texts(groups)
    assert "last_import_line" in names
    assert "_helper" in names
    assert "result" in names
    assert "x" in names


def test_names_in_edit_texts_skips_syntax_errors():
    groups = [("_h", [(1, 2, "def (\n")], "msg")]
    # Should not raise — returns whatever names were parseable.
    names = _names_in_edit_texts(groups)
    assert isinstance(names, set)


def test_missing_free_vars_catches_missing_name():
    # The exact bug pattern: `new_source` is a local variable read in the
    # original block, but the LLM turned it into `transformer.new_source`
    # (an attribute access).  Neither the call site nor the helper body contain
    # a bare `new_source` Name node.
    source = (
        "def run(transformer, file_msgs, filepath):\n"
        "    new_source = get_source()\n"
        "    current_source = new_source\n"
    )
    block_src = "    current_source = new_source\n"
    call_src = "    current_source = _h(transformer, filepath, file_msgs)\n"
    helper_src = (
        "def _h(transformer, filepath, file_msgs):\n"
        "    return transformer.new_source\n"
    )
    assert "new_source" in _missing_free_vars(block_src, [call_src], helper_src, source)


def test_missing_free_vars_no_missing_when_passed_as_arg():
    # Free var is passed as an argument to the helper → not missing.
    source = (
        "def run():\n    new_source = get_source()\n    current_source = new_source\n"
    )
    block_src = "    current_source = new_source\n"
    call_src = "    current_source = _h(new_source)\n"
    helper_src = "def _h(new_source):\n    return new_source\n"
    assert _missing_free_vars(block_src, [call_src], helper_src, source) == set()


def test_missing_free_vars_ignores_block_locals():
    # `x` is assigned AND read within the block — it is a local, not a free
    # variable.  It should not be flagged even if it's absent from the helper.
    source = "def run():\n    x = 1\n    result = x + 1\n"
    block_src = "    x = 1\n    result = x + 1\n"
    call_src = "    result = _h()\n"
    helper_src = "def _h():\n    x = 1\n    return x + 1\n"
    assert _missing_free_vars(block_src, [call_src], helper_src, source) == set()


def test_missing_free_vars_ignores_module_level_names():
    # `compute`, `transform`, `finalize` are module-level function names that
    # are never assigned anywhere — the helper can reference them directly.
    source = (
        "def foo():\n"
        "    x = compute(data)\n"
        "    y = transform(x)\n"
        "    z = finalize(y)\n"
    )
    block_src = "    x = compute(data)\n    y = transform(x)\n    z = finalize(y)\n"
    call_src = "    _helper(data)\n"
    helper_src = "def _helper(data):\n    pass\n"
    assert _missing_free_vars(block_src, [call_src], helper_src, source) == set()


def test_missing_free_vars_syntax_error_in_block_returns_empty():
    assert (
        _missing_free_vars("not valid python!!!", ["x = 1\n"], "def f(): pass\n", "")
        == set()
    )


def test_missing_free_vars_syntax_error_in_replacement_returns_empty():
    source = "def run():\n    a = 1\n"
    assert (
        _missing_free_vars("x = a\n", ["not valid!!!\n"], "def f(): pass\n", source)
        == set()
    )


def test_missing_free_vars_syntax_error_in_source_returns_empty():
    assert (
        _missing_free_vars("x = a\n", ["y = a\n"], "def f(a): pass\n", "not valid!!!")
        == set()
    )


def test_missing_free_vars_empty_block_returns_empty():
    # A block with no reads has no free vars → nothing can be missing.
    source = "def run():\n    x = 1\n"
    block_src = "    x = 1\n"
    call_src = "    _h()\n"
    helper_src = "def _h():\n    x = 1\n"
    assert _missing_free_vars(block_src, [call_src], helper_src, source) == set()


def test_missing_free_vars_function_parameter_is_caught():
    # A function parameter that's free in the block must appear in the
    # replacement — parameters are local to the function and cannot be
    # accessed by a helper without being passed as an argument.
    source = "def run(verbose):\n    msg = verbose\n"
    block_src = "    msg = verbose\n"
    call_src = "    msg = _h()\n"
    helper_src = "def _h():\n    pass\n"
    assert "verbose" in _missing_free_vars(block_src, [call_src], helper_src, source)


def test_names_assigned_in_simple():
    assert _names_assigned_in("x = 1\n") == {"x"}


def test_names_assigned_in_tuple_unpack():
    assert _names_assigned_in("x, y = f()\n") == {"x", "y"}


def test_names_assigned_in_augassign():
    assert _names_assigned_in("x += 1\n") == {"x"}


def test_names_assigned_in_no_assign():
    assert _names_assigned_in("f()\n") == set()


def test_names_assigned_in_syntax_error():
    assert _names_assigned_in("def (\n") == set()


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
