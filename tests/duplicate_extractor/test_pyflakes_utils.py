import textwrap
from crispen.refactors.duplicate_extractor import (
    _missing_free_vars,
    _pyflakes_new_undefined_names,
    _pyflakes_strip_unused_simple_assigns,
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
