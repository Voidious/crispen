from crispen.refactors.duplicate_extractor import (
    _SeqInfo,
    _find_escaping_vars,
    _missing_free_vars,
    _normalize_source,
)


def test_normalize_source_free_variables_match():
    # Blocks differing only in free variable names should get the same fingerprint.
    # This is the core case: `p = a * 2; if p > 100: p += 1` vs the same with q/b.
    src_a = "p = a * 2\nif p > 100:\n    p += 1\n"
    src_b = "q = b * 2\nif q > 100:\n    q += 1\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


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
