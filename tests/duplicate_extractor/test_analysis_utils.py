import textwrap
from crispen.refactors.duplicate_extractor import (
    _collect_attribute_names,
    _collect_called_attr_names,
    _extract_defined_names,
    _SeqInfo,
    _collect_called_names,
    _has_call_to,
    _find_escaping_vars,
    _has_mutable_literal_is_check,
    _has_param_overwritten_before_read,
    _names_assigned_in,
    _missing_free_vars,
    _pyflakes_new_undefined_names,
    _verify_extraction,
    DuplicateExtractor,
)


def test_has_mutable_literal_is_check_set_constructor():
    assert _has_mutable_literal_is_check("if x is set(): pass") is True


def test_has_mutable_literal_is_check_list_constructor():
    assert _has_mutable_literal_is_check("if x is list(): pass") is True


def test_has_mutable_literal_is_check_dict_constructor():
    assert _has_mutable_literal_is_check("if x is dict(): pass") is True


def test_has_mutable_literal_is_check_list_literal():
    assert _has_mutable_literal_is_check("if x is []: pass") is True


def test_has_mutable_literal_is_check_dict_literal():
    assert _has_mutable_literal_is_check("if x is {}: pass") is True


def test_has_mutable_literal_is_check_isnot():
    assert _has_mutable_literal_is_check("if x is not set(): pass") is True


def test_has_mutable_literal_is_check_none_is_fine():
    assert _has_mutable_literal_is_check("if x is None: pass") is False


def test_has_mutable_literal_is_check_isinstance_is_fine():
    assert _has_mutable_literal_is_check("if isinstance(x, set): pass") is False


def test_has_mutable_literal_is_check_equality_is_fine():
    # == comparison with set() is valid; only identity (`is`) is wrong
    assert _has_mutable_literal_is_check("if x == set(): pass") is False


def test_has_mutable_literal_is_check_syntax_error():
    assert _has_mutable_literal_is_check("def f(x:") is False


def test_verify_extraction_rejects_mutable_is_in_helper():
    helper = "def h(x):\n    if x is set(): return True\n    return False\n"
    assert _verify_extraction(helper, ["h(a)\n"]) is False


def test_verify_extraction_rejects_mutable_is_in_replacement():
    helper = "def h(x):\n    return x\n"
    assert _verify_extraction(helper, ["if r is set(): pass\n"]) is False


def test_verify_extraction_rejects_indented_mutable_is_in_replacement():
    # Indented replacements (function-body code) are wrapped before checking,
    # so `is set()` is caught even when ast.parse would fail on raw indented text.
    helper = "def h(x):\n    return x\n"
    assert _verify_extraction(helper, ["    if r is set(): pass\n"]) is False


def test_collect_attribute_names_basic():
    assert _collect_attribute_names("x.foo()\ny.bar") == {"foo", "bar"}


def test_collect_attribute_names_nested():
    assert "baz" in _collect_attribute_names("a.b.baz()")


def test_collect_attribute_names_syntax_error():
    assert _collect_attribute_names("def f(x:") == set()


def test_collect_attribute_names_no_attrs():
    assert _collect_attribute_names("x = 1 + 2") == set()


def test_collect_called_attr_names_method_call():
    # obj.foo() → "foo" is a called attribute
    assert _collect_called_attr_names("obj.foo()") == {"foo"}


def test_collect_called_attr_names_ignores_plain_access():
    # obj.bar (not called) → not included
    assert "bar" not in _collect_called_attr_names("x = obj.bar")


def test_collect_called_attr_names_ignores_type_annotation():
    # ast.AST used as a type annotation is NOT a method call → not flagged
    assert "AST" not in _collect_called_attr_names(
        "def f(x) -> Optional[ast.AST]: pass"
    )


def test_collect_called_attr_names_syntax_error():
    assert _collect_called_attr_names("def f(x:") == set()


def test_collect_called_attr_names_no_calls():
    assert _collect_called_attr_names("x = 1 + 2") == set()


def test_has_call_to_direct_call():
    assert _has_call_to("foo", "foo()\n") is True


def test_has_call_to_attribute_call():
    assert _has_call_to("foo", "obj.foo()\n") is True


def test_has_call_to_missing():
    assert _has_call_to("foo", "bar()\n") is False


def test_has_call_to_syntax_error():
    assert _has_call_to("foo", "def f(x:") is False


def test_has_param_overwritten_before_read_false_when_param_is_read():
    # Parameter is read before (or without) being reassigned — should return False.
    helper = "def fn(x):\n    return x + 1\n"
    assert _has_param_overwritten_before_read(helper) is False


def test_has_param_overwritten_before_read_true_when_immediately_overwritten():
    # Parameter is assigned on the first statement without being read — True.
    helper = "def setup(client):\n    client = object()\n    return client\n"
    assert _has_param_overwritten_before_read(helper) is True


def test_has_param_overwritten_before_read_false_for_conditional_default():
    # The ``if x is None: x = default`` pattern reads before writing — False.
    helper = "def fn(x=None):\n    if x is None:\n        x = []\n    return x\n"
    assert _has_param_overwritten_before_read(helper) is False


def test_has_param_overwritten_before_read_vararg_and_kwarg():
    # Covers the vararg/kwarg branches — neither is overwritten here.
    helper = "def fn(*args, **kwargs):\n    return args, kwargs\n"
    assert _has_param_overwritten_before_read(helper) is False


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


def test_extract_defined_names_basic():
    source = textwrap.dedent(
        """\
        def foo():
            pass

        async def bar():
            pass

        class Baz:
            pass
        """
    )
    assert _extract_defined_names(source) == {"foo", "bar", "Baz"}


def test_extract_defined_names_syntax_error():
    assert _extract_defined_names("def (\n") == set()


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


def test_collect_called_names_direct():
    names = _collect_called_names("foo()\n")
    assert "foo" in names


def test_collect_called_names_method():
    names = _collect_called_names("obj.bar()\n")
    assert "bar" in names


def test_collect_called_names_empty():
    names = _collect_called_names("x = 1\n")
    assert names == set()


def test_collect_called_names_syntax_error():
    names = _collect_called_names("def f(: pass")
    assert names == set()


def test_collect_called_names_other_callable():
    # func is a subscript (neither Name nor Attribute): funcs[0]()
    # Covers the elif-False branch in _collect_called_names.
    names = _collect_called_names("funcs[0]()\n")
    assert "funcs" not in names  # subscript call adds nothing


def test_no_source_no_analysis():
    de = DuplicateExtractor([(1, 5)])
    assert de._new_source is None
    assert de.get_rewritten_source() is None
