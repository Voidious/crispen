import textwrap
from crispen.refactors.duplicate_extractor import (
    _FunctionInfo,
    _build_function_body_fps,
    _collect_ast_store_names,
    _collect_attribute_names,
    _collect_called_attr_names,
    _collect_called_names,
    _extract_defined_names,
    _has_call_to,
    _has_funcdef,
    _names_assigned_in,
    _normalize_source,
    _strip_helper_docstring,
    _strip_unused_call_assignments,
)


def test_normalize_source_normalizes_vars():
    src = "result = compute(data)\noutput = transform(result)\n"
    norm = _normalize_source(src)
    # All names (both assigned and free) are replaced with positional placeholders
    assert "result" not in norm
    assert "output" not in norm
    assert "compute" not in norm
    assert "data" not in norm


def test_normalize_source_same_fingerprint():
    src_a = "x = compute(data)\ny = transform(x)\n"
    src_b = "val = compute(data)\nres = transform(val)\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


def test_normalize_source_different_ops():
    # Structurally different code (different number of statements) should differ
    src_a = "x = a + b\n"
    src_b = "x = a + b\ny = x * 2\n"
    assert _normalize_source(src_a) != _normalize_source(src_b)


def test_normalize_source_invalid_syntax():
    src = "def f(: pass"
    # Falls back to original source
    assert _normalize_source(src) == src


def test_normalize_source_load_context_replaced():
    # Var assigned then used: both should be normalized the same
    src_a = "x = 1\ny = x + 1\n"
    src_b = "a = 1\nb = a + 1\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


def test_normalize_source_load_not_in_map():
    # Free variables (Load context, never stored) are also normalized,
    # so two blocks with different free variable names get the same fingerprint.
    src_a = "y = a + 1\n"
    src_b = "z = b + 1\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


def test_normalize_source_repeated_store():
    # Same name assigned twice: _placeholder called with cached key (False branch)
    src = "x = 1\nx = 2\n"
    norm = _normalize_source(src)
    # Both assignments normalize to the same placeholder
    assert norm.count("_v0") == 2


def test_normalize_source_del_context():
    # Del context falls through to return node unchanged
    src = "del x\n"
    norm = _normalize_source(src)
    assert "x" in norm


def test_normalize_source_free_variables_match():
    # Blocks differing only in free variable names should get the same fingerprint.
    # This is the core case: `p = a * 2; if p > 100: p += 1` vs the same with q/b.
    src_a = "p = a * 2\nif p > 100:\n    p += 1\n"
    src_b = "q = b * 2\nif q > 100:\n    q += 1\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


def test_normalize_source_indented_blocks_match():
    # Source collected from inside a function is indented; dedent must happen
    # before ast.parse so that structurally identical blocks still match.
    src_a = "    p = a * 2\n    if p > 100:\n        p += 1\n"
    src_b = "    q = b * 2\n    if q > 100:\n        q += 1\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


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


def test_has_funcdef_present():
    assert _has_funcdef("_helper", "def _helper(x):\n    pass\n") is True


def test_has_funcdef_async():
    assert _has_funcdef("_helper", "async def _helper(x):\n    pass\n") is True


def test_has_funcdef_missing():
    assert _has_funcdef("_helper", "def other(x):\n    pass\n") is False


def test_has_funcdef_syntax_error():
    assert _has_funcdef("_helper", "def f(x:") is False


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


def _make_func_info(name: str, body_source: str = "    pass\n") -> _FunctionInfo:
    return _FunctionInfo(
        name=name,
        source=f"def {name}():\n{body_source}",
        scope="<module>",
        body_source=body_source,
        body_stmt_count=1,
        params=[],
    )


def test_build_fps_includes_called():
    body = "    x = 1\n    y = 2\n    z = 3\n"
    func = _make_func_info("foo", body)
    fps = _build_function_body_fps([func], {"foo"})
    fp = _normalize_source(body)
    assert fp in fps
    assert fps[fp].name == "foo"


def test_build_fps_excludes_uncalled():
    func = _make_func_info("bar")
    fps = _build_function_body_fps([func], {"foo"})
    assert fps == {}


def test_build_fps_empty_functions():
    fps = _build_function_body_fps([], {"foo"})
    assert fps == {}


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


def test_collect_ast_store_names_simple_name():
    import ast

    node = ast.parse("x = 1").body[0].targets[0]
    names: list = []
    _collect_ast_store_names(node, names)
    assert names == ["x"]


def test_collect_ast_store_names_tuple():
    import ast

    node = ast.parse("a, b = 1, 2").body[0].targets[0]
    names: list = []
    _collect_ast_store_names(node, names)
    assert set(names) == {"a", "b"}


def test_collect_ast_store_names_nested_tuple():
    import ast

    node = ast.parse("(a, (b, c)) = x").body[0].targets[0]
    names: list = []
    _collect_ast_store_names(node, names)
    assert set(names) == {"a", "b", "c"}


def test_collect_ast_store_names_non_name_non_tuple_noop():
    # ast.Attribute target (e.g. self.x) → nothing collected.
    import ast

    node = ast.parse("self.x = 1").body[0].targets[0]
    names: list = []
    _collect_ast_store_names(node, names)
    assert names == []


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
