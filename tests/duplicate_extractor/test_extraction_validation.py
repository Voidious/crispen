from crispen.refactors.duplicate_extractor import (
    _collect_attribute_names,
    _collect_called_attr_names,
    _collect_called_names,
    _has_call_to,
    _has_funcdef,
    _has_mutable_literal_is_check,
    _is_pure_literal,
    _verify_extraction,
)


def test_verify_extraction_valid():
    helper = "def helper(x):\n    return x + 1\n"
    replacements = ["result = helper(a)\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_invalid_helper():
    helper = "def helper(x:\n    pass\n"  # unclosed paren → syntax error after dedent
    replacements = ["result = helper(a)\n"]
    assert _verify_extraction(helper, replacements) is False


def test_verify_extraction_invalid_replacement():
    helper = "def helper(x):\n    return x\n"
    # Dedented replacement still has a syntax error
    replacements = ["result = helper(a\n"]  # unclosed paren
    assert _verify_extraction(helper, replacements) is False


def test_verify_extraction_no_helper_source():
    # Exercises the helper_source is None branch (skips helper compile check).
    assert _verify_extraction(None, ["result = f()\n"]) is True


def test_verify_extraction_fails_on_param_overwrite():
    # Helper where the parameter is immediately overwritten before being read.
    helper = "def setup(mock_obj):\n    mock_obj = object()\n    return mock_obj\n"
    assert _verify_extraction(helper, ["x = setup(y)\n"]) is False


def test_verify_extraction_allows_return_in_replacement():
    # Replacements inside function bodies legally contain 'return'; the dummy-
    # function wrapper must allow this without triggering a false rejection.
    helper = "def helper(x):\n    return x\n"
    replacements = ["    return helper(a)\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_multiline_return_replacement():
    # Multi-line replacement ending with a return statement.
    helper = "def helper(source):\n    return helper(source)\n"
    replacements = [
        "    tree = helper(source)\n    if tree is None:\n        return set()\n"
    ]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_continue_in_replacement():
    # 'continue' is valid inside a loop body; the dummy wrapper now includes a
    # for loop so this is not rejected as a SyntaxError.
    helper = "def helper():\n    pass\n"
    replacements = ["    if done:\n        continue\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_break_in_replacement():
    # Same as above but for 'break'.
    helper = "def helper():\n    pass\n"
    replacements = ["    if done:\n        break\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_await_in_replacement():
    # Replacements inside async functions legally contain 'await'; the async
    # dummy-function wrapper must allow this without triggering a false rejection.
    helper = "async def helper(x):\n    return await x\n"
    replacements = ["    result = await helper(coro)\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_async_helper():
    # async def helpers are valid Python and must compile successfully.
    helper = "async def helper(client, x):\n    return await client.get(x)\n"
    replacements = ["    val = await helper(client, url)\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_rejects_invalid_await_replacement():
    # Replacement with `await` that also has a real syntax error must still fail.
    helper = "async def helper(x):\n    return await x\n"
    replacements = ["    result = await helper(coro\n"]  # unclosed paren
    assert _verify_extraction(helper, replacements) is False


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


def test_has_funcdef_present():
    assert _has_funcdef("_helper", "def _helper(x):\n    pass\n") is True


def test_has_funcdef_async():
    assert _has_funcdef("_helper", "async def _helper(x):\n    pass\n") is True


def test_has_funcdef_missing():
    assert _has_funcdef("_helper", "def other(x):\n    pass\n") is False


def test_has_funcdef_syntax_error():
    assert _has_funcdef("_helper", "def f(x:") is False


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
