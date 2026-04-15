import textwrap
from crispen.refactors.duplicate_extractor import (
    _collect_ast_store_names,
    _collect_attribute_names,
    _collect_called_attr_names,
    _collect_called_names,
    _extract_defined_names,
    _has_call_to,
    _has_funcdef,
    _names_assigned_in,
    _names_in_edit_texts,
)


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
