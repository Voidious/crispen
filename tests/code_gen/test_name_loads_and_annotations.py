from __future__ import annotations
import textwrap
from crispen.file_limiter.code_gen import (
    _collect_name_loads,
    _collect_quoted_annotation_names,
)


def test_collect_name_loads_basic():
    source = "x = foo + bar"
    names = _collect_name_loads(source)
    assert "foo" in names
    assert "bar" in names


def test_collect_name_loads_store_not_included():
    source = "x = 1"
    names = _collect_name_loads(source)
    # x is a Store, not a Load
    assert "x" not in names


def test_collect_name_loads_syntax_error():
    assert _collect_name_loads("def (invalid") == set()


def test_collect_name_loads_excludes_function_params():
    # 'client' is a parameter of test_foo — excluded from loads inside the body.
    source = "def test_foo(client):\n    client.call()\n"
    names = _collect_name_loads(source)
    assert "client" not in names


def test_collect_name_loads_includes_non_param_name():
    # 'helper' is not a parameter of test_foo — still counted as a load.
    source = "def test_foo(client):\n    helper(client)\n"
    names = _collect_name_loads(source)
    assert "helper" in names
    assert "client" not in names


def test_collect_name_loads_excludes_nested_function_params():
    # Inner function params are excluded only within that function's own body.
    source = textwrap.dedent(
        """\
        def outer(x):
            def inner(y):
                return y + x
            return inner
        """
    )
    names = _collect_name_loads(source)
    assert "y" not in names  # param of inner — excluded inside inner body
    assert "x" not in names  # param of outer — excluded inside outer body


def test_collect_name_loads_includes_annotation_names():
    # Type annotations are in the outer scope — their names are included.
    source = "def f(x: MyType) -> ReturnType:\n    pass\n"
    names = _collect_name_loads(source)
    assert "MyType" in names
    assert "ReturnType" in names
    assert "x" not in names  # param name itself, not counted


def test_collect_name_loads_includes_decorator_names():
    # Decorator expressions are in the outer scope.
    source = "@pytest.fixture\ndef client():\n    pass\n"
    names = _collect_name_loads(source)
    assert "pytest" in names


def test_collect_name_loads_kw_defaults_none_skipped():
    # kw_defaults may contain None for keyword-only args without defaults.
    # None entries must not cause a crash and are simply skipped.
    source = "def f(*, a, b=DEFAULT):\n    pass\n"
    names = _collect_name_loads(source)
    assert "DEFAULT" in names
    assert "a" not in names
    assert "b" not in names


def test_collect_name_loads_annotated_vararg_kwarg():
    # *args: T and **kwargs: T annotations are in the outer scope.
    source = "def f(*args: VarType, **kwargs: KwType):\n    pass\n"
    names = _collect_name_loads(source)
    assert "VarType" in names
    assert "KwType" in names


def test_collect_name_loads_excludes_local_variable_assignments():
    # A name assigned in the function body is a local variable — not an import.
    # Loads of that name (e.g. attribute access) must not generate cross-file imports.
    source = textwrap.dedent(
        """\
        def test_foo(tmp_path):
            helpers = tmp_path / "helpers.py"
            helpers.write_text("X = 1", encoding="utf-8")
            assert str(helpers.resolve()) == "x"
        """
    )
    names = _collect_name_loads(source)
    assert "helpers" not in names
    assert "tmp_path" not in names  # also a param — still excluded


def test_collect_name_loads_local_store_does_not_suppress_outer_loads():
    # A local assignment in an inner function must not suppress the outer scope's load.
    source = textwrap.dedent(
        """\
        def outer():
            use(helper)
            def inner():
                helper = 1
                use(helper)
        """
    )
    names = _collect_name_loads(source)
    # outer() loads 'helper' (not locally defined there); inner() assigns it locally.
    assert "helper" in names


def test_collect_quoted_annotation_names_basic():
    # "MyType" in a string annotation → detected.
    source = 'def f(x: "MyType") -> None:\n    pass\n'
    names = _collect_quoted_annotation_names(source)
    assert "MyType" in names


def test_collect_quoted_annotation_names_optional():
    # Optional["_LLMAccumulator"] — the inner string is parsed.
    source = 'def f(x: Optional["_LLMAccumulator"]) -> None:\n    pass\n'
    names = _collect_quoted_annotation_names(source)
    assert "_LLMAccumulator" in names


def test_collect_quoted_annotation_names_return():
    # Quoted return annotation.
    source = 'def f() -> "ReturnType":\n    pass\n'
    names = _collect_quoted_annotation_names(source)
    assert "ReturnType" in names


def test_collect_quoted_annotation_names_annassign():
    # Variable annotation: x: "MyClass"
    source = 'x: "MyClass"\n'
    names = _collect_quoted_annotation_names(source)
    assert "MyClass" in names


def test_collect_quoted_annotation_names_unquoted_not_included():
    # Normal (unquoted) annotation names are NOT returned by this function.
    source = "def f(x: MyType) -> None:\n    pass\n"
    names = _collect_quoted_annotation_names(source)
    assert "MyType" not in names


def test_collect_quoted_annotation_names_syntax_error():
    # Unparseable source returns empty set (no crash).
    assert _collect_quoted_annotation_names("def (invalid") == set()


def test_collect_quoted_annotation_names_inner_syntax_error():
    # A string annotation that isn't valid Python is silently ignored.
    source = 'def f(x: "not valid python !!") -> None:\n    pass\n'
    names = _collect_quoted_annotation_names(source)
    assert names == set()


def test_collect_quoted_annotation_names_vararg_kwarg():
    # *args and **kwargs with quoted annotations.
    source = 'def f(*args: "VarType", **kwargs: "KwType") -> None:\n    pass\n'
    names = _collect_quoted_annotation_names(source)
    assert "VarType" in names
    assert "KwType" in names


def test_collect_quoted_annotation_names_annassign_with_value():
    # x: "MyClass" = SomeFactory() — annotation has quoted name AND there is a value.
    # The _walk branch for AnnAssign with node.value must execute.
    source = 'x: "MyClass" = object()\n'
    names = _collect_quoted_annotation_names(source)
    assert "MyClass" in names
