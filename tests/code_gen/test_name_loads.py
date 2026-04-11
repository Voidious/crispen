from __future__ import annotations
import textwrap
from crispen.file_limiter.code_gen import (
    _collect_name_loads,
    _collect_name_stores,
    _rewrite_module_var_names,
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


def test_collect_name_stores_simple_assign():
    assert _collect_name_stores("X = 1\n") == {"X"}


def test_collect_name_stores_multiple_assigns():
    src = "X = 1\nY = 2\n"
    assert _collect_name_stores(src) == {"X", "Y"}


def test_collect_name_stores_augassign():
    assert _collect_name_stores("X += 1\n") == {"X"}


def test_collect_name_stores_annotated_assign_with_value():
    assert _collect_name_stores("X: int = 42\n") == {"X"}


def test_collect_name_stores_annotated_assign_without_value():
    # Declaration only (no assignment) — not a store.
    assert _collect_name_stores("X: int\n") == set()


def test_collect_name_stores_function_body_not_included():
    # Assignments inside function bodies are not module-level stores.
    src = "def f():\n    X = 1\n"
    assert _collect_name_stores(src) == set()


def test_collect_name_stores_load_not_included():
    assert _collect_name_stores("y = X\n") == {"y"}
    assert "X" not in _collect_name_stores("y = X\n")


def test_collect_name_stores_syntax_error():
    assert _collect_name_stores("def (broken:\n") == set()


def test_collect_name_stores_empty():
    assert _collect_name_stores("") == set()


def test_collect_name_stores_non_name_assign_target():
    # Tuple-unpacking targets are not plain Name nodes — must not crash.
    src = "a, b = 1, 2\n"
    result = _collect_name_stores(src)
    assert "a" not in result  # tuple target, not a plain Name store
    assert "b" not in result


def test_collect_name_stores_non_name_augassign_target():
    # Attribute augmented assignment — target is Attribute, not Name.
    src = "obj.x += 1\n"
    result = _collect_name_stores(src)
    assert result == set()


def test_rewrite_module_var_names_basic():
    src = "def fn():\n    if SAFE_MODE:\n        pass\n"
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert "conversion.SAFE_MODE" in result
    # bare SAFE_MODE no longer appears as a standalone Name
    import ast

    tree = ast.parse(result)
    bare = [
        n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "SAFE_MODE"
    ]
    assert bare == []


def test_rewrite_module_var_names_skips_attribute_access():
    # obj.SAFE_MODE must NOT become obj.conversion.SAFE_MODE — the regex approach
    # would corrupt this; the AST approach correctly skips it because 'SAFE_MODE'
    # is the attr string of an Attribute node, not an ast.Name load.
    src = "def fn():\n    return obj.SAFE_MODE\n"
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert result == src


def test_rewrite_module_var_names_skips_strings():
    src = 'x = "SAFE_MODE"\n'
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert result == src


def test_rewrite_module_var_names_skips_comments():
    src = "# use SAFE_MODE here\nx = 1\n"
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert result == src


def test_rewrite_module_var_names_no_partial_name_match():
    # SAFE_MODE_EXTRA is a different identifier and must not be rewritten
    src = "x = SAFE_MODE_EXTRA\ny = SAFE_MODE\n"
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert "SAFE_MODE_EXTRA" in result
    assert "y = conversion.SAFE_MODE" in result


def test_rewrite_module_var_names_empty_rewrites():
    src = "def fn():\n    return SAFE_MODE\n"
    result = _rewrite_module_var_names(src, {})
    assert result == src


def test_rewrite_module_var_names_initial_syntax_error_returns_original():
    # Unparseable source at the start → return unchanged (first ast.parse fails)
    src = "def fn(\n"
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert result == src


def test_rewrite_module_var_names_no_name_nodes_returns_original():
    # Source has no Name nodes for the given key → return unchanged
    src = "x = 1\n"
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert result == src


def test_rewrite_module_var_names_verify_bare_name_survives_returns_original():
    # If a rewrite introduces a new bare Name that itself appears in rewrites,
    # verification catches it and returns the original source.
    # rewrites={"A": "mod.A", "mod": "pkg.mod"}: rewriting "A" → "mod.A" leaves
    # "mod" as a bare Name load, which is in rewrites → verification fails.
    src = "x = A\n"
    result = _rewrite_module_var_names(src, {"A": "mod.A", "mod": "pkg.mod"})
    assert result == src


def test_rewrite_module_var_names_verify_syntax_error_returns_original(monkeypatch):
    # If re-parsing the rewritten result raises SyntaxError (defensive guard),
    # the original source is returned unchanged.
    import crispen.file_limiter.code_gen as _code_gen
    import ast as _ast

    call_count = [0]
    real_parse = _ast.parse

    def patched_parse(src, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] >= 2:  # fail on the verification parse
            raise SyntaxError("synthetic verify failure")
        return real_parse(src, *args, **kwargs)

    monkeypatch.setattr(_code_gen.ast, "parse", patched_parse)
    src = "x = SAFE_MODE\n"
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert result == src
