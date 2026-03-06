from __future__ import annotations
import textwrap
from unittest.mock import patch
import libcst as cst
import pytest
from libcst.metadata import MetadataWrapper, PositionProvider
from crispen.refactors.function_splitter import (
    _ApiTimeout,
    _choose_best_split,
    _find_valid_splits,
    _generate_call,
    _generate_helper_source,
    _has_nested_funcdef,
    _has_yield,
    _head_effective_lines,
    _module_global_names,
    _run_with_timeout,
    _stmts_source,
    FunctionSplitter,
)
from .test_name_helpers_and_helpers import _make_mock_response


def _parse_func(source: str):
    """Return (body_stmts, positions, source_lines) for the first function.

    Uses a CSTVisitor to capture body_stmts from the wrapper's internal copy,
    ensuring they match the keys in the positions dict.
    """
    tree = cst.parse_module(source)
    wrapper = MetadataWrapper(tree)
    positions = wrapper.resolve(PositionProvider)

    class _Getter(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self):
            self.stmts: list = []

        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            if not self.stmts:  # first function only
                self.stmts = list(node.body.body)

    getter = _Getter()
    wrapper.visit(getter)
    source_lines = source.splitlines(keepends=True)
    return getter.stmts, positions, source_lines


def test_stmts_source_basic():
    src = "def foo():\n    x = 1\n    y = 2\n    z = 3\n"
    stmts, positions, lines = _parse_func(src)
    result = _stmts_source(stmts[:2], lines, positions)
    assert "x = 1" in result
    assert "y = 2" in result
    assert "z = 3" not in result


def test_stmts_source_empty():
    src = "def foo():\n    x = 1\n"
    _, positions, lines = _parse_func(src)
    assert _stmts_source([], lines, positions) == ""


def test_stmts_source_dedented():
    src = "def foo():\n    x = 1\n    y = 2\n"
    stmts, positions, lines = _parse_func(src)
    result = _stmts_source(stmts, lines, positions)
    # Should be dedented (no leading 4-space indent)
    assert result.startswith("x = 1") or result.startswith("x = 1\n")


def test_head_effective_lines_no_docstring():
    src = "def foo():\n    x = 1\n    y = 2\n    z = 3\n"
    stmts, positions, lines = _parse_func(src)
    # split_idx=2: head=[x,y], last=y at line 3, first=x at line 2 → 3-2+2=3
    result = _head_effective_lines(stmts, 2, positions, False)
    assert result == 3


def test_head_effective_lines_with_docstring_normal():
    src = 'def foo():\n    """doc"""\n    x = 1\n    y = 2\n    z = 3\n'
    stmts, positions, lines = _parse_func(src)
    # split_idx=3: head=[doc, x, y], first_non_doc=x at line 3, last=y at line 4
    # 4-3+2=3
    result = _head_effective_lines(stmts, 3, positions, True)
    assert result == 3


def test_head_effective_lines_only_docstring_in_head():
    # split_idx=1 with docstring: first_non_doc_idx=1 >= split_idx=1 → returns 1
    src = 'def foo():\n    """doc"""\n    x = 1\n    y = 2\n'
    stmts, positions, lines = _parse_func(src)
    result = _head_effective_lines(stmts, 1, positions, True)
    assert result == 1


def test_find_valid_splits_all_valid():
    src = "def foo():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n"
    stmts, positions, lines = _parse_func(src)
    # With a very loose limit, all splits should be valid
    result = _find_valid_splits(stmts, positions, max_lines=1000)
    assert len(result) > 0
    # Ordered latest first
    assert result == sorted(result, reverse=True)


def test_find_valid_splits_none_valid():
    # max_lines=1 means even a 1-stmt head (+ return call = 2 lines) is invalid
    src = "def foo():\n    a = 1\n    b = 2\n    c = 3\n"
    stmts, positions, lines = _parse_func(src)
    result = _find_valid_splits(stmts, positions, max_lines=1)
    assert result == []


def test_find_valid_splits_stops_at_max_candidates():
    # 7 statements → iterates from 6 down, stops after 5 valid candidates
    src = "def foo():\n" + "".join(f"    a{i} = {i}\n" for i in range(7))
    stmts, positions, lines = _parse_func(src)
    result = _find_valid_splits(stmts, positions, max_lines=1000)
    assert len(result) == 5


def test_find_valid_splits_fewer_than_max():
    # 4 statements → at most 3 valid splits (indices 3, 2, 1)
    src = "def foo():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n"
    stmts, positions, lines = _parse_func(src)
    result = _find_valid_splits(stmts, positions, max_lines=1000)
    assert 1 <= len(result) <= 3


def test_find_valid_splits_empty_body():
    # Should not crash with an empty list (though normally not called)
    result = _find_valid_splits([], {}, max_lines=1000)
    assert result == []


def test_find_valid_splits_nested_funcdef_restricts_upper():
    # First nested funcdef at index 2 → valid splits only at indices ≤ 2.
    src = textwrap.dedent(
        """\
        def outer():
            a = 1
            b = 2
            def inner():
                pass
            c = 3
            d = 4
    """
    )
    stmts, positions, lines = _parse_func(src)
    # body_stmts: [a=1, b=2, def inner, c=3, d=4]
    # First nested funcdef at index 2 → upper=2 → range(2, 0, -1) = [2, 1]
    result = _find_valid_splits(stmts, positions, max_lines=1000)
    assert all(i <= 2 for i in result)
    assert 3 not in result
    assert 4 not in result


def test_choose_best_split_fewest_params():
    # Two splits: one has free vars, one doesn't
    src = textwrap.dedent(
        """\
        def foo(external):
            a = 1
            b = external + 1
    """
    )
    stmts, positions, lines = _parse_func(src)
    # split_idx=1: tail=[b=external+1] → free vars: [external]
    # split_idx=2: tail=[] → but we need at least 1 stmt in tail,
    # so valid splits are [1] only for 2-stmt function
    # Let's use 3 stmts with different free var counts
    src2 = textwrap.dedent(
        """\
        def foo(ext):
            a = 1
            b = ext + 1
            c = a + b
    """
    )
    stmts2, positions2, lines2 = _parse_func(src2)
    # split_idx=1: tail=[b=ext+1, c=a+b] → free vars: [a, ext] (a from head)
    # Actually 'a' is assigned in head (split_idx=1 → head=[a=1]) and used in tail
    # So tail [b=ext+1, c=a+b] has free vars: [a, ext]
    # split_idx=2: tail=[c=a+b] → free vars: [a, b] (assigned in head)
    # Wait no, head=[a=1, b=ext+1] so tail=[c=a+b] has free vars: [a, b]
    # split_idx=3: not valid (needs at least 1 in tail)
    # So split_idx=1 has 2 free vars [a, ext], split_idx=2 has 2 free vars [a, b]
    # Tie → choose earliest in list = latest split = 2
    valid_splits = [2, 1]  # latest first
    split_idx, params, _ = _choose_best_split(
        stmts2, valid_splits, lines2, positions2, ["ext"]
    )
    # Both have 2 free vars, tie broken by latest (first in list) = 2
    assert split_idx == 2


def test_choose_best_split_fewer_params_wins():
    # Use a source where one split clearly has fewer params
    src = textwrap.dedent(
        """\
        def foo():
            a = 1
            b = 2
            c = a + b
    """
    )
    stmts, positions, lines = _parse_func(src)
    # split_idx=1: tail=[b=2, c=a+b] → free vars: [a] (1 free var)
    # split_idx=2: tail=[c=a+b] → free vars: [a, b] (2 free vars)
    valid_splits = [2, 1]
    split_idx, params, _ = _choose_best_split(stmts, valid_splits, lines, positions, [])
    # split_idx=1 has 1 free var (a) vs split_idx=2 has 2 free vars (a, b)
    assert split_idx == 1
    assert params == ["a"]


def test_choose_best_split_single_candidate():
    src = "def foo():\n    x = 1\n    y = 2\n"
    stmts, positions, lines = _parse_func(src)
    split_idx, params, _ = _choose_best_split(stmts, [1], lines, positions, [])
    assert split_idx == 1


def test_choose_best_split_self_in_tail_returns_instance_method():
    # Tail requires self → extracted as instance method, not static
    src = textwrap.dedent(
        """\
        class Foo:
            def method(self, x):
                a = 1
                b = self.value + a
        """
    )
    stmts, positions, lines = _parse_func(src)
    # split_idx=1: tail=[b = self.value + a] → free: [a, self] → instance method
    result = _choose_best_split(stmts, [1], lines, positions, ["self", "x"])
    assert result is not None
    split_idx, params, is_instance_method = result
    assert split_idx == 1
    assert is_instance_method is True
    assert "self" not in params  # self is implicit, not in params list
    assert "a" in params  # a is still a real param


def test_choose_best_split_empty_splits_returns_none():
    # No valid split candidates → None returned
    src = "def foo():\n    x = 1\n    y = 2\n"
    stmts, positions, lines = _parse_func(src)
    result = _choose_best_split(stmts, [], lines, positions, [])
    assert result is None


def test_choose_best_split_filters_module_globals():
    # Tail references a module-level import; it must not appear in params.
    src = textwrap.dedent(
        """\
        def foo():
            x = 1
            y = os.path.join("a", "b")
        """
    )
    stmts, positions, lines = _parse_func(src)
    # Without filtering: "os" would be a free var of the tail.
    # With module_globals={"os"}: "os" is filtered out → params = []
    result = _choose_best_split(stmts, [1], lines, positions, [], module_globals={"os"})
    assert result is not None
    _, params, _ = result
    assert "os" not in params


def test_module_global_names_imports():
    source = "import ast\nfrom pathlib import Path\nimport libcst as cst\n"
    result = _module_global_names(source)
    assert "ast" in result
    assert "Path" in result
    assert "cst" in result


def test_module_global_names_functions_and_classes():
    source = "def foo():\n    pass\n\nclass Bar:\n    pass\n"
    result = _module_global_names(source)
    assert "foo" in result
    assert "Bar" in result


def test_module_global_names_assignments():
    source = "_CONST = frozenset()\nVALUE: int = 42\n"
    result = _module_global_names(source)
    assert "_CONST" in result
    assert "VALUE" in result


def test_module_global_names_syntax_error():
    result = _module_global_names("def foo(")
    assert result == set()


def test_module_global_names_tuple_assign_target_not_collected():
    # Tuple-unpacking: Assign target is a Tuple node, not a Name → skipped
    source = "a, b = 1, 2\n"
    result = _module_global_names(source)
    assert "a" not in result
    assert "b" not in result


def test_module_global_names_ann_assign_non_name_target_skipped():
    # AnnAssign where target is an Attribute, not a Name → skipped
    source = "Foo.x: int\n"
    result = _module_global_names(source)
    assert "x" not in result


def test_generate_helper_source_with_staticmethod():
    result = _generate_helper_source(
        name="process",
        params=["x", "y"],
        tail_source="return x + y\n",
        func_indent="    ",
        is_static=True,
        add_docstring=False,
    )
    assert "@staticmethod" in result
    assert "def _process(x, y):" in result
    assert "return x + y" in result
    assert result.startswith("    @staticmethod")


def test_generate_helper_source_without_staticmethod():
    result = _generate_helper_source(
        name="process",
        params=["x"],
        tail_source="return x * 2\n",
        func_indent="",
        is_static=False,
        add_docstring=False,
    )
    assert "@staticmethod" not in result
    assert "def _process(x):" in result
    assert "return x * 2" in result


def test_generate_helper_source_with_docstring():
    result = _generate_helper_source(
        name="process",
        params=[],
        tail_source="return 42\n",
        func_indent="",
        is_static=False,
        add_docstring=True,
    )
    assert '"""' in result
    assert "return 42" in result


def test_generate_helper_source_instance_method():
    result = _generate_helper_source(
        name="process",
        params=["a"],
        tail_source="return self.x + a\n",
        func_indent="    ",
        is_static=False,
        add_docstring=False,
        is_instance_method=True,
    )
    assert "@staticmethod" not in result
    assert "def _process(self, a):" in result
    assert "return self.x + a" in result


def test_generate_helper_source_indentation_correct():
    result = _generate_helper_source(
        name="helper",
        params=[],
        tail_source="x = 1\ny = 2\n",
        func_indent="    ",
        is_static=False,
        add_docstring=False,
    )
    # Body should be indented by 8 spaces (func_indent=4 + body_indent=4)
    assert "        x = 1" in result
    assert "        y = 2" in result


def test_generate_call_with_class():
    result = _generate_call("helper", ["x", "y"], "MyClass", "    ")
    assert result == "    return MyClass._helper(x, y)"


def test_generate_call_module_level():
    result = _generate_call("helper", ["a"], None, "        ")
    assert result == "        return _helper(a)"


def test_generate_call_no_params():
    result = _generate_call("do_work", [], None, "    ")
    assert result == "    return _do_work()"


def test_generate_call_class_no_params():
    result = _generate_call("do_work", [], "Foo", "    ")
    assert result == "    return Foo._do_work()"


def test_generate_call_instance_method():
    result = _generate_call("process", ["a", "b"], "MyClass", "    ", True)
    assert result == "    return self._process(a, b)"


def test_generate_call_instance_method_no_params():
    result = _generate_call("process", [], "MyClass", "    ", True)
    assert result == "    return self._process()"


def test_has_yield_simple():
    src = "def gen():\n    yield 1\n"
    func = cst.parse_module(src).body[0]
    assert _has_yield(func) is True


def test_has_yield_from():
    src = "def gen():\n    yield from [1, 2]\n"
    func = cst.parse_module(src).body[0]
    assert _has_yield(func) is True


def test_has_yield_none():
    src = "def foo():\n    return 1\n"
    func = cst.parse_module(src).body[0]
    assert _has_yield(func) is False


def test_has_yield_nested_not_counted():
    src = textwrap.dedent(
        """\
        def foo():
            def inner():
                yield 1
            return inner
    """
    )
    func = cst.parse_module(src).body[0]
    # yield is inside nested function, should not count
    assert _has_yield(func) is False


def test_has_nested_funcdef_with_nested():
    src = textwrap.dedent(
        """\
        def outer():
            x = 1
            def inner():
                return x
            return inner
    """
    )
    func = cst.parse_module(src).body[0]
    assert _has_nested_funcdef(func) is True


def test_has_nested_funcdef_without_nested():
    src = "def foo():\n    x = 1\n    return x\n"
    func = cst.parse_module(src).body[0]
    assert _has_nested_funcdef(func) is False


def test_has_nested_funcdef_first_stmt():
    # Nested funcdef is the very first statement in the body
    src = textwrap.dedent(
        """\
        def outer():
            def inner():
                pass
            return inner()
    """
    )
    func = cst.parse_module(src).body[0]
    assert _has_nested_funcdef(func) is True


def test_run_with_timeout_success():
    result = _run_with_timeout(lambda x: x * 2, 5, 21)
    assert result == 42


def test_run_with_timeout_exceeds():
    import time

    with pytest.raises(_ApiTimeout):
        _run_with_timeout(lambda: time.sleep(10), timeout=0.05)


def _make_long_func(n_stmts: int, func_name: str = "long_func") -> str:
    """Build a function with n_stmts independent assignments."""
    lines = [f"def {func_name}():\n"]
    for i in range(n_stmts):
        lines.append(f"    a{i} = {i}\n")
    lines.append("    return 0\n")
    return "".join(lines)


def test_function_splitter_under_limits_no_op():
    # A small function should not be split
    src = "def small():\n    x = 1\n    return x\n"
    splitter = FunctionSplitter([(1, 10)], source=src, verbose=False)
    assert splitter.get_rewritten_source() is None


def test_function_splitter_parse_error_no_crash():
    # Invalid source should not crash
    splitter = FunctionSplitter([(1, 10)], source="def f(\n  !!invalid", verbose=False)
    assert splitter.get_rewritten_source() is None


def test_function_splitter_out_of_range_no_op():
    # Function exists but is outside changed ranges
    src = _make_long_func(80)
    splitter = FunctionSplitter([(200, 300)], source=src, verbose=False, max_lines=10)
    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_over_line_limit(mock_anthropic):
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["process_tail"])
    )
    src = _make_long_func(80)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],
            source=src,
            verbose=False,
            max_lines=50,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    assert "_process_tail" in result
    assert "return _process_tail(" in result
    assert len(splitter.changes_made) >= 1


@patch("crispen.llm_client.anthropic")
def test_function_splitter_nested_funcdef_not_split(mock_anthropic):
    # A long function containing a nested funcdef should never be split,
    # even if it far exceeds the line limit.  Splitting across a closure
    # boundary produces cascading re-splits and semantically fragile helpers.
    lines = ["def func_with_closure():\n"]
    for i in range(80):
        lines.append(f"    a{i} = {i}\n")
    lines.append("    def inner():\n")
    lines.append("        return 0\n")
    lines.append("    return inner()\n")
    src = "".join(lines)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=10
        )

    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_async_skipped(mock_anthropic):
    # Async functions should not be split
    src = (
        "async def foo():\n"
        + "".join(f"    a{i} = {i}\n" for i in range(80))
        + "    return 0\n"
    )

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=10
        )

    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_generator_skipped(mock_anthropic):
    # Generator functions should not be split
    src = (
        "def gen():\n"
        + "".join(f"    a{i} = {i}\n" for i in range(80))
        + "    yield 0\n"
    )

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=10
        )

    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_llm_fallback_on_api_error(mock_anthropic):
    # API key not set → get_api_key raises CrispenAPIError → fallback names used
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["tail"])
    )
    src = _make_long_func(60, "my_func")

    # No ANTHROPIC_API_KEY → get_api_key raises → fallback to "my_func_helper"
    with patch.dict("os.environ", {}, clear=True):
        # Remove any existing API key
        import os

        os.environ.pop("ANTHROPIC_API_KEY", None)
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=30
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    # Fallback name used: "my_func_helper"
    assert "_my_func_helper" in result


@patch("crispen.llm_client.anthropic")
def test_function_splitter_recursive_split(mock_anthropic):
    # With small max_lines and broad changed_ranges, triggers multiple iterations
    # First call names helper for first function, second call for helper
    mock_anthropic.Anthropic.return_value.messages.create.side_effect = [
        _make_mock_response(["part1"]),
        _make_mock_response(["part2"]),
        _make_mock_response(["part3"]),
    ]

    # 13 body statements → with max_lines=5, needs multiple splits
    src = _make_long_func(13, "func")

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],  # broad range covers all helpers too
            source=src,
            verbose=False,
            max_lines=5,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    # Multiple splits occurred
    assert len(splitter.changes_made) >= 2


@patch("crispen.llm_client.anthropic")
def test_function_splitter_syntax_error_in_output_is_skipped(mock_anthropic):
    # If the assembled edit is invalid Python, the change is not applied
    # We simulate this by making _generate_call return something invalid
    # Instead, test the path via a function with 1-stmt body (no valid split)
    src = "def foo():\n    x = 1\n"  # only 1 stmt → can't split
    splitter = FunctionSplitter([(1, 10)], source=src, verbose=False, max_lines=0)
    # body lines=1 > 0=max_lines → tries to split but len(body_stmts)=1 < 2 → skip
    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_no_valid_split_skipped(mock_anthropic):
    # max_lines=1 → even a head with 1 stmt (+return call=2) > max_lines=1
    # So no valid splits → no change
    src = _make_long_func(5, "foo")

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter([(1, 1000)], source=src, verbose=False, max_lines=1)

    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_with_helper_docstrings(mock_anthropic):
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["process"])
    )
    src = _make_long_func(80)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],
            source=src,
            verbose=False,
            max_lines=50,
            helper_docstrings=True,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    assert '"""' in result


@patch("crispen.llm_client.anthropic")
def test_function_splitter_class_method(mock_anthropic):
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["tail_work"])
    )
    lines = ["class Foo:\n", "    def method(self):\n"]
    for i in range(80):
        lines.append(f"        a{i} = {i}\n")
    lines.append("        return 0\n")
    src = "".join(lines)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],
            source=src,
            verbose=False,
            max_lines=50,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    # Class methods use staticmethod and ClassName._ call
    assert "@staticmethod" in result
    assert "Foo._tail_work(" in result


@patch("crispen.llm_client.anthropic")
def test_function_splitter_llm_timeout_fallback(mock_anthropic):
    # LLM call times out → fallback names

    mock_anthropic.Anthropic.return_value.messages.create.side_effect = _ApiTimeout(
        "timed out"
    )
    src = _make_long_func(60, "slow_func")

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],
            source=src,
            verbose=False,
            max_lines=30,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    # Fallback name "slow_func_helper" used
    assert "_slow_func_helper" in result


def test_function_splitter_empty_source():
    """FunctionSplitter created with no source does nothing."""
    splitter = FunctionSplitter([(1, 10)])
    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_max_iterations_loop_exhausted(mock_anthropic):
    """Loop runs to completion (no break) when max iterations reached."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["helper"])
    )
    src = _make_long_func(80, "foo")

    # Patch _MAX_SPLIT_ITERATIONS to 1 → loop runs exactly once without breaking
    # (break only occurs at START of next iteration when tasks=[])
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("crispen.refactors.function_splitter._MAX_SPLIT_ITERATIONS", 1):
            splitter = FunctionSplitter(
                [(1, 1000)], source=src, verbose=False, max_lines=50
            )

    result = splitter.get_rewritten_source()
    assert result is not None
    assert len(splitter.changes_made) == 1


@patch("crispen.llm_client.anthropic")
def test_function_splitter_syntax_error_in_generated_output(mock_anthropic):
    """If assembled output fails compile(), the change is not applied."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["helper"])
    )
    src = _make_long_func(80, "foo")

    import builtins as _builtins

    orig_compile = _builtins.compile

    def _selective_compile(source, filename, mode, *args, **kwargs):
        if filename == "<string>":
            raise SyntaxError("mocked error for test")
        return orig_compile(source, filename, mode, *args, **kwargs)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("builtins.compile", side_effect=_selective_compile):
            splitter = FunctionSplitter(
                [(1, 1000)], source=src, verbose=False, max_lines=50
            )

    assert splitter.get_rewritten_source() is None


@patch(
    "crispen.refactors.function_splitter._has_new_undefined_names", return_value=True
)
@patch("crispen.llm_client.anthropic")
def test_function_splitter_pyflakes_rejects_output(mock_anthropic, mock_has_undef):
    """If pyflakes detects new undefined names in output, the split is not applied."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["helper"])
    )
    src = _make_long_func(80, "foo")

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=50
        )

    # Pyflakes check returned True → split not applied
    assert splitter.get_rewritten_source() is None


def test_engine_includes_function_splitter_no_op(tmp_path):
    """FunctionSplitter is in _REFACTORS and runs without error for simple files."""
    from crispen.engine import run_engine
    from crispen.config import CrispenConfig

    py_file = tmp_path / "sample.py"
    py_file.write_text("def foo():\n    return 1\n")
    config = CrispenConfig(max_function_length=75)
    msgs = list(run_engine({str(py_file): [(1, 2)]}, verbose=False, config=config))
    # No split needed — no messages expected (or just no errors)
    assert all("FunctionSplitter" not in m for m in msgs)


@patch("crispen.llm_client.anthropic")
def test_function_splitter_method_self_needed_uses_instance_method(mock_anthropic):
    """When every tail needs self, split into a regular instance method helper."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["tail_work"])
    )
    lines = ["class Foo:\n", "    def method(self):\n"]
    for i in range(40):
        lines.append(f"        a{i} = self.val + {i}\n")
    lines.append("        return 0\n")
    src = "".join(lines)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=20
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    assert "@staticmethod" not in result
    assert "return self._tail_work(" in result
    assert "def _tail_work(self" in result


@patch("crispen.llm_client.anthropic")
def test_function_splitter_skips_name_collision(mock_anthropic):
    """Helper name colliding with an existing function causes the task to be dropped."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["helper"])  # would produce _helper
    )
    # _helper already exists; the LLM would name the extracted helper "helper"
    existing = "def _helper():\n    pass\n\n\n"
    src = existing + _make_long_func(80)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=50
        )

    # collision detected → task dropped → no rewrite
    assert splitter.get_rewritten_source() is None
