from __future__ import annotations
import textwrap
from unittest.mock import MagicMock, patch
from crispen.refactors.function_splitter import (
    _FuncInfo,
    _SplitTask,
    _choose_best_split,
    _find_valid_splits,
    _generate_call,
    _generate_helper_source,
    _llm_name_helpers,
)
from .test_analysis import _make_mock_response, _parse_func


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


def _make_task(func_name, params=None, tail_source="return 0\n"):
    """Create a minimal _SplitTask for testing _llm_name_helpers."""
    mock_node = MagicMock()
    mock_node.name.value = func_name
    fi = _FuncInfo(
        node=mock_node,
        start_line=1,
        end_line=5,
        class_name=None,
        indent="",
        original_params=[],
    )
    return _SplitTask(fi, 1, params or [], tail_source=tail_source)


@patch("crispen.llm_client.anthropic")
def test_llm_name_helpers_success(mock_anthropic):
    mock_response = _make_mock_response(["process_tail"])
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    tasks = [_make_task("my_func")]
    client = mock_anthropic.Anthropic.return_value
    result = _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", tasks)
    assert result == ["process_tail"]


@patch("crispen.llm_client.anthropic")
def test_llm_name_helpers_result_none(mock_anthropic):
    # LLM returns no tool use block
    mock_response = MagicMock()
    mock_response.content = []
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    tasks = [_make_task("my_func")]
    client = mock_anthropic.Anthropic.return_value
    result = _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", tasks)
    # Falls back to "my_func_helper"
    assert result == ["my_func_helper"]


@patch("crispen.llm_client.anthropic")
def test_llm_name_helpers_no_names_key(mock_anthropic):
    # LLM returns tool use but without "names" key
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "name_helper_functions"
    mock_block.input = {"something_else": []}
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    tasks = [_make_task("my_func")]
    client = mock_anthropic.Anthropic.return_value
    result = _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", tasks)
    assert result == ["my_func_helper"]


@patch("crispen.llm_client.anthropic")
def test_llm_name_helpers_strips_leading_underscore(mock_anthropic):
    mock_response = _make_mock_response(["__private_name"])
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    tasks = [_make_task("foo")]
    client = mock_anthropic.Anthropic.return_value
    result = _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", tasks)
    assert result == ["private_name"]


@patch("crispen.llm_client.anthropic")
def test_llm_name_helpers_all_underscores_uses_helper(mock_anthropic):
    mock_response = _make_mock_response(["___"])
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    tasks = [_make_task("foo")]
    client = mock_anthropic.Anthropic.return_value
    result = _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", tasks)
    assert result == ["helper"]


@patch("crispen.llm_client.anthropic")
def test_llm_name_helpers_bad_item_skipped(mock_anthropic):
    # One item has a TypeError (e.g. name is not a string)
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "name_helper_functions"
    mock_block.input = {
        "names": [{"id": "0", "name": None}]  # None.lstrip() raises AttributeError
    }
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    tasks = [_make_task("foo")]
    client = mock_anthropic.Anthropic.return_value
    result = _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", tasks)
    # Falls back to "foo_helper" because item had AttributeError
    assert result == ["foo_helper"]


@patch("crispen.llm_client.anthropic")
def test_llm_name_helpers_with_class_name(mock_anthropic):
    mock_response = _make_mock_response(["process"])
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    mock_node = MagicMock()
    mock_node.name.value = "method"
    fi = _FuncInfo(
        node=mock_node,
        start_line=1,
        end_line=5,
        class_name="MyClass",
        indent="    ",
        original_params=[],
    )
    task = _SplitTask(fi, 1, [], tail_source="return 0\n")
    client = mock_anthropic.Anthropic.return_value
    result = _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", [task])
    assert result == ["process"]
