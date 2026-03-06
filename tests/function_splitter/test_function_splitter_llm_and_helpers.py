from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.refactors.function_splitter import (
    _FuncInfo,
    _SplitTask,
    _generate_call,
    _generate_helper_source,
    _llm_name_helpers,
)


def _make_mock_response(names_list):
    """Build a mock Anthropic message response for the name_helper_functions tool."""
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "name_helper_functions"
    mock_block.input = {
        "names": [{"id": str(i), "name": n} for i, n in enumerate(names_list)]
    }
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    return mock_response


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
