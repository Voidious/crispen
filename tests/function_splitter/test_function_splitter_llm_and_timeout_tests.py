from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from crispen.refactors.function_splitter import (
    _ApiTimeout,
    _FuncInfo,
    _SplitTask,
    _llm_name_helpers,
    _run_with_timeout,
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


def test_run_with_timeout_success():
    result = _run_with_timeout(lambda x: x * 2, 5, 21)
    assert result == 42


def test_run_with_timeout_exceeds():
    import time

    with pytest.raises(_ApiTimeout):
        _run_with_timeout(lambda: time.sleep(10), timeout=0.05)


def test_run_with_timeout_propagates_exception():
    def _raise():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        _run_with_timeout(_raise, 5)


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
