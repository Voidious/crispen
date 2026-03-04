from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.refactors.function_splitter import _llm_name_helpers
from tests.fs_llm_name_helper_runner_v2 import _run_llm_name_helpers
from tests.fs_mock_llm_response_factory_v2 import _make_mock_response
from tests.test_fs_llm_name_helpers_fixtures import _make_task


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
    result = _run_llm_name_helpers(mock_anthropic, mock_response, "my_func")
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
    result = _run_llm_name_helpers(mock_anthropic, mock_response, "my_func")
    assert result == ["my_func_helper"]
