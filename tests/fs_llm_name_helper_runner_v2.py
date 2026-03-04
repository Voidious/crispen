from __future__ import annotations
from crispen.refactors.function_splitter import _llm_name_helpers
from tests.test_fs_llm_name_helpers_fixtures import _make_task


def _run_llm_name_helpers(mock_anthropic, mock_response, func_name: str):
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    tasks = [_make_task(func_name)]
    client = mock_anthropic.Anthropic.return_value
    return _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", tasks)
