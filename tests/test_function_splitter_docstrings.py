from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import FunctionSplitter
from tests.test_function_splitter_helpers import _make_long_func
from tests.test_parsing_helpers import _make_mock_response


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
