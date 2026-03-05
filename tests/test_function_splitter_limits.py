from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import FunctionSplitter
from tests.mock_response_helpers import _make_mock_response
from tests.test_function_splitter_helpers import _make_long_func


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
