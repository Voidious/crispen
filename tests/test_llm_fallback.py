from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import _ApiTimeout, FunctionSplitter
from tests.test_fixtures import _make_long_func
from tests.test_mock_response import _make_mock_response


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
