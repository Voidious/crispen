from __future__ import annotations
from unittest.mock import patch
import pytest
from crispen.refactors.function_splitter import (
    _ApiTimeout,
    _run_with_timeout,
    FunctionSplitter,
)
from tests.test_function_splitter_core import _make_long_func


def test_run_with_timeout_propagates_exception():
    def _raise():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        _run_with_timeout(_raise, 5)


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
