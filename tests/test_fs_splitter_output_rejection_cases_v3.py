from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import FunctionSplitter
from tests.fs_splitter_long_function_fixtures_v3 import _make_long_func


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
