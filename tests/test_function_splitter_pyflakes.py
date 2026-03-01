from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import FunctionSplitter
from .mock_responses import _make_mock_response
from .test_long_func import _make_long_func


@patch(
    "crispen.refactors.function_splitter._has_new_undefined_names", return_value=True
)
@patch("crispen.llm_client.anthropic")
def test_function_splitter_pyflakes_rejects_output(mock_anthropic, mock_has_undef):
    """If pyflakes detects new undefined names in output, the split is not applied."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["helper"])
    )
    src = _make_long_func(80, "foo")

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=50
        )

    # Pyflakes check returned True → split not applied
    assert splitter.get_rewritten_source() is None
