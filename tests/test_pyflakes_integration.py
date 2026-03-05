from __future__ import annotations
from unittest.mock import patch
from tests.test_fixtures import _make_long_func
from tests.test_mock_response import _make_mock_response
from tests.test_splitter_helpers import _create_splitter_with_patched_key


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

    splitter = _create_splitter_with_patched_key(src)

    # Pyflakes check returned True → split not applied
    assert splitter.get_rewritten_source() is None
