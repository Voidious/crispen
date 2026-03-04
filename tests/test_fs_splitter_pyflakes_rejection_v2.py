from __future__ import annotations
from unittest.mock import patch
from tests.fs_anthropic_key_splitter_factory_v2 import _make_splitter_with_anthropic_key
from tests.fs_mock_llm_response_factory_v2 import _make_mock_response
from tests.fs_splitter_long_function_fixtures_v3 import _make_long_func


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

    splitter = _make_splitter_with_anthropic_key(src)

    # Pyflakes check returned True → split not applied
    assert splitter.get_rewritten_source() is None
