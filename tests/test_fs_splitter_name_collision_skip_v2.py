from __future__ import annotations
from unittest.mock import patch
from tests.fs_anthropic_key_splitter_factory_v2 import _make_splitter_with_anthropic_key
from tests.fs_mock_llm_response_factory_v2 import _make_mock_response
from tests.fs_splitter_long_function_fixtures_v3 import _make_long_func


@patch("crispen.llm_client.anthropic")
def test_function_splitter_skips_name_collision(mock_anthropic):
    """Helper name colliding with an existing function causes the task to be dropped."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["helper"])  # would produce _helper
    )
    # _helper already exists; the LLM would name the extracted helper "helper"
    existing = "def _helper():\n    pass\n\n\n"
    src = existing + _make_long_func(80)

    splitter = _make_splitter_with_anthropic_key(src)

    # collision detected → task dropped → no rewrite
    assert splitter.get_rewritten_source() is None
