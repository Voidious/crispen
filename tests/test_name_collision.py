from __future__ import annotations
from unittest.mock import patch
from tests.mock_utils import _make_mock_response
from tests.test_function_splitter_helpers import _make_long_func
from tests.test_splitter_fixtures import _make_splitter_with_env


@patch("crispen.llm_client.anthropic")
def test_function_splitter_skips_name_collision(mock_anthropic):
    """Helper name colliding with an existing function causes the task to be dropped."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["helper"])  # would produce _helper
    )
    # _helper already exists; the LLM would name the extracted helper "helper"
    existing = "def _helper():\n    pass\n\n\n"
    src = existing + _make_long_func(80)

    splitter = _make_splitter_with_env(src)

    # collision detected → task dropped → no rewrite
    assert splitter.get_rewritten_source() is None
