from typing import Any
from dataclasses import dataclass
import textwrap
from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import DuplicateExtractor
from tests.duplicate_extractor_test_responses import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)


@dataclass
class SetupAnthropicExtractAndBuildDeResult:
    de: Any
    mock_client: Any
    source: Any
    helper: Any


def _setup_anthropic_extract_and_build_de(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        import os

        def foo():
            x = compute(data)
            y = transform(x)
            z = finalize(y)

        def bar():
            x = compute(data)
            y = transform(x)
            z = finalize(y)
        """
    )
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor([(9, 11)], source=source)

    assert de._new_source is not None
    return SetupAnthropicExtractAndBuildDeResult(
        de=de, mock_client=mock_client, source=source, helper=helper
    )
