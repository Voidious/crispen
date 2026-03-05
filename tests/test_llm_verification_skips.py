from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import DuplicateExtractor
from tests.test_duplicate_extractor_core_extractor import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)
from tests.test_duplicate_extractor_end_to_end import _DUP_RANGES, _DUP_SOURCE


def test_llm_verify_exhausted_skips_group(monkeypatch):
    """All verify attempts fail -> group skipped."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
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
            _make_verify_response(False, ["issue"]),
        ]
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE, llm_verify_retries=0)

    assert de._new_source is None
