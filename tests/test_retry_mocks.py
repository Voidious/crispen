from unittest.mock import MagicMock
from tests.mock_responses import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)


def _setup_retry_on_alg_failure_mocks(mock_anthropic, helper):
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
                "call_site_replacements": ["    _helper(data)\n"],  # wrong count
            }
        ),
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
    return mock_client
