import textwrap
from unittest.mock import MagicMock
from tests.duplicate_extractor_test_responses import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)

# Source that already defines _helper AND has duplicate blocks.
_COLLISION_SOURCE = textwrap.dedent(
    """\
    def _helper(x):
        return x

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
_COLLISION_RANGES = [(9, 11)]  # overlaps bar's body

_RETURN_BLOCK_SOURCE = textwrap.dedent(
    """\
    def foo():
        x = compute(data)
        y = transform(x)
        return y

    def bar():
        x = compute(data)
        y = transform(x)
        return y
    """
)
_RETURN_BLOCK_RANGES = [(7, 9)]  # overlaps bar's body

_PARAM_DUP_SOURCE = textwrap.dedent(
    """\
    def test_a(mock_client):
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def test_b(mock_client):
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_PARAM_DUP_RANGES = [(7, 9)]  # overlaps test_b's body


def _setup_anthropic_retry_on_alg_failure_mocks(mock_anthropic, helper):
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


def _setup_anthropic_verify_rejects_then_retries_mocks(mock_anthropic, helper):
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
        _make_verify_response(False, ["wrong variable name"]),
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
