import textwrap
from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import DuplicateExtractor
from .test_extraction_workflow_core import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)
from .test_high_level_duplicate_extractor_flow import _DUP_RANGES, _DUP_SOURCE
from .test_sequence_collector_and_helpers import _make_seq_info


def test_llm_veto_skips_non_matching_blocks(monkeypatch):
    from crispen.refactors.duplicate_extractor import _llm_veto

    client = MagicMock()
    non_matching = MagicMock()
    non_matching.type = "text"  # not tool_use → if condition False
    matching = MagicMock()
    matching.type = "tool_use"
    matching.name = "evaluate_duplicate"
    matching.input = {"is_valid_duplicate": True, "reason": "same"}
    response = MagicMock()
    response.content = [non_matching, matching]
    client.messages.create.return_value = response

    group = [_make_seq_info(1, 3), _make_seq_info(5, 7)]
    is_valid, reason, _ = _llm_veto(client, group)
    assert is_valid is True


def test_llm_extract_skips_non_matching_blocks(monkeypatch):
    from crispen.refactors.duplicate_extractor import _llm_extract

    client = MagicMock()
    non_matching = MagicMock()
    non_matching.type = "text"  # not tool_use → if condition False
    matching = MagicMock()
    matching.type = "tool_use"
    matching.name = "extract_helper"
    matching.input = {
        "function_name": "helper",
        "placement": "module_level",
        "helper_source": "def helper(): pass\n",
        "call_site_replacements": ["helper()\n"],
    }
    response = MagicMock()
    response.content = [non_matching, matching]
    client.messages.create.return_value = response

    group = [_make_seq_info(1, 3)]
    result = _llm_extract(client, group, "a = 1\n")
    assert result is not None
    assert result["function_name"] == "helper"


# ---------------------------------------------------------------------------
# engine integration: CrispenAPIError propagates
def test_verbose_false_suppresses_stderr(monkeypatch):
    # verbose=False must take all four if-self.verbose False branches without printing.
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

        de = DuplicateExtractor([(9, 11)], source=source, verbose=False)

    assert de._new_source is not None
    assert "_helper" in de._new_source


def _make_veto_response_with_notes(
    is_valid: bool, reason: str, notes: str
) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "evaluate_duplicate"
    block.input = {
        "is_valid_duplicate": is_valid,
        "reason": reason,
        "extraction_notes": notes,
    }
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_veto_notes_passed_to_extract(monkeypatch):
    """extraction_notes from veto are forwarded to the extract prompt."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response_with_notes(True, "same logic", "watch out for x"),
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
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE)

    assert de._new_source is not None
    extract_call = mock_client.messages.create.call_args_list[1]
    extract_prompt = extract_call.kwargs["messages"][0]["content"]
    assert "watch out for x" in extract_prompt
