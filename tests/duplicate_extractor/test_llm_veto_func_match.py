from unittest.mock import MagicMock, patch
import pytest
from crispen.errors import CrispenAPIError
from crispen.refactors.duplicate_extractor import _FunctionInfo, _llm_veto_func_match
from .test_sequence_collector import _make_seq_info


def _make_veto_func_match_response(is_valid: bool, reason: str = "test") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "evaluate_duplicate"
    block.input = {"is_valid_duplicate": is_valid, "reason": reason}
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_llm_veto_func_match_accepted():
    client = MagicMock()
    client.messages.create.return_value = _make_veto_func_match_response(
        True, "same op"
    )
    seq = _make_seq_info(7, 9, "    x = 1\n")
    func = _FunctionInfo(
        name="fn",
        source="def fn(): pass\n",
        scope="<module>",
        body_source="    pass\n",
        body_stmt_count=1,
        params=[],
    )
    is_valid, reason, _ = _llm_veto_func_match(client, seq, func, "source")
    assert is_valid is True
    assert reason == "same op"


def test_llm_veto_func_match_rejected():
    client = MagicMock()
    client.messages.create.return_value = _make_veto_func_match_response(
        False, "different"
    )
    seq = _make_seq_info(7, 9, "    x = 1\n")
    func = _FunctionInfo(
        name="fn",
        source="def fn(): pass\n",
        scope="<module>",
        body_source="    pass\n",
        body_stmt_count=1,
        params=[],
    )
    is_valid, reason, _ = _llm_veto_func_match(client, seq, func, "source")
    assert is_valid is False


def test_llm_veto_func_match_api_error():
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_anthropic.APIError = Exception
        client = MagicMock()
        client.messages.create.side_effect = Exception("api error")
        seq = _make_seq_info(7, 9, "    x = 1\n")
        func = _FunctionInfo(
            name="fn",
            source="def fn(): pass\n",
            scope="<module>",
            body_source="    pass\n",
            body_stmt_count=1,
            params=[],
        )
        with pytest.raises(CrispenAPIError):
            _llm_veto_func_match(client, seq, func, "source")


def test_llm_veto_func_match_skips_non_matching_blocks():
    """Non-matching content block is skipped; matching block still found."""
    client = MagicMock()
    non_matching = MagicMock()
    non_matching.type = "text"  # not tool_use → False branch of the if
    matching = MagicMock()
    matching.type = "tool_use"
    matching.name = "evaluate_duplicate"
    matching.input = {"is_valid_duplicate": True, "reason": "same"}
    response = MagicMock()
    response.content = [non_matching, matching]
    client.messages.create.return_value = response
    seq = _make_seq_info(7, 9, "    x = 1\n")
    func = _FunctionInfo(
        name="fn",
        source="def fn(): pass\n",
        scope="<module>",
        body_source="    pass\n",
        body_stmt_count=1,
        params=[],
    )
    is_valid, reason, _ = _llm_veto_func_match(client, seq, func, "source")
    assert is_valid is True
