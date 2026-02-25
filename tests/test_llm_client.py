"""Tests for crispen.llm_client — 100% branch coverage."""

import json
from unittest.mock import MagicMock, patch

import pytest

from crispen.errors import CrispenAPIError
from crispen.llm_client import _token_param, call_with_tool, get_api_key, make_client


# ---------------------------------------------------------------------------
# get_api_key
# ---------------------------------------------------------------------------


def test_get_api_key_anthropic_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    assert get_api_key("anthropic") == "ant-key"


def test_get_api_key_anthropic_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(CrispenAPIError, match="ANTHROPIC_API_KEY"):
        get_api_key("anthropic", caller="Test")


def test_get_api_key_moonshot_present(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "moon-key")
    assert get_api_key("moonshot") == "moon-key"


def test_get_api_key_moonshot_missing(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(CrispenAPIError, match="MOONSHOT_API_KEY"):
        get_api_key("moonshot", caller="Test")


def test_get_api_key_openai_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
    assert get_api_key("openai") == "oai-key"


def test_get_api_key_openai_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(CrispenAPIError, match="OPENAI_API_KEY"):
        get_api_key("openai", caller="Test")


def test_get_api_key_lmstudio_no_key_needed():
    # LM Studio never requires an API key — returns the placeholder regardless.
    assert get_api_key("lmstudio") == "lm-studio"


# ---------------------------------------------------------------------------
# _token_param
# ---------------------------------------------------------------------------


def test_token_param_modern_models():
    for model in (
        "gpt-5",
        "gpt-5.1",
        "o1-mini",
        "o3",
        "o4-mini",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
        "computer-use-preview",
    ):
        assert _token_param(model) == "max_completion_tokens", model


def test_token_param_legacy_models():
    for model in ("gpt-3.5-turbo", "gpt-4", "moonshot-v1-32k", "qwen3-8b"):
        assert _token_param(model) == "max_tokens", model


# ---------------------------------------------------------------------------
# make_client
# ---------------------------------------------------------------------------


def test_make_client_anthropic():
    with patch("crispen.llm_client.anthropic") as mock_ant:
        mock_ant.Anthropic.return_value = MagicMock()
        client = make_client("anthropic", "key", timeout=30.0)
        mock_ant.Anthropic.assert_called_once_with(api_key="key", timeout=30.0)
        assert client is mock_ant.Anthropic.return_value


def test_make_client_moonshot():
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.OpenAI.return_value = MagicMock()
        client = make_client("moonshot", "key", timeout=30.0)
        mock_oai.OpenAI.assert_called_once()
        call_kwargs = mock_oai.OpenAI.call_args[1]
        assert call_kwargs["api_key"] == "key"
        assert "moonshot" in call_kwargs["base_url"]
        assert client is mock_oai.OpenAI.return_value


def test_make_client_openai_uses_default_url():
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.OpenAI.return_value = MagicMock()
        make_client("openai", "key", timeout=30.0)
        call_kwargs = mock_oai.OpenAI.call_args[1]
        assert call_kwargs["base_url"] is None


def test_make_client_lmstudio_uses_default_url():
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.OpenAI.return_value = MagicMock()
        make_client("lmstudio", "lm-studio", timeout=30.0)
        call_kwargs = mock_oai.OpenAI.call_args[1]
        assert "localhost" in call_kwargs["base_url"]


def test_make_client_base_url_override():
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.OpenAI.return_value = MagicMock()
        make_client("lmstudio", "lm-studio", base_url="http://custom:8080/v1")
        call_kwargs = mock_oai.OpenAI.call_args[1]
        assert call_kwargs["base_url"] == "http://custom:8080/v1"


# ---------------------------------------------------------------------------
# call_with_tool — anthropic provider
# ---------------------------------------------------------------------------


_TOOL = {
    "name": "evaluate_duplicate",
    "description": "Evaluate duplicates",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_valid_duplicate": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["is_valid_duplicate", "reason"],
    },
}
_MESSAGES = [{"role": "user", "content": "test prompt"}]


def _make_anthropic_response(tool_name: str, input_data: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = input_data
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_call_with_tool_anthropic_success():
    client = MagicMock()
    client.messages.create.return_value = _make_anthropic_response(
        "evaluate_duplicate", {"is_valid_duplicate": True, "reason": "same"}
    )
    result = call_with_tool(
        client,
        "anthropic",
        "claude-sonnet-4-6",
        256,
        _TOOL,
        "evaluate_duplicate",
        _MESSAGES,
    )
    assert result == {"is_valid_duplicate": True, "reason": "same"}


def test_call_with_tool_anthropic_skips_non_matching_blocks():
    client = MagicMock()
    # First block is a text block (non-matching), second is the tool_use match.
    text_block = MagicMock()
    text_block.type = "text"
    text_block.name = "other"
    matching_block = MagicMock()
    matching_block.type = "tool_use"
    matching_block.name = "evaluate_duplicate"
    matching_block.input = {"is_valid_duplicate": True, "reason": "ok"}
    resp = MagicMock()
    resp.content = [text_block, matching_block]
    client.messages.create.return_value = resp
    result = call_with_tool(
        client,
        "anthropic",
        "claude-sonnet-4-6",
        256,
        _TOOL,
        "evaluate_duplicate",
        _MESSAGES,
    )
    assert result == {"is_valid_duplicate": True, "reason": "ok"}


def test_call_with_tool_anthropic_api_error():
    with patch("crispen.llm_client.anthropic") as mock_ant:
        mock_ant.APIError = Exception
        client = MagicMock()
        client.messages.create.side_effect = Exception("rate limit")
        with pytest.raises(CrispenAPIError, match="Anthropic API error"):
            call_with_tool(
                client,
                "anthropic",
                "claude-sonnet-4-6",
                256,
                _TOOL,
                "evaluate_duplicate",
                _MESSAGES,
                caller="Test",
            )


# ---------------------------------------------------------------------------
# call_with_tool — moonshot provider
# ---------------------------------------------------------------------------


def _make_openai_response(tool_name: str, arguments: dict) -> MagicMock:
    tc = MagicMock()
    tc.function.arguments = json.dumps(arguments)
    message = MagicMock()
    message.tool_calls = [tc]
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_call_with_tool_moonshot_success():
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.APIError = Exception
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            "evaluate_duplicate", {"is_valid_duplicate": True, "reason": "same"}
        )
        result = call_with_tool(
            client,
            "moonshot",
            "moonshot-v1-32k",
            256,
            _TOOL,
            "evaluate_duplicate",
            _MESSAGES,
        )
    assert result == {"is_valid_duplicate": True, "reason": "same"}
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert call_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "evaluate_duplicate"},
    }


def test_call_with_tool_moonshot_api_error():
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.APIError = Exception
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("rate limit")
        with pytest.raises(CrispenAPIError, match="moonshot API error"):
            call_with_tool(
                client,
                "moonshot",
                "moonshot-v1-32k",
                256,
                _TOOL,
                "evaluate_duplicate",
                _MESSAGES,
                caller="Test",
            )


def test_call_with_tool_moonshot_malformed_json():
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.APIError = Exception
        client = MagicMock()
        tc = MagicMock()
        tc.function.arguments = '{"key": "unterminated'
        message = MagicMock()
        message.tool_calls = [tc]
        choice = MagicMock()
        choice.message = message
        resp = MagicMock()
        resp.choices = [choice]
        client.chat.completions.create.return_value = resp
        result = call_with_tool(
            client,
            "moonshot",
            "moonshot-v1-32k",
            256,
            _TOOL,
            "evaluate_duplicate",
            _MESSAGES,
        )
    assert result is None


# ---------------------------------------------------------------------------
# call_with_tool — openai-compatible (non-moonshot) providers
# ---------------------------------------------------------------------------


def test_call_with_tool_openai_success():
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.APIError = Exception
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            "evaluate_duplicate", {"is_valid_duplicate": True, "reason": "same"}
        )
        result = call_with_tool(
            client,
            "openai",
            "gpt-4o",
            256,
            _TOOL,
            "evaluate_duplicate",
            _MESSAGES,
        )
    assert result == {"is_valid_duplicate": True, "reason": "same"}
    call_kwargs = client.chat.completions.create.call_args[1]
    # extra_body must NOT be set for non-moonshot providers
    assert "extra_body" not in call_kwargs
    # gpt-4o requires max_completion_tokens, not max_tokens
    assert "max_completion_tokens" in call_kwargs
    assert "max_tokens" not in call_kwargs


def test_call_with_tool_openai_api_error():
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.APIError = Exception
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("rate limit")
        with pytest.raises(CrispenAPIError, match="openai API error"):
            call_with_tool(
                client,
                "openai",
                "gpt-4o",
                256,
                _TOOL,
                "evaluate_duplicate",
                _MESSAGES,
                caller="Test",
            )


def test_call_with_tool_tool_choice_override():
    """tool_choice_override sends the string directly instead of named-function dict."""
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.APIError = Exception
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            "evaluate_duplicate", {"is_valid_duplicate": True, "reason": "same"}
        )
        call_with_tool(
            client,
            "lmstudio",
            "qwen3-8b",
            256,
            _TOOL,
            "evaluate_duplicate",
            _MESSAGES,
            tool_choice_override="required",
        )
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["tool_choice"] == "required"
