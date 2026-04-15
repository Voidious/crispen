"""Tests for crispen.llm_client — 100% branch coverage."""

import json
from unittest.mock import MagicMock, patch

import pytest

from crispen.errors import CrispenAPIError
from crispen.llm_client import (
    _token_param,
    call_with_tool,
    get_api_key,
    make_client,
)


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
    resp.usage = MagicMock()
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 50
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
    assert result.tool_input == {"is_valid_duplicate": True, "reason": "same"}
    assert result.elapsed >= 0
    assert isinstance(result.input_tokens, int)


def test_call_with_tool_anthropic_no_matching_block_returns_none():
    """When no block matches tool_name, tool_input is None."""
    client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.name = "other"
    resp = MagicMock()
    resp.content = [text_block]
    resp.usage = MagicMock()
    resp.usage.input_tokens = 10
    resp.usage.output_tokens = 5
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
    assert result.tool_input is None


def test_call_with_tool_anthropic_stop_reason_max_tokens_logs_warning(capsys):
    """When stop_reason is 'max_tokens', a truncation warning is printed to stderr."""
    client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.name = "other"
    resp = MagicMock()
    resp.content = [text_block]
    resp.stop_reason = "max_tokens"
    resp.usage.input_tokens = 10
    resp.usage.output_tokens = 256
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
    assert result.tool_input is None
    assert result.truncated is True
    assert "stop_reason=max_tokens" in capsys.readouterr().err


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
    resp.usage = MagicMock()
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 50
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
    assert result.tool_input == {"is_valid_duplicate": True, "reason": "ok"}


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


def test_call_with_tool_anthropic_429_retries_then_succeeds(monkeypatch, capsys):
    """429 from Anthropic triggers sleep + retry; succeeds on second attempt."""

    class _RateLimit429(Exception):
        status_code = 429

    sleep_calls = []
    monkeypatch.setattr(
        "crispen.llm_client.time.sleep", lambda s: sleep_calls.append(s)
    )

    with patch("crispen.llm_client.anthropic") as mock_ant:
        mock_ant.APIError = _RateLimit429
        client = MagicMock()
        success_resp = _make_anthropic_response(
            "evaluate_duplicate", {"is_valid_duplicate": True, "reason": "ok"}
        )
        client.messages.create.side_effect = [_RateLimit429("overloaded"), success_resp]
        result = call_with_tool(
            client,
            "anthropic",
            "claude-sonnet-4-6",
            256,
            _TOOL,
            "evaluate_duplicate",
            _MESSAGES,
            caller="Test",
            rate_limit_retries=2,
            rate_limit_backoff=5.0,
        )

    assert result.tool_input == {"is_valid_duplicate": True, "reason": "ok"}
    assert sleep_calls == [5.0]
    err = capsys.readouterr().err
    assert "rate limit (429)" in err
    assert "attempt 2/3" in err


def test_call_with_tool_anthropic_429_exponential_backoff(monkeypatch):
    """Backoff doubles on each successive 429 retry."""

    class _RateLimit429(Exception):
        status_code = 429

    sleep_calls = []
    monkeypatch.setattr(
        "crispen.llm_client.time.sleep", lambda s: sleep_calls.append(s)
    )

    with patch("crispen.llm_client.anthropic") as mock_ant:
        mock_ant.APIError = _RateLimit429
        client = MagicMock()
        success_resp = _make_anthropic_response(
            "evaluate_duplicate", {"is_valid_duplicate": True, "reason": "ok"}
        )
        client.messages.create.side_effect = [
            _RateLimit429("rl"),
            _RateLimit429("rl"),
            success_resp,
        ]
        call_with_tool(
            client,
            "anthropic",
            "claude-sonnet-4-6",
            256,
            _TOOL,
            "evaluate_duplicate",
            _MESSAGES,
            rate_limit_retries=3,
            rate_limit_backoff=10.0,
        )

    assert sleep_calls == [10.0, 20.0]


def test_call_with_tool_anthropic_429_exhausts_retries(monkeypatch):
    """When all retries are exhausted on 429, CrispenAPIError is raised."""

    class _RateLimit429(Exception):
        status_code = 429

    sleep_calls = []
    monkeypatch.setattr(
        "crispen.llm_client.time.sleep", lambda s: sleep_calls.append(s)
    )

    with patch("crispen.llm_client.anthropic") as mock_ant:
        mock_ant.APIError = _RateLimit429
        client = MagicMock()
        client.messages.create.side_effect = _RateLimit429("rate limited")
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
                rate_limit_retries=2,
                rate_limit_backoff=5.0,
            )

    assert sleep_calls == [5.0, 10.0]


def test_call_with_tool_anthropic_returns_timing_and_tokens():
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
    assert result.tool_input == {"is_valid_duplicate": True, "reason": "same"}
    assert result.elapsed >= 0
    assert result.input_tokens == 100
    assert result.output_tokens == 50


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
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 80
    resp.usage.completion_tokens = 40
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
    assert result.tool_input == {"is_valid_duplicate": True, "reason": "same"}
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert call_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "evaluate_duplicate"},
    }


def test_call_with_tool_moonshot_api_error():
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.BadRequestError = Exception
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
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 80
        resp.usage.completion_tokens = 40
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
    assert result.tool_input is None


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
    assert result.tool_input == {"is_valid_duplicate": True, "reason": "same"}
    call_kwargs = client.chat.completions.create.call_args[1]
    # extra_body must NOT be set for non-moonshot providers
    assert "extra_body" not in call_kwargs
    # gpt-4o requires max_completion_tokens, not max_tokens
    assert "max_completion_tokens" in call_kwargs
    assert "max_tokens" not in call_kwargs


def test_call_with_tool_openai_returns_timing_and_tokens():
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.APIError = Exception
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            "evaluate_duplicate", {"is_valid_duplicate": True}
        )
        result = call_with_tool(
            client, "openai", "gpt-4o", 256, _TOOL, "evaluate_duplicate", _MESSAGES
        )
    assert result.tool_input == {"is_valid_duplicate": True}
    assert result.elapsed >= 0
    assert result.input_tokens == 80
    assert result.output_tokens == 40


def test_call_with_tool_openai_api_error():
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.BadRequestError = Exception
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


def test_call_with_tool_openai_invalid_prompt_returns_none(capsys):
    """invalid_prompt 400 returns tool_input=None and prints to stderr."""

    class _InvalidPromptError(Exception):
        code = "invalid_prompt"

    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.BadRequestError = _InvalidPromptError
        mock_oai.APIError = Exception
        client = MagicMock()
        client.chat.completions.create.side_effect = _InvalidPromptError("policy flag")
        result = call_with_tool(
            client,
            "openai",
            "gpt-5.1",
            256,
            _TOOL,
            "evaluate_duplicate",
            _MESSAGES,
            caller="Test",
        )
    assert result.tool_input is None
    err = capsys.readouterr().err
    assert "openai API error" in err
    assert "policy flag" in err


def test_call_with_tool_openai_bad_request_other_code_raises_with_diag():
    """BadRequestError with code != 'invalid_prompt' raises CrispenAPIError + diag."""

    class _BadRequest(Exception):
        code = None  # e.g. 'invalid_request_error' JSON-parse failure

    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.BadRequestError = _BadRequest
        mock_oai.APIError = Exception
        client = MagicMock()
        client.chat.completions.create.side_effect = _BadRequest("invalid json body")
        with pytest.raises(CrispenAPIError) as exc_info:
            call_with_tool(
                client,
                "openai",
                "gpt-5.2",
                512,
                _TOOL,
                "evaluate_duplicate",
                _MESSAGES,
                caller="Test",
            )
    msg = str(exc_info.value)
    assert "openai API error" in msg
    assert "messages_chars=" in msg


def test_call_with_tool_openai_bad_request_ctrl_chars_in_diag():
    """BadRequestError diagnostic reports control characters found in messages."""

    class _BadRequest(Exception):
        code = None

    ctrl_messages = [{"role": "user", "content": "hello\x00world"}]
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.BadRequestError = _BadRequest
        mock_oai.APIError = Exception
        client = MagicMock()
        client.chat.completions.create.side_effect = _BadRequest("bad")
        with pytest.raises(CrispenAPIError) as exc_info:
            call_with_tool(
                client,
                "openai",
                "gpt-5.2",
                512,
                _TOOL,
                "evaluate_duplicate",
                ctrl_messages,
                caller="Test",
            )
    msg = str(exc_info.value)
    assert "ctrl=" in msg


def test_call_with_tool_openai_bad_request_json_error_in_diag():
    """BadRequestError diagnostic reports json_error when messages can't serialize."""

    class _BadRequest(Exception):
        code = None

    # A set is not JSON-serializable → triggers the except branch in the diagnostic.
    bad_messages = [{"role": "user", "content": {"not", "serializable"}}]
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.BadRequestError = _BadRequest
        mock_oai.APIError = Exception
        client = MagicMock()
        client.chat.completions.create.side_effect = _BadRequest("bad")
        with pytest.raises(CrispenAPIError) as exc_info:
            call_with_tool(
                client,
                "openai",
                "gpt-5.2",
                512,
                _TOOL,
                "evaluate_duplicate",
                bad_messages,
                caller="Test",
            )
    msg = str(exc_info.value)
    assert "json_error=" in msg


def test_call_with_tool_openai_parse_error_400_retries_and_succeeds(capsys):
    """BadRequestError with 'parse' in the message retries and succeeds."""

    class _BadRequest(Exception):
        code = None

    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.BadRequestError = _BadRequest
        mock_oai.APIError = Exception
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _BadRequest("cannot parse the JSON body"),
            _make_openai_response("evaluate_duplicate", {"is_valid_duplicate": True}),
        ]
        result = call_with_tool(
            client,
            "openai",
            "gpt-5.2",
            512,
            _TOOL,
            "evaluate_duplicate",
            _MESSAGES,
            caller="Test",
            rate_limit_retries=1,
        )
    err = capsys.readouterr().err
    assert "(400, retrying)" in err
    assert result.tool_input is not None


def test_call_with_tool_openai_parse_error_400_no_retry_slots_raises():
    """parse-error retry is blocked when rate_limit_retries=0."""

    class _BadRequest(Exception):
        code = None

    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.BadRequestError = _BadRequest
        mock_oai.APIError = Exception
        client = MagicMock()
        client.chat.completions.create.side_effect = _BadRequest(
            "cannot parse the JSON body"
        )
        with pytest.raises(CrispenAPIError) as exc_info:
            call_with_tool(
                client,
                "openai",
                "gpt-5.2",
                512,
                _TOOL,
                "evaluate_duplicate",
                _MESSAGES,
                caller="Test",
                rate_limit_retries=0,
            )
    msg = str(exc_info.value)
    assert "messages_chars=" in msg


def test_call_with_tool_openai_bad_request_surrogates_in_diag():
    """BadRequestError diagnostic reports lone surrogates found in messages."""

    class _BadRequest(Exception):
        code = None

    surr_messages = [{"role": "user", "content": "hello\ud800world"}]
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.BadRequestError = _BadRequest
        mock_oai.APIError = Exception
        client = MagicMock()
        client.chat.completions.create.side_effect = _BadRequest("bad request")
        with pytest.raises(CrispenAPIError) as exc_info:
            call_with_tool(
                client,
                "openai",
                "gpt-5.2",
                512,
                _TOOL,
                "evaluate_duplicate",
                surr_messages,
                caller="Test",
            )
    msg = str(exc_info.value)
    assert "surrogates=" in msg


def test_call_with_tool_openai_non_bad_request_api_error_raises():
    """Non-BadRequestError openai.APIError is re-raised as CrispenAPIError."""

    class _BadRequest(Exception):
        code = "invalid_prompt"

    class _OtherAPIError(Exception):
        pass

    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.BadRequestError = _BadRequest
        mock_oai.APIError = _OtherAPIError
        client = MagicMock()
        client.chat.completions.create.side_effect = _OtherAPIError("rate limit")
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


def test_call_with_tool_openai_empty_choices_returns_none_tool_input():
    """When choices is empty, tool_input is None."""
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.APIError = Exception
        client = MagicMock()
        resp = MagicMock()
        resp.choices = []
        resp.usage.prompt_tokens = 80
        resp.usage.completion_tokens = 40
        client.chat.completions.create.return_value = resp
        result = call_with_tool(
            client,
            "openai",
            "gpt-4o",
            256,
            _TOOL,
            "evaluate_duplicate",
            _MESSAGES,
        )
    assert result.tool_input is None


def test_call_with_tool_openai_finish_reason_length_logs_warning(capsys):
    """When finish_reason is 'length', a truncation warning is printed to stderr."""
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.APIError = Exception
        client = MagicMock()
        tc = MagicMock()
        tc.function.arguments = (
            '{"decisions": [{"group_id": 0, "action": "migr'  # truncated
        )
        choice = MagicMock()
        choice.message.tool_calls = [tc]
        choice.finish_reason = "length"
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage.prompt_tokens = 80
        resp.usage.completion_tokens = 845
        client.chat.completions.create.return_value = resp
        result = call_with_tool(
            client,
            "deepseek",
            "deepseek-chat",
            845,
            _TOOL,
            "evaluate_duplicate",
            _MESSAGES,
        )
    assert result.tool_input is None
    assert result.truncated is True
    assert "finish_reason=length" in capsys.readouterr().err


def test_call_with_tool_anthropic_usage_attribute_error_returns_zero_tokens():
    """When response.usage raises AttributeError, tokens default to 0."""
    client = MagicMock()
    resp = _make_anthropic_response("evaluate_duplicate", {"is_valid_duplicate": True})
    del resp.usage  # make resp.usage raise AttributeError
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
    assert result.input_tokens == 0
    assert result.output_tokens == 0


def test_call_with_tool_openai_usage_attribute_error_returns_zero_tokens():
    """When response.usage raises AttributeError, tokens default to 0."""
    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.APIError = Exception
        client = MagicMock()
        resp = _make_openai_response("evaluate_duplicate", {"is_valid_duplicate": True})
        del resp.usage  # make resp.usage raise AttributeError
        client.chat.completions.create.return_value = resp
        result = call_with_tool(
            client,
            "openai",
            "gpt-4o",
            256,
            _TOOL,
            "evaluate_duplicate",
            _MESSAGES,
        )
    assert result.input_tokens == 0
    assert result.output_tokens == 0


def test_call_with_tool_openai_429_retries_then_succeeds(monkeypatch, capsys):
    """429 from OpenAI triggers sleep + retry; succeeds on second attempt."""

    class _APIError(Exception):
        pass

    class _BadRequest(_APIError):
        code = "other"

    class _RateLimit429(_APIError):
        status_code = 429

    sleep_calls = []
    monkeypatch.setattr(
        "crispen.llm_client.time.sleep", lambda s: sleep_calls.append(s)
    )

    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.BadRequestError = _BadRequest
        mock_oai.APIError = _APIError
        client = MagicMock()
        success_resp = _make_openai_response(
            "evaluate_duplicate", {"is_valid_duplicate": True, "reason": "ok"}
        )
        client.chat.completions.create.side_effect = [
            _RateLimit429("overloaded"),
            success_resp,
        ]
        result = call_with_tool(
            client,
            "openai",
            "gpt-4o",
            256,
            _TOOL,
            "evaluate_duplicate",
            _MESSAGES,
            caller="Test",
            rate_limit_retries=2,
            rate_limit_backoff=5.0,
        )

    assert result.tool_input == {"is_valid_duplicate": True, "reason": "ok"}
    assert sleep_calls == [5.0]
    err = capsys.readouterr().err
    assert "rate limit (429)" in err
    assert "attempt 2/3" in err


def test_call_with_tool_openai_429_exhausts_retries(monkeypatch):
    """When all OpenAI retries are exhausted on 429, CrispenAPIError is raised."""

    class _APIError(Exception):
        pass

    class _BadRequest(_APIError):
        code = "other"

    class _RateLimit429(_APIError):
        status_code = 429

    sleep_calls = []
    monkeypatch.setattr(
        "crispen.llm_client.time.sleep", lambda s: sleep_calls.append(s)
    )

    with patch("crispen.llm_client.openai") as mock_oai:
        mock_oai.BadRequestError = _BadRequest
        mock_oai.APIError = _APIError
        client = MagicMock()
        client.chat.completions.create.side_effect = _RateLimit429("rate limited")
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
                rate_limit_retries=1,
                rate_limit_backoff=5.0,
            )

    assert sleep_calls == [5.0]


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
