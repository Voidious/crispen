"""Unified LLM client supporting Anthropic and OpenAI-compatible providers."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import anthropic
import openai

from .errors import CrispenAPIError

_PROVIDER_BASE_URLS: dict[str, str] = {
    "moonshot": "https://api.moonshot.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "lmstudio": "http://localhost:1234/v1",
}


@dataclass
class LLMCallResult:
    """Timing and token data for a single LLM tool call."""

    tool_input: Optional[dict]
    elapsed: float
    input_tokens: int
    output_tokens: int
    truncated: bool = False  # True when the response was cut off by the token limit


# Maps provider name to its required environment variable.
# None means no API key is required (e.g. LM Studio running locally).
def _token_param(model: str) -> str:
    """Return the correct token-limit parameter name for an OpenAI model."""
    if model.startswith(
        ("gpt-5", "o1", "o3", "o4", "computer-use", "gpt-4.1", "gpt-4o")
    ):
        return "max_completion_tokens"
    return "max_tokens"


_PROVIDER_ENV_VARS: dict[str, Optional[str]] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "lmstudio": None,
}


def get_api_key(provider: str, caller: str = "crispen") -> str:
    """Return the API key for *provider* from the environment.

    Raises CrispenAPIError if the required environment variable is not set.
    LM Studio does not require an API key and always returns a placeholder.
    """
    env_var = _PROVIDER_ENV_VARS.get(provider, "ANTHROPIC_API_KEY")
    if env_var is None:
        return "lm-studio"
    api_key = os.environ.get(env_var)
    if not api_key:
        raise CrispenAPIError(
            f"{caller}: {env_var} is not set.\n"
            "Commit blocked. To skip all hooks: git commit --no-verify"
        )
    return api_key


def make_client(
    provider: str,
    api_key: str,
    timeout: float = 60.0,
    base_url: Optional[str] = None,
) -> Any:
    """Create and return an LLM client for *provider*.

    For OpenAI-compatible providers (moonshot, openai, deepseek, lmstudio), the
    base URL is resolved from *base_url* (if given) or the built-in default for the
    provider.  Pass *base_url* to override the default (e.g. a custom LM Studio port).
    """
    if provider == "anthropic":
        return anthropic.Anthropic(api_key=api_key, timeout=timeout)
    resolved_url = base_url or _PROVIDER_BASE_URLS.get(provider)
    return openai.OpenAI(api_key=api_key, base_url=resolved_url, timeout=timeout)


def call_with_tool(
    client: Any,
    provider: str,
    model: str,
    max_tokens: int,
    tool: dict,
    tool_name: str,
    messages: list,
    caller: str = "crispen",
    tool_choice_override: Optional[str] = None,
    rate_limit_retries: int = 6,
    rate_limit_backoff: float = 20.0,
) -> LLMCallResult:
    """Call the LLM with forced tool use; return an LLMCallResult.

    ``tool_input`` is None when the model did not invoke the tool.
    Raises CrispenAPIError on API errors.

    HTTP 429 rate-limit responses are retried up to *rate_limit_retries* times
    with exponential backoff starting at *rate_limit_backoff* seconds.
    """
    _rl_delay = rate_limit_backoff
    if provider == "anthropic":
        for _attempt in range(rate_limit_retries + 1):  # pragma: no branch
            try:
                t0 = time.perf_counter()
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool_name},
                    messages=messages,
                )
                break
            except anthropic.APIError as exc:
                if (
                    getattr(exc, "status_code", None) == 429
                    and _attempt < rate_limit_retries
                ):
                    print(
                        f"crispen: {caller}: rate limit (429), retrying in"
                        f" {_rl_delay:.0f}s"
                        f" (attempt {_attempt + 2}/{rate_limit_retries + 1})...",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(_rl_delay)
                    _rl_delay *= 2
                    continue
                raise CrispenAPIError(
                    f"{caller}: Anthropic API error: {exc}\n"
                    "Commit blocked. To skip all hooks: git commit --no-verify"
                ) from exc
        tool_input = None
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                tool_input = block.input
                break
        _truncated = False
        if tool_input is None and response.stop_reason == "max_tokens":
            print(
                f"crispen: {caller}: anthropic response truncated"
                f" (stop_reason=max_tokens, max_tokens={max_tokens})",
                file=sys.stderr,
                flush=True,
            )
            _truncated = True
        try:
            in_tok = int(response.usage.input_tokens)
            out_tok = int(response.usage.output_tokens)
        except (AttributeError, TypeError, ValueError):
            in_tok, out_tok = 0, 0
        return LLMCallResult(
            tool_input=tool_input,
            elapsed=time.perf_counter() - t0,
            input_tokens=in_tok,
            output_tokens=out_tok,
            truncated=_truncated,
        )
    else:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        }
        if tool_choice_override is not None:
            resolved_tool_choice: Any = tool_choice_override
        else:
            resolved_tool_choice = {"type": "function", "function": {"name": tool_name}}
        create_kwargs: dict[str, Any] = {
            "model": model,
            _token_param(model): max_tokens,
            "tools": [openai_tool],
            "tool_choice": resolved_tool_choice,
            "messages": messages,
        }
        if provider == "moonshot":
            create_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        _parse_retries_left = 2  # retries for transient 400 "cannot parse JSON" errors
        for _attempt in range(rate_limit_retries + 1):  # pragma: no branch
            try:
                t0 = time.perf_counter()
                response = client.chat.completions.create(**create_kwargs)
                break
            except openai.BadRequestError as exc:
                if getattr(exc, "code", None) == "invalid_prompt":
                    # Content policy flag — print warning and return None gracefully
                    # so callers skip the function rather than crashing the pipeline.
                    print(
                        f"crispen: {caller}: {provider} API error: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return LLMCallResult(
                        tool_input=None,
                        elapsed=time.perf_counter() - t0,
                        input_tokens=0,
                        output_tokens=0,
                    )
                # Retry transient 400 "cannot parse JSON body" errors.
                if (
                    "parse" in str(exc).lower()
                    and _parse_retries_left > 0
                    and _attempt < rate_limit_retries
                ):
                    _parse_retries_left -= 1
                    print(
                        f"crispen: {caller}: {provider} API error"
                        f" (400, retrying): {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                # Collect diagnostic info to help root-cause 400 errors.
                _diag: list[str] = []
                try:
                    _msg_json = json.dumps(messages)
                    _diag.append(f"messages_chars={len(_msg_json)}")
                    _ctrl = [
                        f"\\u{i:04x}" for i in range(0x20) if f"\\u{i:04x}" in _msg_json
                    ]
                    if _ctrl:
                        _diag.append(f"ctrl={_ctrl}")
                    # Check for lone surrogates (U+D800–U+DFFF); json.dumps
                    # encodes them as \udXXX, which strict servers reject.
                    _surr_pfx = [
                        "\\ud8",
                        "\\ud9",
                        "\\uda",
                        "\\udb",
                        "\\udc",
                        "\\udd",
                        "\\ude",
                        "\\udf",
                    ]
                    _surr = [p for p in _surr_pfx if p in _msg_json]
                    if _surr:
                        _diag.append(f"surrogates={_surr}")
                except (TypeError, ValueError) as _je:
                    _diag.append(f"json_error={_je!r}")
                raise CrispenAPIError(
                    f"{caller}: {provider} API error: {exc}"
                    + (f"\n  diag: {', '.join(_diag)}" if _diag else "")
                    + "\nCommit blocked. To skip all hooks: git commit --no-verify"
                ) from exc
            except openai.APIError as exc:
                if (
                    getattr(exc, "status_code", None) == 429
                    and _attempt < rate_limit_retries
                ):
                    print(
                        f"crispen: {caller}: rate limit (429), retrying in"
                        f" {_rl_delay:.0f}s"
                        f" (attempt {_attempt + 2}/{rate_limit_retries + 1})...",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(_rl_delay)
                    _rl_delay *= 2
                    continue
                raise CrispenAPIError(
                    f"{caller}: {provider} API error: {exc}\n"
                    "Commit blocked. To skip all hooks: git commit --no-verify"
                ) from exc
        tool_input = None
        if response.choices and response.choices[0].message.tool_calls:
            tc = response.choices[0].message.tool_calls[0]
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_input = None
        _truncated = False
        if tool_input is None and response.choices:
            if response.choices[0].finish_reason == "length":
                print(
                    f"crispen: {caller}: {provider} response truncated"
                    f" (finish_reason=length, max_tokens={max_tokens})",
                    file=sys.stderr,
                    flush=True,
                )
                _truncated = True
        try:
            in_tok = int(response.usage.prompt_tokens)
            out_tok = int(response.usage.completion_tokens)
        except (AttributeError, TypeError, ValueError):
            in_tok, out_tok = 0, 0
        return LLMCallResult(
            tool_input=tool_input,
            elapsed=time.perf_counter() - t0,
            input_tokens=in_tok,
            output_tokens=out_tok,
            truncated=_truncated,
        )
