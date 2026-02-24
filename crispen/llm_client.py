"""Unified LLM client supporting Anthropic and OpenAI-compatible providers."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import anthropic
import openai

from .errors import CrispenAPIError

_PROVIDER_BASE_URLS: dict[str, str] = {
    "moonshot": "https://api.moonshot.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "lmstudio": "http://localhost:1234/v1",
}

# Maps provider name to its required environment variable.
# None means no API key is required (e.g. LM Studio running locally).
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
) -> Optional[dict]:
    """Call the LLM with forced tool use; return the tool input dict or None.

    Raises CrispenAPIError on API errors.
    """
    if provider == "anthropic":
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool_name},
                messages=messages,
            )
        except anthropic.APIError as exc:
            raise CrispenAPIError(
                f"{caller}: Anthropic API error: {exc}\n"
                "Commit blocked. To skip all hooks: git commit --no-verify"
            ) from exc
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input
        return None  # pragma: no cover
    else:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        }
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "tools": [openai_tool],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
            "messages": messages,
        }
        if provider == "moonshot":
            create_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        try:
            response = client.chat.completions.create(**create_kwargs)
        except openai.APIError as exc:
            raise CrispenAPIError(
                f"{caller}: {provider} API error: {exc}\n"
                "Commit blocked. To skip all hooks: git commit --no-verify"
            ) from exc
        if response.choices and response.choices[0].message.tool_calls:
            tc = response.choices[0].message.tool_calls[0]
            try:
                return json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                return None
        return None  # pragma: no cover
