"""Unified LLM client for crispen supporting Anthropic and Moonshot providers."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import anthropic
import openai

from .errors import CrispenAPIError

_MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"

_PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
}


def get_api_key(provider: str, caller: str = "crispen") -> str:
    """Return the API key for *provider* from the environment.

    Raises CrispenAPIError if the required environment variable is not set.
    """
    env_var = _PROVIDER_ENV_VARS.get(provider, "ANTHROPIC_API_KEY")
    api_key = os.environ.get(env_var)
    if not api_key:
        raise CrispenAPIError(
            f"{caller}: {env_var} is not set.\n"
            "Commit blocked. To skip all hooks: git commit --no-verify"
        )
    return api_key


def make_client(provider: str, api_key: str, timeout: float = 60.0) -> Any:
    """Create and return an LLM client for *provider*."""
    if provider == "moonshot":
        return openai.OpenAI(
            api_key=api_key,
            base_url=_MOONSHOT_BASE_URL,
            timeout=timeout,
        )
    else:
        return anthropic.Anthropic(api_key=api_key, timeout=timeout)


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
    if provider == "moonshot":
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        }
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                tools=[openai_tool],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                messages=messages,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except openai.APIError as exc:
            raise CrispenAPIError(
                f"{caller}: Moonshot API error: {exc}\n"
                "Commit blocked. To skip all hooks: git commit --no-verify"
            ) from exc
        if response.choices and response.choices[0].message.tool_calls:
            tc = response.choices[0].message.tool_calls[0]
            try:
                return json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                return None
        return None  # pragma: no cover
    else:
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
