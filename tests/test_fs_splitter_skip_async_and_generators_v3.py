from __future__ import annotations
from unittest.mock import patch
from tests.fs_anthropic_key_rewrite_regression_v2 import (
    _splitter_rewrite_is_none_with_anthropic_key,
)


@patch("crispen.llm_client.anthropic")
def test_function_splitter_async_skipped(mock_anthropic):
    # Async functions should not be split
    src = (
        "async def foo():\n"
        + "".join(f"    a{i} = {i}\n" for i in range(80))
        + "    return 0\n"
    )

    _splitter_rewrite_is_none_with_anthropic_key(src)


@patch("crispen.llm_client.anthropic")
def test_function_splitter_generator_skipped(mock_anthropic):
    # Generator functions should not be split
    src = (
        "def gen():\n"
        + "".join(f"    a{i} = {i}\n" for i in range(80))
        + "    yield 0\n"
    )

    _splitter_rewrite_is_none_with_anthropic_key(src)
