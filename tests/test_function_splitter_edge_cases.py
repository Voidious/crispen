from __future__ import annotations
from unittest.mock import patch
from tests.test_function_splitter_helpers import _create_splitter_and_assert_no_rewrite


@patch("crispen.llm_client.anthropic")
def test_function_splitter_nested_funcdef_not_split(mock_anthropic):
    # A long function containing a nested funcdef should never be split,
    # even if it far exceeds the line limit.  Splitting across a closure
    # boundary produces cascading re-splits and semantically fragile helpers.
    lines = ["def func_with_closure():\n"]
    for i in range(80):
        lines.append(f"    a{i} = {i}\n")
    lines.append("    def inner():\n")
    lines.append("        return 0\n")
    lines.append("    return inner()\n")
    src = "".join(lines)

    _create_splitter_and_assert_no_rewrite(src)


@patch("crispen.llm_client.anthropic")
def test_function_splitter_async_skipped(mock_anthropic):
    # Async functions should not be split
    src = (
        "async def foo():\n"
        + "".join(f"    a{i} = {i}\n" for i in range(80))
        + "    return 0\n"
    )

    _create_splitter_and_assert_no_rewrite(src)


@patch("crispen.llm_client.anthropic")
def test_function_splitter_generator_skipped(mock_anthropic):
    # Generator functions should not be split
    src = (
        "def gen():\n"
        + "".join(f"    a{i} = {i}\n" for i in range(80))
        + "    yield 0\n"
    )

    _create_splitter_and_assert_no_rewrite(src)
