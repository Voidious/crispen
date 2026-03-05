from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import FunctionSplitter
from tests.mock_response_helpers import _make_mock_response
from tests.test_function_splitter_helpers import _make_long_func


@patch("crispen.llm_client.anthropic")
def test_function_splitter_recursive_split(mock_anthropic):
    # With small max_lines and broad changed_ranges, triggers multiple iterations
    # First call names helper for first function, second call for helper
    mock_anthropic.Anthropic.return_value.messages.create.side_effect = [
        _make_mock_response(["part1"]),
        _make_mock_response(["part2"]),
        _make_mock_response(["part3"]),
    ]

    # 13 body statements → with max_lines=5, needs multiple splits
    src = _make_long_func(13, "func")

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],  # broad range covers all helpers too
            source=src,
            verbose=False,
            max_lines=5,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    # Multiple splits occurred
    assert len(splitter.changes_made) >= 2


@patch("crispen.llm_client.anthropic")
def test_function_splitter_max_iterations_loop_exhausted(mock_anthropic):
    """Loop runs to completion (no break) when max iterations reached."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["helper"])
    )
    src = _make_long_func(80, "foo")

    # Patch _MAX_SPLIT_ITERATIONS to 1 → loop runs exactly once without breaking
    # (break only occurs at START of next iteration when tasks=[])
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("crispen.refactors.function_splitter._MAX_SPLIT_ITERATIONS", 1):
            splitter = FunctionSplitter(
                [(1, 1000)], source=src, verbose=False, max_lines=50
            )

    result = splitter.get_rewritten_source()
    assert result is not None
    assert len(splitter.changes_made) == 1
