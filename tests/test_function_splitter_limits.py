from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import FunctionSplitter
from tests.test_function_splitter_helpers import _make_long_func
from tests.test_parsing_helpers import _make_mock_response


def test_function_splitter_under_limits_no_op():
    # A small function should not be split
    src = "def small():\n    x = 1\n    return x\n"
    splitter = FunctionSplitter([(1, 10)], source=src, verbose=False)
    assert splitter.get_rewritten_source() is None


def test_function_splitter_out_of_range_no_op():
    # Function exists but is outside changed ranges
    src = _make_long_func(80)
    splitter = FunctionSplitter([(200, 300)], source=src, verbose=False, max_lines=10)
    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_over_line_limit(mock_anthropic):
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["process_tail"])
    )
    src = _make_long_func(80)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],
            source=src,
            verbose=False,
            max_lines=50,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    assert "_process_tail" in result
    assert "return _process_tail(" in result
    assert len(splitter.changes_made) >= 1


def test_function_splitter_empty_source():
    """FunctionSplitter created with no source does nothing."""
    splitter = FunctionSplitter([(1, 10)])
    assert splitter.get_rewritten_source() is None


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
