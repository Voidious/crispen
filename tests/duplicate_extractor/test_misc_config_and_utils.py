from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import (
    _SeqInfo,
    _seq_ends_with_return,
    _replacement_contains_return,
    _replacement_steals_post_block_line,
    DuplicateExtractor,
)
from .test_duplicate_extractor_indentation_utils import _make_seq_with_source
from .test_extraction_workflow_core import _make_veto_response
from .test_high_level_duplicate_extractor_flow import _DUP_RANGES, _DUP_SOURCE


def test_duplicate_extractor_custom_model_used(monkeypatch):
    """Custom model string is passed to the Anthropic API."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.return_value = _make_veto_response(False, "no")
        DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE, model="claude-opus-4-6")
    # Verify the custom model was passed
    call_kwargs = mock_client.messages.create.call_args_list[0][1]
    assert call_kwargs["model"] == "claude-opus-4-6"


def test_seq_ends_with_return_true():
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    return x\n"))
        is True
    )


def test_seq_ends_with_return_false_no_return():
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    y = 2\n")) is False
    )


def test_seq_ends_with_return_syntax_error():
    assert _seq_ends_with_return(_make_seq_with_source("    (\n")) is False


def test_seq_ends_with_return_empty_body():
    # Pure whitespace → ast.parse produces an empty module body.
    assert _seq_ends_with_return(_make_seq_with_source("   \n")) is False


def test_seq_ends_with_return_bare_return():
    # Bare `return` is equivalent to returning None — not flagged.
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    return\n")) is False
    )


def test_seq_ends_with_return_return_none():
    # Explicit `return None` is also equivalent to implicit None — not flagged.
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    return None\n"))
        is False
    )


def test_replacement_contains_return_true():
    assert _replacement_contains_return("    return x\n") is True


def test_replacement_contains_return_false():
    assert _replacement_contains_return("    _helper()\n") is False


def test_replacement_contains_return_syntax_error():
    # Unclosed paren inside the wrapper → SyntaxError → False.
    assert _replacement_contains_return("    (\n") is False


def _make_steal_seq(end_line: int) -> _SeqInfo:
    return _SeqInfo(
        stmts=[], start_line=1, end_line=end_line, scope="f", source="", fingerprint=""
    )


def test_replacement_steals_post_block_at_eof():
    # Block is the last line of the file — no post-block line exists.
    source_lines = ["x = 1\n"]
    seq = _make_steal_seq(1)  # next_idx=1 >= len=1 → skip
    assert not _replacement_steals_post_block_line(
        [seq], ["y = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_blank_after():
    # Post-block line is blank but there is a non-blank line further down.
    # The check must scan past the blank to find the real post-block code.
    source_lines = ["x = 1\n", "\n", "y = 2\n"]
    seq = _make_steal_seq(1)  # next_idx=1 → "\n" → scan → next_idx=2 → "y = 2"
    assert _replacement_steals_post_block_line([seq], ["y = 2\n"], source_lines)


def test_replacement_steals_post_block_blank_after_no_match():
    # Blank after block, but replacement doesn't steal the non-blank post-block line.
    source_lines = ["x = 1\n", "\n", "y = 2\n"]
    seq = _make_steal_seq(1)
    assert not _replacement_steals_post_block_line(
        [seq], ["z = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_all_blank_after():
    # Only blank lines follow the block — no real post-block line to steal.
    source_lines = ["x = 1\n", "\n", "\n"]
    seq = _make_steal_seq(1)
    assert not _replacement_steals_post_block_line(
        [seq], ["z = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_no_match():
    # Replacement last line doesn't match post-block line.
    source_lines = ["x = 1\n", "y = 2\n"]
    seq = _make_steal_seq(1)  # next_idx=1 → "y = 2"
    assert not _replacement_steals_post_block_line(
        [seq], ["z = helper()\n"], source_lines
    )


def test_replacement_steals_post_block_match():
    # Replacement last line matches post-block line → steal detected.
    source_lines = ["x = 1\n", "y = 2\n"]
    seq = _make_steal_seq(1)  # next_idx=1 → "y = 2"
    assert _replacement_steals_post_block_line(
        [seq], ["z = helper()\ny = 2\n"], source_lines
    )
