from crispen.refactors.duplicate_extractor import (
    _SeqInfo,
    _replacement_steals_post_block_line,
)


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
    # Post-block line is blank — skip (blank lines are not "code").
    source_lines = ["x = 1\n", "\n", "y = 2\n"]
    seq = _make_steal_seq(1)  # next_idx=1 → "\n" → stripped is empty
    assert not _replacement_steals_post_block_line([seq], ["y = 2\n"], source_lines)


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
