from crispen.refactors.duplicate_extractor import _SeqInfo


def _make_esc_seq(start: int, end: int) -> _SeqInfo:
    """Create a _SeqInfo for escaping-vars tests."""
    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope="foo",
        source="",
        fingerprint="",
    )
