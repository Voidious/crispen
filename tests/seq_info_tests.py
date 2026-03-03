from crispen.refactors.duplicate_extractor import _SeqInfo


def _make_seq_info(start: int, end: int, src: str = "") -> _SeqInfo:
    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope="foo",
        source=src,
        fingerprint="",
    )
