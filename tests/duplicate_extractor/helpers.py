import libcst as cst
from crispen.refactors.duplicate_extractor import _SeqInfo


def _parse_stmt(src: str) -> cst.BaseStatement:
    return cst.parse_module(src).body[0]


def _make_seq(start: int, end: int) -> _SeqInfo:
    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope="<module>",
        source="",
        fingerprint="",
    )


def _make_seq_with_source(source: str) -> _SeqInfo:
    return _SeqInfo(
        stmts=[], start_line=1, end_line=1, scope="f", source=source, fingerprint=""
    )
