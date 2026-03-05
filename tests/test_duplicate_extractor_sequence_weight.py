from crispen.refactors.duplicate_extractor import _SeqInfo, _sequence_weight
from tests.test_duplicate_extractor_ast_utils import _parse_stmt


def test_sequence_weight_empty():
    assert _sequence_weight([]) == 0


def test_sequence_weight_mixed():
    stmts = [
        _parse_stmt("a = 1\n"),
        _parse_stmt("if x:\n    b = 2\n"),
    ]
    assert _sequence_weight(stmts) == 1 + 2


def _make_seq(start: int, end: int) -> _SeqInfo:
    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope="<module>",
        source="",
        fingerprint="",
    )
