from crispen.refactors.duplicate_extractor import _seq_ends_with_return
from tests.test_helpers import _make_seq_with_source


def test_seq_ends_with_return_return_none():
    # Explicit `return None` is also equivalent to implicit None — not flagged.
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    return None\n"))
        is False
    )
