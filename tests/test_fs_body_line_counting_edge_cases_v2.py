from __future__ import annotations
from crispen.refactors.function_splitter import _count_body_lines


def test_count_body_lines_no_funcdef():
    # Module-level code, no function
    assert _count_body_lines("x = 1\n") == 0
