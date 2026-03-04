from __future__ import annotations
from crispen.refactors.function_splitter import _count_body_lines


def test_count_body_lines_only_docstring():
    # Body has only a docstring → effectively empty
    src = 'def foo():\n    """doc"""\n'
    assert _count_body_lines(src) == 0


def test_count_body_lines_parse_error():
    assert _count_body_lines("def f(\n  !!invalid") == 0
