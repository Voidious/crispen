from crispen.engine import _has_callers_outside_ranges


def test_has_callers_outside_ranges_syntax_error():
    assert _has_callers_outside_ranges("def f(:", "f", [(1, 1)]) is False
