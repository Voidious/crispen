from crispen.engine import _has_callers_outside_ranges


def test_has_callers_outside_ranges_found():
    source = "def f(): pass\nf()\n"  # call on line 2, range is only line 1
    assert _has_callers_outside_ranges(source, "f", [(1, 1)]) is True


def test_has_callers_outside_ranges_not_found():
    source = "def f(): pass\nf()\n"  # call on line 2, range covers line 2
    assert _has_callers_outside_ranges(source, "f", [(1, 2)]) is False


def test_has_callers_outside_ranges_syntax_error():
    assert _has_callers_outside_ranges("def f(:", "f", [(1, 1)]) is False
