from crispen.engine import _find_outside_callers


def test_find_outside_callers_empty_qnames(tmp_path):
    result = _find_outside_callers(str(tmp_path), set(), set())
    assert result == set()
