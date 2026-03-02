from unittest.mock import patch
from crispen.engine import _find_outside_callers


def test_find_outside_callers_deadline_expired(tmp_path):
    """Total budget already exhausted before any file is visited: all blocked."""
    (tmp_path / "other.py").write_text("x = 1\n")
    # A negative timeout makes the deadline fall in the past immediately.
    with patch("crispen.engine._SCOPE_ANALYSIS_TIMEOUT", -1):
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    assert result == {"some.func"}
