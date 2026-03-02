from unittest.mock import patch
from crispen.engine import _find_outside_callers


def test_find_outside_callers_scope_analysis_timeout(tmp_path):
    """When _visit_with_timeout times out, all target qnames are blocked."""
    (tmp_path / "other.py").write_text("x = 1\n")
    with patch("crispen.engine._visit_with_timeout", return_value=False):
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    assert result == {"some.func"}
