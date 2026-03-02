from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import run_engine
from ._block_1111 import _FL_PATCH


def test_file_limiter_skips_short_file(tmp_path):
    """File under max_file_lines → FileLimiter is not called for that file."""
    f = tmp_path / "short.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch(_FL_PATCH) as mock_fl:
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=100),
            )
        )
    mock_fl.assert_not_called()
