from unittest.mock import patch
import pytest
from crispen.config import CrispenConfig
from crispen.engine import run_engine
from crispen.errors import CrispenAPIError
from ._block_1111 import _FL_PATCH


def test_file_limiter_api_error_propagates(tmp_path):
    """CrispenAPIError from FileLimiter propagates out of run_engine."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    with patch(_FL_PATCH, side_effect=CrispenAPIError("rate limit")):
        with pytest.raises(CrispenAPIError, match="rate limit"):
            list(
                run_engine(
                    {str(f): [(1, 1)]},
                    config=CrispenConfig(max_file_lines=5),
                )
            )
