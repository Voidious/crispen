from unittest.mock import patch
import pytest
from crispen.engine import run_engine
from crispen.errors import CrispenAPIError
from ._crispen_api_error_refactor import _CrispenApiErrorRefactor


def test_crispen_api_error_propagates(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch("crispen.engine._REFACTORS", [_CrispenApiErrorRefactor]):
        with pytest.raises(CrispenAPIError):
            list(run_engine({str(f): [(1, 1)]}))
