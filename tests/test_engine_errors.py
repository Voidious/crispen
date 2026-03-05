from unittest.mock import patch
import pytest
from crispen.engine import run_engine
from crispen.errors import CrispenAPIError
from tests.test_engine_helpers import (
    _CrispenApiErrorRefactor,
    _RaisingTransformer,
    _run,
)


def test_skip_parse_error(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def f(:\n    pass\n", encoding="utf-8")
    msgs = _run({str(f): [(1, 2)]})
    assert any("parse error" in m for m in msgs)


def test_skip_transform_error(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch("crispen.engine._REFACTORS", [_RaisingTransformer]):
        msgs = _run({str(f): [(1, 1)]})
    assert any("transform error" in m for m in msgs)


def test_crispen_api_error_propagates(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch("crispen.engine._REFACTORS", [_CrispenApiErrorRefactor]):
        with pytest.raises(CrispenAPIError):
            list(run_engine({str(f): [(1, 1)]}))


def test_tuple_dataclass_transform_error_handled(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")

    class _FailingTD:
        def __init__(self, *a, **kw):
            raise RuntimeError("simulated TupleDataclass failure")

    with patch("crispen.engine.TupleDataclass", _FailingTD):
        msgs = _run({str(f): [(1, 1)]})
    assert any("TupleDataclass" in m and "transform error" in m for m in msgs)
