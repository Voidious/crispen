import textwrap
from unittest.mock import patch
import pytest
from crispen.engine import _compute_qname, run_engine
from crispen.errors import CrispenAPIError
from crispen.refactors.base import Refactor
from tests.engine_helpers import _run
from tests.engine_refactor_classes import _CrispenApiErrorRefactor
from tests.engine_transformers import _RaisingTransformer

def test_skip_missing_file(tmp_path):
    missing = str(tmp_path / "nonexistent.py")
    msgs = _run({missing: [(1, 10)]})
    assert len(msgs) == 1
    assert "SKIP" in msgs[0]
    assert "file not found" in msgs[0]

def test_no_changes_no_messages(tmp_path):
    f = tmp_path / "simple.py"
    f.write_text("x = 1\n", encoding="utf-8")
    msgs = _run({str(f): [(1, 1)]})
    assert msgs == []

def test_applies_refactor_and_writes(tmp_path):
    source = textwrap.dedent(
        """\
        if not x:
            a()
        else:
            b()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    msgs = _run({str(f): [(1, 4)]})
    assert any("IfNotElse" in m for m in msgs)
    assert "if x:" in f.read_text(encoding="utf-8")

def test_rewritten_source_used_when_available(tmp_path):
    """get_rewritten_source() is preferred over new_tree.code when non-None."""
    rewritten = "x = 999  # rewritten\n"

    class _RewritingRefactor(Refactor):
        @classmethod
        def name(cls):
            return "Rewriter"

        def get_rewritten_source(self):
            return rewritten

        def get_changes(self):
            return ["Rewriter: rewrote the file"]

    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch("crispen.engine._REFACTORS", [_RewritingRefactor]):
        msgs = _run({str(f): [(1, 1)]})
    assert any("Rewriter" in m for m in msgs)
    assert f.read_text(encoding="utf-8") == rewritten

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

def test_compute_qname(tmp_path):
    f = tmp_path / "pkg" / "mod.py"
    f.parent.mkdir()
    f.write_text("")
    assert _compute_qname(str(tmp_path), str(f), "my_func") == "pkg.mod.my_func"
