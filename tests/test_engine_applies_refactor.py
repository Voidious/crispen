from tests.engine_if_not_else_fixtures import _write_if_not_else_fixture
from tests.engine_run_helpers import _run


def test_applies_refactor_and_writes(tmp_path):
    f, source = _write_if_not_else_fixture(tmp_path)
    msgs = _run({str(f): [(1, 4)]})
    assert any("IfNotElse" in m for m in msgs)
    assert "if x:" in f.read_text(encoding="utf-8")
