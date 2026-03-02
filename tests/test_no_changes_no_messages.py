from crispen.config import CrispenConfig
from crispen.engine import run_engine


def _run(changed):
    return list(run_engine(changed, config=CrispenConfig(min_tuple_size=3)))


def test_no_changes_no_messages(tmp_path):
    f = tmp_path / "simple.py"
    f.write_text("x = 1\n", encoding="utf-8")
    msgs = _run({str(f): [(1, 1)]})
    assert msgs == []
