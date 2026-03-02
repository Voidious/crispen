from .test_no_changes_no_messages import _run


def test_skip_parse_error(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def f(:\n    pass\n", encoding="utf-8")
    msgs = _run({str(f): [(1, 2)]})
    assert any("parse error" in m for m in msgs)
