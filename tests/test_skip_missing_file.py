from .test_no_changes_no_messages import _run


def test_skip_missing_file(tmp_path):
    missing = str(tmp_path / "nonexistent.py")
    msgs = _run({missing: [(1, 10)]})
    assert len(msgs) == 1
    assert "SKIP" in msgs[0]
    assert "file not found" in msgs[0]
