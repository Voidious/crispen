from crispen.engine import _find_repo_root


def test_find_repo_root_finds_git(tmp_path):
    (tmp_path / ".git").mkdir()
    subdir = tmp_path / "src"
    subdir.mkdir()
    f = subdir / "code.py"
    f.write_text("x = 1\n")
    root = _find_repo_root({str(f): [(1, 1)]})
    assert root == str(tmp_path)
