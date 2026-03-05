from crispen.engine import _compute_qname, _file_to_module, _find_repo_root


def test_find_repo_root_finds_git(tmp_path):
    (tmp_path / ".git").mkdir()
    subdir = tmp_path / "src"
    subdir.mkdir()
    f = subdir / "code.py"
    f.write_text("x = 1\n")
    root = _find_repo_root({str(f): [(1, 1)]})
    assert root == str(tmp_path)


def test_find_repo_root_not_found(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n")
    root = _find_repo_root({str(f): [(1, 1)]})
    assert root is None


def test_file_to_module_regular_file(tmp_path):
    f = tmp_path / "mypkg" / "service.py"
    f.parent.mkdir()
    f.write_text("x = 1\n")
    assert _file_to_module(str(tmp_path), str(f)) == "mypkg.service"


def test_file_to_module_init(tmp_path):
    f = tmp_path / "mypkg" / "__init__.py"
    f.parent.mkdir()
    f.write_text("")
    assert _file_to_module(str(tmp_path), str(f)) == "mypkg"


def test_compute_qname(tmp_path):
    f = tmp_path / "pkg" / "mod.py"
    f.parent.mkdir()
    f.write_text("")
    assert _compute_qname(str(tmp_path), str(f), "my_func") == "pkg.mod.my_func"
