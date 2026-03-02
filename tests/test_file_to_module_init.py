from crispen.engine import _file_to_module


def test_file_to_module_init(tmp_path):
    f = tmp_path / "mypkg" / "__init__.py"
    f.parent.mkdir()
    f.write_text("")
    assert _file_to_module(str(tmp_path), str(f)) == "mypkg"
