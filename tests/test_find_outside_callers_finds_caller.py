from crispen.engine import _find_outside_callers


def test_find_outside_callers_finds_caller(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    service = pkg / "service.py"
    service.write_text("def get_user():\n    return (1, 2, 3)\n")
    outside = tmp_path / "outside.py"
    outside.write_text("from mypkg.service import get_user\nget_user()\n")

    qname = "mypkg.service.get_user"
    diff_files = {str(service.resolve())}
    result = _find_outside_callers(str(tmp_path), {qname}, diff_files)
    assert qname in result
