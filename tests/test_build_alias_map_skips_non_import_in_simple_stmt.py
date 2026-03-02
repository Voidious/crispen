from crispen.engine import _build_alias_map


def test_build_alias_map_skips_non_import_in_simple_stmt(tmp_path):
    # An assignment in SimpleStatementLine is not ImportFrom (line 79).
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("__version__ = '1.0'\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    assert alias_map == {"mypkg.service.get_user": "mypkg.service.get_user"}
