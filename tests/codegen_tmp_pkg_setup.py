from __future__ import annotations


def _setup_tmp_pkg_with_utils(
    tmp_path,
    *,
    pkg_name: str = "mypkg",
    mod_name: str = "utils.py",
    mod_source: str = "def _helper():\n    pass\n",
):
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / pkg_name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    mod = pkg / mod_name
    mod.write_text(mod_source)
    return pkg, mod
