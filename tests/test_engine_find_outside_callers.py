from crispen.engine import _EXCLUDED_DIR_NAMES, _find_outside_callers


def test_find_outside_callers_excludes_venv_dirs(tmp_path):
    """Files inside excluded directories (.venv, __pycache__, etc.) are skipped."""
    for dirname in _EXCLUDED_DIR_NAMES:
        excluded = tmp_path / dirname / "lib"
        excluded.mkdir(parents=True, exist_ok=True)
        (excluded / "pkg.py").write_text(
            "from mypkg.service import get_user\nget_user()\n"
        )
    # Even though each excluded dir has a caller, none should be counted.
    result = _find_outside_callers(str(tmp_path), {"mypkg.service.get_user"}, set())
    assert "mypkg.service.get_user" not in result
