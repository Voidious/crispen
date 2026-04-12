from unittest.mock import patch
import textwrap
from crispen.config import CrispenConfig
from crispen.engine import (
    _blocked_private_scopes,
    _has_callers_outside_ranges,
    run_engine,
)
from .test_engine_core import _run
from .test_engine_cross_file import _make_pkg


def _make_phase1_pkg(root):
    """Helper: return a tmp_path containing a package for Phase 1 tests."""
    pkg = root / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    return pkg


def test_phase1_private_caller_updated(tmp_path):
    """Private function callers in the same file are updated after Phase 1."""
    source = textwrap.dedent(
        """\
        def _make_result():
            return (1, 2, 3)

        def use_it():
            a, b, c = _make_result()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    msgs = _run({str(f): [(1, 100)]})
    result = f.read_text(encoding="utf-8")
    assert "_ = _make_result()" in result
    assert any("CallerUpdater" in m for m in msgs)


def test_phase1_private_no_callers_no_caller_updater_msg(tmp_path):
    """Private transform with no callers produces no CallerUpdater message."""
    source = "def _make_result():\n    return (1, 2, 3)\n"
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    msgs = _run({str(f): [(1, 100)]})
    assert any("TupleDataclass" in m for m in msgs)
    assert not any("CallerUpdater" in m for m in msgs)


def test_phase1_private_caller_updater_exception_ignored(tmp_path):
    """If CallerUpdater raises during Phase 1, the engine continues gracefully."""
    source = textwrap.dedent(
        """\
        def _make_result():
            return (1, 2, 3)

        def use_it():
            a, b, c = _make_result()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    with patch("crispen.engine.CallerUpdater", side_effect=RuntimeError("fail")):
        msgs = _run({str(f): [(1, 100)]})
    # TupleDataclass still ran successfully
    assert any("TupleDataclass" in m for m in msgs)


def test_has_callers_outside_ranges_found():
    source = "def f(): pass\nf()\n"  # call on line 2, range is only line 1
    assert _has_callers_outside_ranges(source, "f", [(1, 1)]) is True


def test_has_callers_outside_ranges_not_found():
    source = "def f(): pass\nf()\n"  # call on line 2, range covers line 2
    assert _has_callers_outside_ranges(source, "f", [(1, 2)]) is False


def test_has_callers_outside_ranges_syntax_error():
    assert _has_callers_outside_ranges("def f(:", "f", [(1, 1)]) is False


def test_blocked_private_scopes_finds_outside_callers():
    # _helper called at line 3, diff range only covers line 1
    source = "def _helper(): pass\n\n_helper()\n"
    blocked = _blocked_private_scopes(source, [(1, 1)])
    assert "_helper" in blocked


def test_blocked_private_scopes_ignores_in_range_callers():
    # _helper called at line 3, diff range covers line 3
    source = "def _helper(): pass\n\n_helper()\n"
    blocked = _blocked_private_scopes(source, [(1, 3)])
    assert "_helper" not in blocked


def test_blocked_private_scopes_syntax_error():
    blocked = _blocked_private_scopes("def f(:", [(1, 1)])
    assert blocked == set()


def test_blocked_private_scopes_ignores_public():
    # Public functions (no leading _) should not appear in blocked set
    source = "def helper(): pass\n\nhelper()\n"
    blocked = _blocked_private_scopes(source, [(1, 1)])
    assert "helper" not in blocked


def test_update_diff_file_callers_false_blocks_private_with_outside_caller(tmp_path):
    """Private function with a caller outside diff ranges is NOT transformed."""
    source = textwrap.dedent(
        """\
        def _make_result():
            return (a, b, c)

        def use_in_diff():
            x, y, z = _make_result()

        def use_outside_diff():
            p, q, r = _make_result()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    config = CrispenConfig(min_tuple_size=3, update_diff_file_callers=False)
    # Diff only covers the function definition and use_in_diff
    msgs = list(run_engine({str(f): [(1, 5)]}, config=config))
    # Should NOT have been transformed (outside callers exist)
    assert not any("TupleDataclass" in m for m in msgs)
    assert "return (a, b, c)" in f.read_text(encoding="utf-8")


def test_update_diff_file_callers_false_allows_private_with_only_diff_callers(
    tmp_path,
):
    """Private function with all callers inside diff is transformed."""
    source = textwrap.dedent(
        """\
        def _make_result():
            return (a, b, c)

        def use_in_diff():
            x, y, z = _make_result()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    config = CrispenConfig(min_tuple_size=3, update_diff_file_callers=False)
    msgs = list(run_engine({str(f): [(1, 5)]}, config=config))
    # Only diff caller exists → transformation should proceed
    assert any("TupleDataclass" in m for m in msgs)


def test_update_diff_file_callers_false_blocks_public_with_diff_file_outside_caller(
    tmp_path,
):
    """Public function with callers outside diff in diff file is skipped."""
    pkg = _make_pkg(tmp_path, "mypkg")

    service = pkg / "service.py"
    service.write_text(
        "def get_user():\n    return (name, age, score)\n", encoding="utf-8"
    )

    api = pkg / "api.py"
    api.write_text(
        "from mypkg.service import get_user\n"
        "def in_diff():\n"
        "    a, b, c = get_user()\n"
        "def not_in_diff():\n"
        "    x, y, z = get_user()\n",
        encoding="utf-8",
    )

    # api.py diff only covers lines 1-3 (in_diff function)
    changed = {str(service): [(1, 2)], str(api): [(1, 3)]}
    config = CrispenConfig(min_tuple_size=3, update_diff_file_callers=False)
    msgs = list(run_engine(changed, _repo_root=str(tmp_path), config=config))

    # get_user has a caller outside the diff (not_in_diff at lines 4-5)
    assert any("callers exist outside the diff" in m for m in msgs)


def test_update_diff_file_callers_false_allows_public_with_all_callers_in_diff(
    tmp_path,
):
    """Public function with all callers inside diff (no diff-file outside callers)."""
    pkg = _make_pkg(tmp_path, "mypkg")

    service = pkg / "service.py"
    service.write_text(
        "def get_user():\n    return (name, age, score)\n", encoding="utf-8"
    )

    api = pkg / "api.py"
    api.write_text(
        "from mypkg.service import get_user\n"
        "def main():\n"
        "    a, b, c = get_user()\n",
        encoding="utf-8",
    )

    changed = {str(service): [(1, 2)], str(api): [(1, 3)]}
    config = CrispenConfig(min_tuple_size=3, update_diff_file_callers=False)
    msgs = list(run_engine(changed, _repo_root=str(tmp_path), config=config))

    # All callers within diff → transformation should proceed even with
    # update_diff_file_callers=False (no callers outside diff ranges)
    assert any("TupleDataclass" in m for m in msgs)
    assert any("CallerUpdater" in m for m in msgs)
