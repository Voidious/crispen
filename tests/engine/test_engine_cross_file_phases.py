from unittest.mock import patch
import textwrap
from crispen.config import CrispenConfig
from crispen.engine import _apply_tuple_dataclass, run_engine
import libcst as cst
from .test_engine_refactor_filters_and_stats import _run


def _make_pkg(root, name):
    pkg = root / name
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    return pkg


def test_cross_file_transforms_public_func_and_caller(tmp_path):
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

    changed = {str(service): [(1, 2)], str(api): [(1, 4)]}
    msgs = list(
        run_engine(
            changed,
            _repo_root=str(tmp_path),
            config=CrispenConfig(min_tuple_size=3),
        )
    )

    assert any("TupleDataclass" in m for m in msgs)
    assert any("CallerUpdater" in m for m in msgs)

    service_text = service.read_text(encoding="utf-8")
    assert "GetUserResult(" in service_text
    assert "@dataclass" in service_text

    api_text = api.read_text(encoding="utf-8")
    assert "_ = get_user()" in api_text
    assert "_.name" in api_text


def test_cross_file_skips_when_outside_caller_exists(tmp_path):
    pkg = _make_pkg(tmp_path, "mypkg")

    service = pkg / "service.py"
    service.write_text(
        "def get_user():\n    return (name, age, score)\n", encoding="utf-8"
    )

    # This file is NOT in the diff but calls get_user.
    outside = pkg / "outside.py"
    outside.write_text(
        "from mypkg.service import get_user\na, b, c = get_user()\n",
        encoding="utf-8",
    )

    changed = {str(service): [(1, 2)]}
    msgs = list(
        run_engine(
            changed,
            _repo_root=str(tmp_path),
            config=CrispenConfig(min_tuple_size=3),
        )
    )

    assert any("callers exist outside the diff" in m for m in msgs)
    assert "return (name, age, score)" in service.read_text(encoding="utf-8")


def test_cross_file_file_not_under_repo_root(tmp_path):
    # repo_root is a separate directory; changed file is not under it.
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    f = tmp_path / "code.py"
    f.write_text("def public_func():\n    return (1, 2, 3)\n", encoding="utf-8")
    # _compute_qname raises ValueError → all_candidates stays empty → 317->406 branch.
    msgs = list(
        run_engine(
            {str(f): [(1, 2)]},
            _repo_root=str(repo_root),
            config=CrispenConfig(min_tuple_size=3),
        )
    )
    assert not any("callers" in m for m in msgs)


def test_no_public_candidates_with_repo_root(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    msgs = list(run_engine({str(f): [(1, 1)]}, _repo_root=str(tmp_path)))
    assert msgs == []


def test_cross_file_one_approved_one_blocked(tmp_path):
    pkg = _make_pkg(tmp_path, "mypkg")

    a = pkg / "a.py"
    a.write_text("def approved_func():\n    return (1, 2, 3)\n", encoding="utf-8")

    b = pkg / "b.py"
    b.write_text("def blocked_func():\n    return (1, 2, 3)\n", encoding="utf-8")

    # outside.py calls blocked_func and is NOT in the diff.
    outside = pkg / "outside.py"
    outside.write_text(
        "from mypkg.b import blocked_func\nblocked_func()\n", encoding="utf-8"
    )

    changed = {str(a): [(1, 2)], str(b): [(1, 2)]}
    msgs = list(
        run_engine(
            changed, _repo_root=str(tmp_path), config=CrispenConfig(min_tuple_size=3)
        )
    )

    # blocked_func is skipped; its identity entry in alias_map hits the 349->348 branch.
    assert any(
        "blocked_func" in m and "callers exist outside the diff" in m for m in msgs
    )
    # approved_func is transformed.
    assert any("TupleDataclass" in m for m in msgs)


def test_cross_file_caller_updater_file_not_under_repo_root(tmp_path):
    subdir = tmp_path / "repo"
    subdir.mkdir()
    (subdir / "__init__.py").write_text("")

    inside = subdir / "service.py"
    inside.write_text("def approved():\n    return (1, 2, 3)\n", encoding="utf-8")

    # This file is in the diff but outside repo_root (subdir).
    outside_code = tmp_path / "outside_code.py"
    outside_code.write_text("x = 1\n", encoding="utf-8")

    changed = {str(inside): [(1, 2)], str(outside_code): [(1, 1)]}
    # No crash; outside_code.py's _file_to_module raises ValueError → continue.
    list(
        run_engine(
            changed, _repo_root=str(subdir), config=CrispenConfig(min_tuple_size=3)
        )
    )


def test_cross_file_caller_updater_parse_error(tmp_path):
    pkg = _make_pkg(tmp_path, "mypkg")

    service = pkg / "service.py"
    service.write_text("def approved():\n    return (1, 2, 3)\n", encoding="utf-8")

    changed = {str(service): [(1, 2)]}

    original_parse = cst.parse_module

    def patched_parse(source):
        # After Phase 2 transforms the source, it will contain "@dataclass".
        # Fail on that call to exercise the 374-375 parse-error branch.
        if "@dataclass" in source:
            raise cst.ParserSyntaxError(
                "fake error", lines=("@dataclass",), raw_line=0, raw_column=0
            )
        return original_parse(source)

    with patch("crispen.engine.cst.parse_module", patched_parse):
        # Should not crash; CallerUpdater pass silently continues.
        list(
            run_engine(
                changed,
                _repo_root=str(tmp_path),
                config=CrispenConfig(min_tuple_size=3),
            )
        )


def test_cross_file_caller_updater_raises(tmp_path):
    pkg = _make_pkg(tmp_path, "mypkg")

    service = pkg / "service.py"
    service.write_text("def approved():\n    return (1, 2, 3)\n", encoding="utf-8")

    changed = {str(service): [(1, 2)]}

    with patch("crispen.engine.CallerUpdater", side_effect=RuntimeError("fail")):
        # Should not crash; the exception is caught.
        list(
            run_engine(
                changed,
                _repo_root=str(tmp_path),
                config=CrispenConfig(min_tuple_size=3),
            )
        )


def test_cross_file_init_alias_detected_as_outside_caller(tmp_path):
    pkg = _make_pkg(tmp_path, "mypkg")

    # Re-export get_user through __init__.py
    (pkg / "__init__.py").write_text(
        "from mypkg.service import get_user\n", encoding="utf-8"
    )

    service = pkg / "service.py"
    service.write_text(
        "def get_user():\n    return (name, age, score)\n", encoding="utf-8"
    )

    # Outside file imports via the alias (pkg.get_user)
    outside = tmp_path / "outside.py"
    outside.write_text(
        "from mypkg import get_user\na, b, c = get_user()\n", encoding="utf-8"
    )

    changed = {str(service): [(1, 2)]}
    msgs = list(
        run_engine(
            changed, _repo_root=str(tmp_path), config=CrispenConfig(min_tuple_size=3)
        )
    )

    assert any("callers exist outside the diff" in m for m in msgs)


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


def test_phase2_apply_tuple_dataclass_td_none(tmp_path):
    """Phase 2 _apply_tuple_dataclass returning td=None is handled gracefully."""
    pkg = _make_pkg(tmp_path, "mypkg")

    service = pkg / "service.py"
    service.write_text("def approved():\n    return (1, 2, 3)\n", encoding="utf-8")

    orig_apply = _apply_tuple_dataclass
    call_count = {"n": 0}

    def patched_apply(filepath, ranges, source, verbose, approved_public_funcs, **kw):
        call_count["n"] += 1
        if call_count["n"] == 2:
            # Phase 2 call: return td=None to exercise the td2 is None branch
            return (source, [], None)
        return orig_apply(
            filepath, ranges, source, verbose, approved_public_funcs, **kw
        )

    with patch("crispen.engine._apply_tuple_dataclass", patched_apply):
        msgs = list(
            run_engine(
                {str(service): [(1, 2)]},
                _repo_root=str(tmp_path),
                config=CrispenConfig(min_tuple_size=3),
            )
        )
    # Should not crash; Phase 2 gracefully skips categorization
    assert isinstance(msgs, list)
