"""Tests for the engine module."""

import textwrap
import threading
from unittest.mock import patch

import libcst as cst
import pytest

from crispen.config import CrispenConfig
from crispen.engine import (
    _EXCLUDED_DIR_NAMES,
    _apply_tuple_dataclass,
    _blocked_private_scopes,
    _build_alias_map,
    _categorize_into_stats,
    _compute_qname,
    _file_to_module,
    _find_outside_callers,
    _find_repo_root,
    _has_callers_outside_ranges,
    _should_run,
    _visit_with_timeout,
    run_engine,
)
from crispen.errors import CrispenAPIError
from crispen.file_limiter.runner import FileLimiterResult
from crispen.refactors.base import Refactor
from crispen.stats import RunStats


def _run(changed):
    return list(run_engine(changed, config=CrispenConfig(min_tuple_size=3)))


# ---------------------------------------------------------------------------
# File not found
# ---------------------------------------------------------------------------


def test_skip_missing_file(tmp_path):
    missing = str(tmp_path / "nonexistent.py")
    msgs = _run({missing: [(1, 10)]})
    assert len(msgs) == 1
    assert "SKIP" in msgs[0]
    assert "file not found" in msgs[0]


# ---------------------------------------------------------------------------
# No changes produced
# ---------------------------------------------------------------------------


def test_no_changes_no_messages(tmp_path):
    f = tmp_path / "simple.py"
    f.write_text("x = 1\n", encoding="utf-8")
    msgs = _run({str(f): [(1, 1)]})
    assert msgs == []


# ---------------------------------------------------------------------------
# Successful transformation — writes file back
# ---------------------------------------------------------------------------


def test_applies_refactor_and_writes(tmp_path):
    source = textwrap.dedent(
        """\
        if not x:
            a()
        else:
            b()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    msgs = _run({str(f): [(1, 4)]})
    assert any("IfNotElse" in m for m in msgs)
    assert "if x:" in f.read_text(encoding="utf-8")


def test_rewritten_source_used_when_available(tmp_path):
    """get_rewritten_source() is preferred over new_tree.code when non-None."""
    rewritten = "x = 999  # rewritten\n"

    class _RewritingRefactor(Refactor):
        @classmethod
        def name(cls):
            return "Rewriter"

        def get_rewritten_source(self):
            return rewritten

        def get_changes(self):
            return ["Rewriter: rewrote the file"]

    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch("crispen.engine._REFACTORS", [_RewritingRefactor]):
        msgs = _run({str(f): [(1, 1)]})
    assert any("Rewriter" in m for m in msgs)
    assert f.read_text(encoding="utf-8") == rewritten


# ---------------------------------------------------------------------------
# Parse error
# ---------------------------------------------------------------------------


def test_skip_parse_error(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def f(:\n    pass\n", encoding="utf-8")
    msgs = _run({str(f): [(1, 2)]})
    assert any("parse error" in m for m in msgs)


# ---------------------------------------------------------------------------
# Transform error
# ---------------------------------------------------------------------------


class _RaisingTransformer(Refactor):
    """A Refactor subclass that always raises during tree traversal."""

    @classmethod
    def name(cls):
        return "RaisingRefactor"

    def leave_Module(self, original_node, updated_node):
        raise RuntimeError("intentional transform error")


def test_skip_transform_error(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch("crispen.engine._REFACTORS", [_RaisingTransformer]):
        msgs = _run({str(f): [(1, 1)]})
    assert any("transform error" in m for m in msgs)


# ---------------------------------------------------------------------------
# CrispenAPIError propagates through engine
# ---------------------------------------------------------------------------


class _CrispenApiErrorRefactor(Refactor):
    @classmethod
    def name(cls):
        return "ApiErrorRefactor"

    def leave_Module(self, original_node, updated_node):
        raise CrispenAPIError("test api error")


def test_crispen_api_error_propagates(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch("crispen.engine._REFACTORS", [_CrispenApiErrorRefactor]):
        with pytest.raises(CrispenAPIError):
            list(run_engine({str(f): [(1, 1)]}))


# ---------------------------------------------------------------------------
# TupleDataclass transform error: td is None (covers 290->293 branch)
# ---------------------------------------------------------------------------


def test_tuple_dataclass_transform_error_handled(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")

    class _FailingTD:
        def __init__(self, *a, **kw):
            raise RuntimeError("simulated TupleDataclass failure")

    with patch("crispen.engine.TupleDataclass", _FailingTD):
        msgs = _run({str(f): [(1, 1)]})
    assert any("TupleDataclass" in m and "transform error" in m for m in msgs)


# ---------------------------------------------------------------------------
# _find_repo_root
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _file_to_module and _compute_qname
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _build_alias_map
# ---------------------------------------------------------------------------


def test_build_alias_map_identity_only(tmp_path):
    # No __init__.py in tmp_path → only identity mapping returned.
    alias_map = _build_alias_map(str(tmp_path), {"a.b.func"})
    assert alias_map == {"a.b.func": "a.b.func"}


def test_build_alias_map_with_reexport(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mypkg.service import get_user\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    assert "mypkg.get_user" in alias_map
    assert alias_map["mypkg.get_user"] == "mypkg.service.get_user"


def test_build_alias_map_star_import_skipped(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mypkg.service import *\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    # Star import does not create an alias
    assert "mypkg.get_user" not in alias_map


def test_build_alias_map_ambiguous_name_skipped(tmp_path):
    # Two canonical qnames share the same function name → alias is ambiguous.
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mypkg.service import get_user\n")
    alias_map = _build_alias_map(
        str(tmp_path),
        {"mypkg.service.get_user", "mypkg.other.get_user"},
    )
    # Ambiguous: skip adding the alias
    assert "mypkg.get_user" not in alias_map


def test_build_alias_map_invalid_init_skipped(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def f(:\n    pass\n")  # invalid Python
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    # Gracefully skips the unreadable __init__.py
    assert alias_map == {"mypkg.service.get_user": "mypkg.service.get_user"}


# ---------------------------------------------------------------------------
# _find_outside_callers
# ---------------------------------------------------------------------------


def test_find_outside_callers_empty_qnames(tmp_path):
    result = _find_outside_callers(str(tmp_path), set(), set())
    assert result == set()


def test_find_outside_callers_no_outside_py_files(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n")
    result = _find_outside_callers(str(tmp_path), {"pkg.func"}, {str(f.resolve())})
    # All .py files are in the diff → nothing to scan outside
    assert result == set()


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


def test_find_outside_callers_no_match(tmp_path):
    outside = tmp_path / "other.py"
    outside.write_text("x = 1\n")
    qname = "mypkg.service.get_user"
    result = _find_outside_callers(str(tmp_path), {qname}, set())
    assert qname not in result


# ---------------------------------------------------------------------------
# Cross-file integration: public function + caller both in diff
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Cross-file: outside callers block the transform
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _build_alias_map: skip non-SimpleStatementLine and non-ImportFrom branches
# ---------------------------------------------------------------------------


def test_build_alias_map_skips_compound_statement(tmp_path):
    # A function definition is a compound statement, not SimpleStatementLine (line 76).
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def helper():\n    pass\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    assert alias_map == {"mypkg.service.get_user": "mypkg.service.get_user"}


def test_build_alias_map_skips_non_import_in_simple_stmt(tmp_path):
    # An assignment in SimpleStatementLine is not ImportFrom (line 79).
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("__version__ = '1.0'\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    assert alias_map == {"mypkg.service.get_user": "mypkg.service.get_user"}


# ---------------------------------------------------------------------------
# _find_outside_callers: call resolves but qname not in targets (118->117)
# ---------------------------------------------------------------------------


def test_find_outside_callers_call_qname_not_target(tmp_path):
    # outside file calls other_func (resolves to mypkg.other.other_func),
    # but target is mypkg.service.get_user → hits the 118->117 branch.
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "other.py").write_text("def other_func(): pass\n")
    caller = tmp_path / "caller.py"
    caller.write_text("from mypkg.other import other_func\nother_func()\n")

    result = _find_outside_callers(str(tmp_path), {"mypkg.service.get_user"}, set())
    assert "mypkg.service.get_user" not in result


# ---------------------------------------------------------------------------
# _find_outside_callers: FullRepoManager build failure (143-145)
# ---------------------------------------------------------------------------


def test_find_outside_callers_manager_build_fails(tmp_path):
    (tmp_path / "other.py").write_text("x = 1\n")
    with patch("crispen.engine.FullRepoManager", side_effect=RuntimeError("fail")):
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    # Conservative: all target qnames are blocked.
    assert result == {"some.func"}


# ---------------------------------------------------------------------------
# _find_outside_callers: wrapper.get_metadata_wrapper_for_path fails (154-155)
# ---------------------------------------------------------------------------


def test_find_outside_callers_wrapper_fails(tmp_path):
    (tmp_path / "other.py").write_text("x = 1\n")
    with patch("crispen.engine.FullRepoManager") as MockFRM:
        MockFRM.return_value.get_metadata_wrapper_for_path.side_effect = RuntimeError(
            "fail"
        )
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    assert result == set()


# ---------------------------------------------------------------------------
# _apply_tuple_dataclass: parse error path (175-176)
# ---------------------------------------------------------------------------


def test_apply_tuple_dataclass_parse_error():
    bad_source = "def f(:\n    pass\n"
    source_out, msgs, td = _apply_tuple_dataclass(
        "fake.py", [(1, 10)], bad_source, False, set()
    )
    assert any("parse error" in m for m in msgs)
    assert td is None
    assert source_out == bad_source


# ---------------------------------------------------------------------------
# _apply_tuple_dataclass: CrispenAPIError propagates (188)
# ---------------------------------------------------------------------------


def test_apply_tuple_dataclass_crispen_api_error():
    with patch("crispen.engine.MetadataWrapper") as MockWrapper:
        MockWrapper.return_value.visit.side_effect = CrispenAPIError("test api error")
        with pytest.raises(CrispenAPIError):
            _apply_tuple_dataclass("f.py", [(1, 1)], "x = 1\n", False, set())


# ---------------------------------------------------------------------------
# Phase 2: file not under repo_root → ValueError caught (314-315, 317->406)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 2: repo_root set but no public candidates (317->406)
# ---------------------------------------------------------------------------


def test_no_public_candidates_with_repo_root(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    msgs = list(run_engine({str(f): [(1, 1)]}, _repo_root=str(tmp_path)))
    assert msgs == []


# ---------------------------------------------------------------------------
# Phase 2: one approved, one blocked → alias loop hits non-approved (349->348)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CallerUpdater pass: file not under repo_root → ValueError (369-370)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CallerUpdater pass: parse error on state["source"] (374-375)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CallerUpdater pass: CallerUpdater constructor raises (387-388)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Cross-file: __init__.py alias is recognised
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _visit_with_timeout
# ---------------------------------------------------------------------------


def test_visit_with_timeout_completes():
    """Fast visit completes within timeout → returns True."""
    from unittest.mock import MagicMock

    wrapper = MagicMock()
    finder = MagicMock()
    assert _visit_with_timeout(wrapper, finder, 5.0) is True
    wrapper.visit.assert_called_once_with(finder)


def test_visit_with_timeout_fires():
    """Slow visit that never completes → returns False after timeout."""
    block = threading.Event()

    class _HangWrapper:
        def visit(self, finder):
            block.wait()  # blocks until released

    result = _visit_with_timeout(_HangWrapper(), object(), 0.01)
    block.set()  # unblock the daemon thread for cleanup
    assert result is False


def test_find_outside_callers_scope_analysis_timeout(tmp_path):
    """When _visit_with_timeout times out, all target qnames are blocked."""
    (tmp_path / "other.py").write_text("x = 1\n")
    with patch("crispen.engine._visit_with_timeout", return_value=False):
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    assert result == {"some.func"}


def test_find_outside_callers_deadline_expired(tmp_path):
    """Total budget already exhausted before any file is visited: all blocked."""
    (tmp_path / "other.py").write_text("x = 1\n")
    # A negative timeout makes the deadline fall in the past immediately.
    with patch("crispen.engine._SCOPE_ANALYSIS_TIMEOUT", -1):
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    assert result == {"some.func"}


# ---------------------------------------------------------------------------
# _find_outside_callers: excluded directory names are not scanned
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 1 private-function caller updates
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _has_callers_outside_ranges
# ---------------------------------------------------------------------------


def test_has_callers_outside_ranges_found():
    source = "def f(): pass\nf()\n"  # call on line 2, range is only line 1
    assert _has_callers_outside_ranges(source, "f", [(1, 1)]) is True


def test_has_callers_outside_ranges_not_found():
    source = "def f(): pass\nf()\n"  # call on line 2, range covers line 2
    assert _has_callers_outside_ranges(source, "f", [(1, 2)]) is False


def test_has_callers_outside_ranges_syntax_error():
    assert _has_callers_outside_ranges("def f(:", "f", [(1, 1)]) is False


# ---------------------------------------------------------------------------
# _blocked_private_scopes
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# run_engine: config parameter
# ---------------------------------------------------------------------------


def test_run_engine_accepts_explicit_config(tmp_path):
    """run_engine works when config is provided explicitly."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    config = CrispenConfig()
    msgs = list(run_engine({str(f): [(1, 1)]}, config=config))
    assert msgs == []


def test_run_engine_config_none_loads_default(tmp_path):
    """run_engine with config=None (default) loads config from disk."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    # config=None triggers load_config() internally
    msgs = list(run_engine({str(f): [(1, 1)]}, config=None))
    assert msgs == []


# ---------------------------------------------------------------------------
# update_diff_file_callers=False: private function blocked by outside callers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# update_diff_file_callers=False: public function blocked by diff-file outside callers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _categorize_into_stats
# ---------------------------------------------------------------------------


def test_categorize_if_not_else():
    s = RunStats()
    _categorize_into_stats(s, "IfNotElse: flipped if/else at line 3")
    assert s.if_not_else == 1
    assert s.total_edits == 1


def test_categorize_tuple_to_dataclass():
    s = RunStats()
    _categorize_into_stats(
        s, "TupleDataclass: replaced 3-tuple with FooResult at line 5"
    )
    assert s.tuple_to_dataclass == 1


def test_categorize_duplicate_matched():
    s = RunStats()
    _categorize_into_stats(s, "DuplicateExtractor: replaced '_f' body with call to 'g'")
    assert s.duplicate_matched == 1
    assert s.duplicate_extracted == 0


def test_categorize_duplicate_extracted():
    s = RunStats()
    _categorize_into_stats(
        s, "DuplicateExtractor: extracted '_helper' from 2 duplicate blocks"
    )
    assert s.duplicate_extracted == 1
    assert s.duplicate_matched == 0


def test_categorize_function_split():
    s = RunStats()
    _categorize_into_stats(s, "split 'big_func': extracted _step_two")
    assert s.function_split == 1


def test_categorize_other_message_ignored():
    s = RunStats()
    _categorize_into_stats(s, "CallerUpdater: expanded FooResult unpacking at line 7")
    assert s.total_edits == 0


# ---------------------------------------------------------------------------
# run_engine: stats parameter is populated
# ---------------------------------------------------------------------------


def test_run_engine_stats_populated(tmp_path):
    source = "if not x:\n    a()\nelse:\n    b()\n"
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    s = RunStats()
    list(run_engine({str(f): [(1, 4)]}, config=CrispenConfig(), stats=s))
    assert s.if_not_else == 1
    assert s.files_edited == [str(f)]
    assert s.lines_changed > 0


def test_run_engine_stats_none_default(tmp_path):
    """When stats is None (default), engine runs without error."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    msgs = list(run_engine({str(f): [(1, 1)]}, config=CrispenConfig()))
    assert msgs == []


# ---------------------------------------------------------------------------
# Phase 2 _apply_tuple_dataclass returning td=None (covers 579->567 branch)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# FileLimiter (Phase 3 of engine)
# ---------------------------------------------------------------------------

_FL_PATCH = "crispen.engine.run_file_limiter"


def test_file_limiter_disabled_by_max_file_lines_zero(tmp_path):
    """max_file_lines=0 disables FileLimiter entirely (branch: if > 0 is False)."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    with patch(_FL_PATCH) as mock_fl:
        list(run_engine({str(f): [(1, 1)]}, config=CrispenConfig(max_file_lines=0)))
    mock_fl.assert_not_called()


def test_file_limiter_skips_short_file(tmp_path):
    """File under max_file_lines → FileLimiter is not called for that file."""
    f = tmp_path / "short.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch(_FL_PATCH) as mock_fl:
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=100),
            )
        )
    mock_fl.assert_not_called()


def test_file_limiter_abort_adds_skip_message(tmp_path):
    """FileLimiter abort → SKIP message added; no new files written."""
    f = tmp_path / "big.py"
    original = "".join(f"var_{i} = {i}\n" for i in range(10))
    f.write_text(original, encoding="utf-8")
    abort_result = FileLimiterResult(
        original_source=original,
        new_files={},
        messages=[f"SKIP {f} (FileLimiter): file cannot be split"],
        abort=True,
    )
    with patch(_FL_PATCH, return_value=abort_result):
        msgs = list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
            )
        )
    assert any("SKIP" in m and "FileLimiter" in m for m in msgs)
    assert not (tmp_path / "utils.py").exists()


def test_file_limiter_no_messages_no_new_files(tmp_path):
    """FileLimiter returns empty messages + no new files → no output, no writes."""
    f = tmp_path / "big.py"
    original = "".join(f"var_{i} = {i}\n" for i in range(10))
    f.write_text(original, encoding="utf-8")
    no_op_result = FileLimiterResult(
        original_source=original,
        new_files={},
        messages=[],
        abort=False,
    )
    with patch(_FL_PATCH, return_value=no_op_result):
        msgs = list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
            )
        )
    assert not any("FileLimiter" in m for m in msgs)


def test_file_limiter_success_writes_new_file(tmp_path):
    """FileLimiter success → new file written, original source updated."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    success_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"utils.py": "# new file\n"},
        messages=[f"{f}: FileLimiter: moved foo → utils.py"],
        abort=False,
    )
    with patch(_FL_PATCH, return_value=success_result):
        msgs = list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
            )
        )
    assert any("FileLimiter" in m for m in msgs)
    new_file = tmp_path / "utils.py"
    assert new_file.exists()
    assert new_file.read_text(encoding="utf-8") == "# new file\n"
    # Original file updated with reduced source.
    assert f.read_text(encoding="utf-8") == "# reduced\n"


def test_file_limiter_creates_nested_directory(tmp_path):
    """FileLimiter target in subdir → parent dirs and __init__.py created."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    success_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"helpers/utils.py": "# helpers\n"},
        messages=[f"{f}: FileLimiter: moved bar → helpers/utils.py"],
        abort=False,
    )
    with patch(_FL_PATCH, return_value=success_result):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
            )
        )
    nested = tmp_path / "helpers" / "utils.py"
    assert nested.exists()
    assert nested.read_text(encoding="utf-8") == "# helpers\n"
    # Subdirectory is initialised as a Python package.
    assert (tmp_path / "helpers" / "__init__.py").exists()


def test_file_limiter_existing_init_not_overwritten(tmp_path):
    """If the target subdir already has __init__.py, it is not overwritten."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    (helpers / "__init__.py").write_text("# existing\n", encoding="utf-8")
    success_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"helpers/utils.py": "# utils\n"},
        messages=[],
        abort=False,
    )
    with patch(_FL_PATCH, return_value=success_result):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
            )
        )
    assert (helpers / "__init__.py").read_text(encoding="utf-8") == "# existing\n"


def test_file_limiter_api_error_propagates(tmp_path):
    """CrispenAPIError from FileLimiter propagates out of run_engine."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    with patch(_FL_PATCH, side_effect=CrispenAPIError("rate limit")):
        with pytest.raises(CrispenAPIError, match="rate limit"):
            list(
                run_engine(
                    {str(f): [(1, 1)]},
                    config=CrispenConfig(max_file_lines=5),
                )
            )


# ---------------------------------------------------------------------------
# _should_run
# ---------------------------------------------------------------------------


def test_should_run_defaults_allow_all():
    cfg = CrispenConfig()
    for name in (
        "if_not_else",
        "duplicate_extractor",
        "function_splitter",
        "tuple_dataclass",
        "file_limiter",
    ):
        assert _should_run(name, cfg) is True


def test_should_run_enabled_list_allows_listed():
    cfg = CrispenConfig(enabled_refactors=["if_not_else", "function_splitter"])
    assert _should_run("if_not_else", cfg) is True
    assert _should_run("function_splitter", cfg) is True


def test_should_run_enabled_list_blocks_unlisted():
    cfg = CrispenConfig(enabled_refactors=["if_not_else"])
    assert _should_run("duplicate_extractor", cfg) is False
    assert _should_run("tuple_dataclass", cfg) is False
    assert _should_run("file_limiter", cfg) is False


def test_should_run_disabled_list_blocks_listed():
    cfg = CrispenConfig(disabled_refactors=["function_splitter", "file_limiter"])
    assert _should_run("function_splitter", cfg) is False
    assert _should_run("file_limiter", cfg) is False


def test_should_run_disabled_list_allows_unlisted():
    cfg = CrispenConfig(disabled_refactors=["function_splitter"])
    assert _should_run("if_not_else", cfg) is True
    assert _should_run("tuple_dataclass", cfg) is True


def test_should_run_enabled_takes_precedence_over_disabled():
    # enabled_refactors non-empty → disabled_refactors is ignored
    cfg = CrispenConfig(
        enabled_refactors=["if_not_else"],
        disabled_refactors=["if_not_else"],
    )
    assert _should_run("if_not_else", cfg) is True


# ---------------------------------------------------------------------------
# Engine integration — enabled_refactors / disabled_refactors
# ---------------------------------------------------------------------------


def test_engine_disabled_refactors_skips_if_not_else(tmp_path):
    """With if_not_else disabled the pattern is left unchanged."""
    source = textwrap.dedent(
        """\
        if not x:
            a()
        else:
            b()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    msgs = list(
        run_engine(
            {str(f): [(1, 4)]},
            config=CrispenConfig(disabled_refactors=["if_not_else"]),
        )
    )
    assert not any("IfNotElse" in m for m in msgs)
    assert f.read_text(encoding="utf-8") == source


def test_engine_enabled_refactors_runs_only_listed(tmp_path):
    """enabled_refactors=["if_not_else"] — other refactors don't touch the file."""
    source = textwrap.dedent(
        """\
        if not x:
            a()
        else:
            b()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")

    called = []

    class _Spy(Refactor):
        @classmethod
        def name(cls):
            return "Spy"

        def get_changes(self):
            called.append("Spy")
            return []

    with patch("crispen.engine._REFACTORS", [_Spy]):
        with patch("crispen.engine._REFACTOR_KEY", {_Spy: "spy"}):
            list(
                run_engine(
                    {str(f): [(1, 4)]},
                    config=CrispenConfig(enabled_refactors=["if_not_else"]),
                )
            )

    # _Spy is not in enabled_refactors, so it must not have been called.
    assert called == []


def test_engine_file_limiter_skipped_when_disabled(tmp_path):
    """file_limiter in disabled_refactors prevents FileLimiter from running."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    success_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"utils.py": "# new\n"},
        messages=["FileLimiter: moved"],
        abort=False,
    )
    with patch(_FL_PATCH, return_value=success_result) as mock_fl:
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    disabled_refactors=["file_limiter"],
                ),
            )
        )
    mock_fl.assert_not_called()


def test_engine_match_function_disabled_passes_flag_to_duplicate_extractor(tmp_path):
    """disabled_refactors=["match_function"] passes match_functions=False to DE."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")

    constructed_with: dict = {}

    original_init = __import__(
        "crispen.refactors.duplicate_extractor", fromlist=["DuplicateExtractor"]
    ).DuplicateExtractor.__init__

    def _spy_init(self, *args, **kwargs):
        constructed_with.update(kwargs)
        original_init(self, *args, **kwargs)

    with patch("crispen.engine.DuplicateExtractor.__init__", side_effect=_spy_init):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(disabled_refactors=["match_function"]),
            )
        )

    assert constructed_with.get("match_functions") is False


def test_engine_match_function_enabled_by_default(tmp_path):
    """Without any filter, match_functions=True is passed to DuplicateExtractor."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")

    constructed_with: dict = {}

    original_init = __import__(
        "crispen.refactors.duplicate_extractor", fromlist=["DuplicateExtractor"]
    ).DuplicateExtractor.__init__

    def _spy_init(self, *args, **kwargs):
        constructed_with.update(kwargs)
        original_init(self, *args, **kwargs)

    with patch("crispen.engine.DuplicateExtractor.__init__", side_effect=_spy_init):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(),
            )
        )

    assert constructed_with.get("match_functions") is True
