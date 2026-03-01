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
from .test_phase1 import _run


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


# ---------------------------------------------------------------------------
# Cross-file integration: public function + caller both in diff
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cross-file: outside callers block the transform
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _build_alias_map: skip non-SimpleStatementLine and non-ImportFrom branches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _find_outside_callers: call resolves but qname not in targets (118->117)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _find_outside_callers: FullRepoManager build failure (143-145)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _find_outside_callers: wrapper.get_metadata_wrapper_for_path fails (154-155)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _apply_tuple_dataclass: parse error path (175-176)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _apply_tuple_dataclass: CrispenAPIError propagates (188)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 2: file not under repo_root → ValueError caught (314-315, 317->406)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 2: repo_root set but no public candidates (317->406)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 2: one approved, one blocked → alias loop hits non-approved (349->348)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CallerUpdater pass: file not under repo_root → ValueError (369-370)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CallerUpdater pass: parse error on state["source"] (374-375)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CallerUpdater pass: CallerUpdater constructor raises (387-388)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cross-file: __init__.py alias is recognised
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _visit_with_timeout
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _find_outside_callers: excluded directory names are not scanned
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 1 private-function caller updates
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _has_callers_outside_ranges
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _blocked_private_scopes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# run_engine: config parameter
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# update_diff_file_callers=False: private function blocked by outside callers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# update_diff_file_callers=False: public function blocked by diff-file outside callers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _categorize_into_stats
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# run_engine: stats parameter is populated
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 2 _apply_tuple_dataclass returning td=None (covers 579->567 branch)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FileLimiter (Phase 3 of engine)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _should_run
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Engine integration — enabled_refactors / disabled_refactors
# ---------------------------------------------------------------------------
