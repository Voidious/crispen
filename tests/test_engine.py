"""Tests for the engine module."""

import textwrap
from unittest.mock import patch

import libcst as cst
import pytest

from crispen.config import CrispenConfig
from crispen.engine import _apply_tuple_dataclass, run_engine
from crispen.errors import CrispenAPIError
from crispen.file_limiter.runner import FileLimiterResult
from crispen.stats import RunStats
from tests.engine.test_engine_core import _make_pkg


# ---------------------------------------------------------------------------
# File not found
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# No changes produced
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Successful transformation — writes file back
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Parse error
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Transform error
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CrispenAPIError propagates through engine
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TupleDataclass transform error: td is None (covers 290->293 branch)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _find_repo_root
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _file_to_module and _compute_qname
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _build_alias_map
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _find_outside_callers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cross-file integration: public function + caller both in diff
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# run_engine: stats parameter is populated
# ---------------------------------------------------------------------------


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


def test_file_limiter_subdir_split_non_test_deletes_original(tmp_path):
    """Non-test subdir split → original file deleted; __init__.py gets split content."""
    f = tmp_path / "service.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    success_result = FileLimiterResult(
        original_source=f.read_text(encoding="utf-8"),  # reset to original → no write
        new_files={
            "service/__init__.py": "# init\n",
            "service/utils.py": "# utils\n",
        },
        messages=[f"{f}: FileLimiter: moved foo → service/utils.py"],
        abort=False,
        subdir_name="service",
    )
    s = RunStats()
    with patch(_FL_PATCH, return_value=success_result):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
                stats=s,
            )
        )
    # Original service.py must be deleted.
    assert not f.exists()
    # Package files must exist.
    assert (tmp_path / "service" / "__init__.py").read_text(
        encoding="utf-8"
    ) == "# init\n"
    assert (tmp_path / "service" / "utils.py").read_text(
        encoding="utf-8"
    ) == "# utils\n"
    # All original lines must be counted as deleted so verified_lines ≤ lines_deleted.
    assert s.lines_deleted == 10


def test_file_limiter_subdir_split_test_keeps_original(tmp_path):
    """Test subdir split → original test file kept (not deleted)."""
    f = tmp_path / "test_service.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    re_export_src = "# re-exports\n"
    success_result = FileLimiterResult(
        original_source=re_export_src,
        new_files={"service/test_utils.py": "# test utils\n"},
        messages=[],
        abort=False,
        subdir_name="service",
    )
    with patch(_FL_PATCH, return_value=success_result):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
            )
        )
    # Original test file must still exist (with re-export content written back).
    assert f.exists()
    assert f.read_text(encoding="utf-8") == re_export_src


def test_file_limiter_subdir_split_has_main_keeps_original(tmp_path):
    """Non-test subdir split with has_main → original file kept and updated."""
    f = tmp_path / "service.py"
    original_src = "".join(f"var_{i} = {i}\n" for i in range(10))
    f.write_text(original_src, encoding="utf-8")
    re_export_src = (
        "from service_lib.utils import foo\n\nif __name__ == '__main__':\n    foo()\n"
    )
    success_result = FileLimiterResult(
        original_source=re_export_src,
        new_files={"service_lib/utils.py": "def foo():\n    pass\n"},
        messages=[],
        abort=False,
        subdir_name="service_lib",
        has_main=True,
    )
    with patch(_FL_PATCH, return_value=success_result):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
            )
        )
    # Original service.py must still exist (not deleted).
    assert f.exists()
    # It should be updated with the re-export stubs + __main__.
    assert f.read_text(encoding="utf-8") == re_export_src
    # New subdir file must exist.
    assert (tmp_path / "service_lib" / "utils.py").read_text(encoding="utf-8") == (
        "def foo():\n    pass\n"
    )


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


def test_file_limiter_recursive_splits_new_file(tmp_path):
    """When a new file from FileLimiter is over the limit, it is recursively split."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    # First call: original file → creates "chunk.py" which is still over the limit.
    first_result = FileLimiterResult(
        original_source="# reduced original\n",
        new_files={"chunk.py": "".join(f"x_{i} = {i}\n" for i in range(10))},
        messages=[f"{f}: FileLimiter: moved vars → chunk.py"],
        abort=False,
    )
    # Second call (recursive): chunk.py → creates "chunk_a.py" and "chunk_b.py".
    second_result = FileLimiterResult(
        original_source="# reduced chunk\n",
        new_files={"chunk_a.py": "# a\n", "chunk_b.py": "# b\n"},
        messages=[str(tmp_path / "chunk.py") + ": FileLimiter: moved → chunk_a/b"],
        abort=False,
    )

    call_count = 0

    def _fl_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return first_result
        return second_result

    with patch(_FL_PATCH, side_effect=_fl_side_effect):
        msgs = list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5, file_limiter_recursive=True),
            )
        )

    assert call_count == 2
    # Messages from the recursive call are yielded.
    assert any("chunk_a/b" in m for m in msgs)
    # Recursive split wrote additional files.
    assert (tmp_path / "chunk_a.py").exists()
    assert (tmp_path / "chunk_b.py").exists()
    # chunk.py was updated with the reduced source from the recursive split.
    assert (tmp_path / "chunk.py").read_text(encoding="utf-8") == "# reduced chunk\n"


def test_file_limiter_recursive_disabled(tmp_path):
    """file_limiter_recursive=False skips recursive processing of new files."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    oversized = "".join(f"x_{i} = {i}\n" for i in range(10))
    first_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"chunk.py": oversized},
        messages=[],
        abort=False,
    )

    call_count = 0

    def _fl_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        return first_result

    with patch(_FL_PATCH, side_effect=_fl_side_effect):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5, file_limiter_recursive=False),
            )
        )

    # Only one call: recursive processing was disabled.
    assert call_count == 1


def test_file_limiter_recursive_abort_stops_recursion(tmp_path):
    """Recursive call that aborts does not enqueue further files."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    oversized = "".join(f"x_{i} = {i}\n" for i in range(10))
    first_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"chunk.py": oversized},
        messages=[],
        abort=False,
    )
    abort_result = FileLimiterResult(
        original_source=oversized,
        new_files={},
        messages=["SKIP chunk.py (FileLimiter): cannot be split"],
        abort=True,
    )

    side_effects = [first_result, abort_result]

    with patch(_FL_PATCH, side_effect=side_effects):
        msgs = list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5, file_limiter_recursive=True),
            )
        )

    assert any("cannot be split" in m for m in msgs)


def test_file_limiter_recursive_api_error_propagates(tmp_path):
    """CrispenAPIError during recursive FileLimiter call propagates out."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    oversized = "".join(f"x_{i} = {i}\n" for i in range(10))
    first_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"chunk.py": oversized},
        messages=[],
        abort=False,
    )

    side_effects = [first_result, CrispenAPIError("rate limit")]

    with patch(_FL_PATCH, side_effect=side_effects):
        with pytest.raises(CrispenAPIError, match="rate limit"):
            list(
                run_engine(
                    {str(f): [(1, 1)]},
                    config=CrispenConfig(max_file_lines=5, file_limiter_recursive=True),
                )
            )


def test_file_limiter_recursive_creates_nested_init(tmp_path):
    """Recursive FileLimiter creating a file in a subdirectory creates __init__.py."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    oversized = "".join(f"x_{i} = {i}\n" for i in range(10))
    first_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"chunk.py": oversized},
        messages=[],
        abort=False,
    )
    # Recursive call creates a file in a subdirectory.
    second_result = FileLimiterResult(
        original_source="# reduced chunk\n",
        new_files={"sub/part.py": "# part\n"},
        messages=[],
        abort=False,
    )

    with patch(_FL_PATCH, side_effect=[first_result, second_result]):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5, file_limiter_recursive=True),
            )
        )

    assert (tmp_path / "sub" / "part.py").exists()
    assert (tmp_path / "sub" / "__init__.py").exists()


def test_file_limiter_recursive_chains(tmp_path):
    """A file created by a recursive call that is still over the limit is re-queued."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    oversized = "".join(f"x_{i} = {i}\n" for i in range(10))
    first_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"chunk.py": oversized},
        messages=[],
        abort=False,
    )
    # chunk.py recursive call itself creates another oversized file.
    second_result = FileLimiterResult(
        original_source="# reduced chunk\n",
        new_files={"chunk2.py": oversized},
        messages=[],
        abort=False,
    )
    third_result = FileLimiterResult(
        original_source="# reduced chunk2\n",
        new_files={},
        messages=[],
        abort=True,
    )

    with patch(_FL_PATCH, side_effect=[first_result, second_result, third_result]):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5, file_limiter_recursive=True),
            )
        )

    assert (tmp_path / "chunk.py").exists()
    assert (tmp_path / "chunk2.py").exists()


def test_file_limiter_recursive_source_unchanged(tmp_path):
    """Recursive result with same original_source does not rewrite the file."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    oversized = "".join(f"x_{i} = {i}\n" for i in range(10))
    first_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"chunk.py": oversized},
        messages=[],
        abort=False,
    )
    # Recursive call: original_source equals the input source → no rewrite.
    second_result = FileLimiterResult(
        original_source=oversized,  # same as what was written
        new_files={"part.py": "# part\n"},
        messages=[],
        abort=False,
    )

    with patch(_FL_PATCH, side_effect=[first_result, second_result]):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5, file_limiter_recursive=True),
            )
        )

    # chunk.py content is the oversized source (unchanged); the engine did not
    # rewrite it because original_source == r_source.
    assert (tmp_path / "chunk.py").read_text(encoding="utf-8") == oversized


def test_file_limiter_recursive_subdir_split_deletes_file(tmp_path):
    """Recursive FileLimiter subdir split on a non-test file deletes the file."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    oversized = "".join(f"x_{i} = {i}\n" for i in range(10))
    first_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"chunk.py": oversized},
        messages=[],
        abort=False,
    )
    # Recursive call triggers subdir split: chunk.py → chunk/ package.
    second_result = FileLimiterResult(
        original_source=oversized,
        new_files={"chunk/__init__.py": "# init\n", "chunk/utils.py": "# utils\n"},
        messages=[],
        abort=False,
        subdir_name="chunk",
    )

    with patch(_FL_PATCH, side_effect=[first_result, second_result]):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5, file_limiter_recursive=True),
            )
        )

    # chunk.py was deleted because subdir_name is set and it's not a test file.
    assert not (tmp_path / "chunk.py").exists()
    assert (tmp_path / "chunk" / "__init__.py").exists()


# ---------------------------------------------------------------------------
# _should_run
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Engine integration — enabled_refactors / disabled_refactors
# ---------------------------------------------------------------------------


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
