from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import run_engine
from crispen.errors import CrispenAPIError
from crispen.file_limiter.runner import FileLimiterResult
from crispen.stats import RunStats
import pytest


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


def test_file_limiter_empty_original_source_deletes_file(tmp_path):
    """FileLimiter returns empty original_source → original file is deleted."""
    f = tmp_path / "big.py"
    original = "".join(f"var_{i} = {i}\n" for i in range(10))
    f.write_text(original, encoding="utf-8")
    # All content was moved out; original_source is empty (all entities migrated).
    # new_files content is kept short (≤ max_file_lines) so it doesn't re-enter
    # the recursive queue (file_limiter_recursive defaults to True).
    moved_source = "# moved content\n"
    drained_result = FileLimiterResult(
        original_source="",
        new_files={"utils.py": moved_source},
        messages=[f"{f}: FileLimiter: moved all → utils.py"],
        abort=False,
    )
    with patch(_FL_PATCH, return_value=drained_result):
        msgs = list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
            )
        )
    assert any("FileLimiter" in m for m in msgs)
    # Original file must be deleted, not left as a blank file.
    assert not f.exists()
    # New file must exist with the moved content.
    assert (tmp_path / "utils.py").exists()


def test_file_limiter_empty_init_py_preserved(tmp_path):
    """__init__.py is never deleted even when FileLimiter drains it to empty."""
    f = tmp_path / "__init__.py"
    original = "".join(f"def func_{i}():\n    pass\n\n" for i in range(10))
    f.write_text(original, encoding="utf-8")
    drained = FileLimiterResult(
        original_source="",
        new_files={"utils.py": "# moved\n"},
        messages=[f"{f}: FileLimiter: moved all → utils.py"],
        abort=False,
    )
    with patch(_FL_PATCH, return_value=drained):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
            )
        )
    # __init__.py must still exist (empty is fine; deletion would break the package).
    assert f.exists()
    assert f.read_text(encoding="utf-8") == ""


def test_file_limiter_llm_timing_recorded_in_stats(tmp_path):
    """When FileLimiterResult has llm_elapsed > 0, record_llm_call is invoked."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    timed_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"utils.py": "# new\n"},
        messages=[f"{f}: FileLimiter: moved → utils.py"],
        abort=False,
        llm_elapsed=1.5,
        llm_input_tokens=100,
        llm_output_tokens=50,
    )
    stats = RunStats()
    with patch(_FL_PATCH, return_value=timed_result):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
                stats=stats,
            )
        )
    assert "file_limiter" in stats.llm_elapsed_by_category
