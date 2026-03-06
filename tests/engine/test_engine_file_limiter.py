from unittest.mock import patch
import pytest
from crispen.config import CrispenConfig
from crispen.engine import run_engine
from crispen.errors import CrispenAPIError
from crispen.file_limiter.runner import FileLimiterResult

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
    with patch(_FL_PATCH, return_value=success_result):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
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
