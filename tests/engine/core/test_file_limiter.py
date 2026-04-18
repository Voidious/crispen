from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import run_engine
from crispen.file_limiter.runner import FileLimiterResult
from crispen.stats import RunStats
from ..helpers import _FL_PATCH


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


def test_file_limiter_recursive_empty_original_source_deletes_file(tmp_path):
    """Recursive FileLimiter with empty original_source deletes the recursive file."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    oversized = "".join(f"x_{i} = {i}\n" for i in range(10))
    # chunk_a content is short so it doesn't re-enter the recursive queue.
    small = "# chunk_a content\n"
    first_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"chunk.py": oversized},
        messages=[],
        abort=False,
    )
    # Recursive call drains chunk.py entirely; original_source is empty.
    second_result = FileLimiterResult(
        original_source="",
        new_files={"chunk_a.py": small},
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

    # chunk.py was drained and must be deleted.
    assert not (tmp_path / "chunk.py").exists()
    # New file from the recursive split must exist.
    assert (tmp_path / "chunk_a.py").exists()


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


def test_file_limiter_recursive_llm_timing_recorded_in_stats(tmp_path):
    """Recursive FileLimiterResult with llm_elapsed > 0 triggers record_llm_call."""
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    first_result = FileLimiterResult(
        original_source="# reduced original\n",
        new_files={"chunk.py": "".join(f"x_{i} = {i}\n" for i in range(10))},
        messages=[f"{f}: moved vars → chunk.py"],
        abort=False,
    )
    second_result = FileLimiterResult(
        original_source="# reduced chunk\n",
        new_files={"chunk_a.py": "# a\n"},
        messages=[],
        abort=False,
        llm_elapsed=2.0,
        llm_input_tokens=200,
        llm_output_tokens=80,
    )
    stats = RunStats()
    call_count = 0

    def _fl_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        return first_result if call_count == 1 else second_result

    with patch(_FL_PATCH, side_effect=_fl_side_effect):
        list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5, file_limiter_recursive=True),
                stats=stats,
            )
        )
    assert "file_limiter" in stats.llm_elapsed_by_category
