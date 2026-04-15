from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import run_engine
from crispen.errors import CrispenAPIError
from crispen.file_limiter.runner import FileLimiterResult
from crispen.stats import RunStats
import pytest
from .file_limiter import _FL_PATCH


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


def test_file_limiter_subdir_split_empty_source_file_already_deleted(tmp_path):
    """Subdir split deletes the original file; empty original_source skips re-unlink."""
    f = tmp_path / "big.py"
    original = "".join(f"var_{i} = {i}\n" for i in range(10))
    f.write_text(original, encoding="utf-8")
    # subdir_name causes Phase 3 to delete the original file.  original_source=""
    # means the per_file loop sees an empty source for a file that no longer
    # exists — exercising the elif-is-False branch (803→805).
    # new_files content kept short (≤ max_file_lines) to avoid recursive queue.
    subdir_result = FileLimiterResult(
        original_source="",
        new_files={"big/__init__.py": "# package\n"},
        messages=[f"{f}: FileLimiter: subdir split → big/"],
        abort=False,
        subdir_name="big",
    )
    with patch(_FL_PATCH, return_value=subdir_result):
        msgs = list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
            )
        )
    assert any("FileLimiter" in m for m in msgs)
    # Original file was deleted by the subdir split; must not exist.
    assert not f.exists()
    # Package init was created.
    assert (tmp_path / "big" / "__init__.py").exists()


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


def test_file_limiter_recursive_empty_init_py_preserved(tmp_path):
    """__init__.py created during recursive split is kept even when drained empty."""
    orig = tmp_path / "big.py"
    orig.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    oversized = "".join(f"x_{i} = {i}\n" for i in range(10))
    first_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"pkg/__init__.py": oversized},
        messages=[],
        abort=False,
    )
    # Recursive pass drains pkg/__init__.py; original_source is empty.
    second_result = FileLimiterResult(
        original_source="",
        new_files={"pkg/utils.py": "# utils\n"},
        messages=[],
        abort=False,
    )
    with patch(_FL_PATCH, side_effect=[first_result, second_result]):
        list(
            run_engine(
                {str(orig): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5, file_limiter_recursive=True),
            )
        )
    # pkg/__init__.py must survive as an empty file, not be deleted.
    assert (tmp_path / "pkg" / "__init__.py").exists()
    assert (tmp_path / "pkg" / "__init__.py").read_text(encoding="utf-8") == ""


def test_file_limiter_recursive_test_deletion_patches_parent_inline_imports(tmp_path):
    """When a recursive split deletes a test file, inline imports in the parent
    file that point to the deleted module are updated to the new locations."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

    # Parent test file with inline imports referencing a module that will be
    # created by the first split and then deleted by the recursive split.
    parent_src = (
        "def run_comprehensive_tests():\n"
        "    from tests.test_collections import TestA, TestB\n"
        "    TestA()\n"
        "    TestB()\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    run_comprehensive_tests()\n"
    )
    parent_file = tmp_path / "tests" / "test_suite.py"
    parent_file.parent.mkdir(parents=True)
    parent_file.write_text(parent_src, encoding="utf-8")

    # First pass: parent → creates test_collections.py (oversized).
    # The original source has the inline import already present (as if
    # _inject_inline_test_imports_original added it).
    first_result = FileLimiterResult(
        original_source=parent_src,  # unchanged (inline import already injected)
        new_files={
            "test_collections.py": (
                "class TestA:\n    pass\n\nclass TestB:\n    pass\n"
            )
        },
        messages=[],
        abort=False,
    )

    # Recursive pass: test_collections.py → split into sub/test_a.py + sub/test_b.py.
    # All entities migrated; original_source is empty → file will be deleted.
    recursive_result = FileLimiterResult(
        original_source="",
        new_files={
            "sub/test_a.py": "class TestA:\n    pass\n",
            "sub/test_b.py": "class TestB:\n    pass\n",
        },
        messages=[],
        abort=False,
    )

    with patch(_FL_PATCH, side_effect=[first_result, recursive_result]):
        list(
            run_engine(
                {str(parent_file): [(1, len(parent_src.splitlines()))]},
                config=CrispenConfig(max_file_lines=2, file_limiter_recursive=True),
            )
        )

    # test_collections.py was deleted.
    assert not (parent_file.parent / "test_collections.py").exists()

    # Parent file now has updated inline imports pointing to the new locations.
    updated = parent_file.read_text(encoding="utf-8")
    assert "from tests.sub.test_a import TestA" in updated
    assert "from tests.sub.test_b import TestB" in updated
    assert "from tests.test_collections import" not in updated


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
