from __future__ import annotations
from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import SplitResult
from crispen.file_limiter.runner import (
    _MAIN_SUBDIR_SUFFIXES,
    _has_main_block,
    _is_whole_file_diff,
    run_file_limiter,
)
from .test_runner_core import (
    _CONFIG_NO_RETRY,
    _PATCH_ADVISE,
    _PATCH_CLASSIFY,
    _PATCH_GEN,
    _make_classified,
    _make_entity,
)


# A two-line source whose diff_ranges covers the whole file, triggering subdir
# split for "big.py" → subdir_name="big".  Path("big") must not exist on disk.
_SUBDIR_SRC = "x = 1\ny = 2\n"
_SUBDIR_RANGES = [(1, 2)]


def _plan_two_same_target() -> FileLimiterPlan:
    """Two groups, both assigned to the same target file."""
    return FileLimiterPlan(
        set3_migrate=[],
        placements=[
            GroupPlacement(group=["foo"], target_file="utils.py"),
            GroupPlacement(group=["bar"], target_file="utils.py"),
        ],
        abort=False,
    )


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_all_in_one_file_subdir_retries_and_fails(
    mock_classify, mock_advise, mock_gen
):
    # Subdir split + all groups → same file → guard triggers every attempt.
    # Two groups required so the n_groups > 1 pre-loop check doesn't fire first.
    mock_classify.return_value = ClassifiedEntities(
        entities=[_make_entity("foo", 1, 1), _make_entity("bar", 2, 2)],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[["foo"], ["bar"]],
        set_3_groups=[],
        abort=False,
    )
    mock_advise.return_value = _plan_two_same_target()
    cfg = CrispenConfig(file_limiter_retries=0)

    result = run_file_limiter("big.py", "", _SUBDIR_SRC, _SUBDIR_RANGES, cfg)

    assert result.abort is False
    assert any("single file" in m for m in result.messages)
    mock_gen.assert_not_called()


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_all_in_one_file_subdir_retries_and_succeeds(
    mock_classify, mock_advise, mock_gen
):
    # Subdir split: first attempt all in one file, second splits into two.
    entity1 = _make_entity("foo", 1, 1)
    entity2 = _make_entity("bar", 2, 2)
    # Two groups required so the n_groups > 1 pre-loop check doesn't fire first.
    mock_classify.return_value = ClassifiedEntities(
        entities=[entity1, entity2],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[["foo"], ["bar"]],
        set_3_groups=[],
        abort=False,
    )
    mock_advise.side_effect = [
        _plan_two_same_target(),
        FileLimiterPlan(
            set3_migrate=[],
            placements=[
                GroupPlacement(group=["foo"], target_file="utils.py"),
                GroupPlacement(group=["bar"], target_file="helpers.py"),
            ],
            abort=False,
        ),
    ]
    mock_gen.return_value = SplitResult(
        new_files={
            "big/utils.py": "x = 1",
            "big/helpers.py": "y = 2",
        },
        original_source=_SUBDIR_SRC,
        abort=False,
    )
    cfg = CrispenConfig(file_limiter_retries=1)

    result = run_file_limiter("big.py", "", _SUBDIR_SRC, _SUBDIR_RANGES, cfg)

    assert result.abort is False
    assert any("single file" in m for m in result.messages)
    assert any("FileLimiter: moved" in m for m in result.messages)
    assert mock_advise.call_args_list[1].kwargs["prev_placement_failure"] != ""


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_all_in_one_file_non_subdir_allowed(
    mock_classify, mock_advise, mock_gen
):
    # Non-subdir split: all groups → same file is always fine.
    entity1 = _make_entity("foo", 1, 2)
    entity2 = _make_entity("bar", 3, 4)
    mock_classify.return_value = _make_classified(entities=[entity1, entity2])
    mock_advise.return_value = FileLimiterPlan(
        set3_migrate=[],
        placements=[
            GroupPlacement(group=["foo"], target_file="utils.py"),
            GroupPlacement(group=["bar"], target_file="utils.py"),
        ],
        abort=False,
    )
    mock_gen.return_value = SplitResult(
        new_files={"utils.py": "def foo():\n    pass\ndef bar():\n    pass"},
        original_source="# reduced\n",
        abort=False,
    )

    # diff_ranges=[] → not a whole-file diff → subdir_name=None → guard inactive.
    result = run_file_limiter(
        "big.py",
        "",
        "def foo():\n    pass\ndef bar():\n    pass\n",
        [],
        _CONFIG_NO_RETRY,
    )

    assert result.abort is False
    assert not any("single file" in m for m in result.messages)
    mock_gen.assert_called_once()


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_single_group_subdir_aborts_silently(
    mock_classify, mock_advise, mock_gen
):
    # Subdir split with only 1 group: moving it would just rename the file,
    # not split it, causing infinite subdirectory nesting across runs.
    # Abort immediately without calling the LLM.
    entity = _make_entity("foo", 1, 1)
    mock_classify.return_value = ClassifiedEntities(
        entities=[entity],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[["foo"]],
        set_3_groups=[],
        abort=False,
    )
    cfg = CrispenConfig(file_limiter_retries=0)

    result = run_file_limiter("big.py", "", _SUBDIR_SRC, _SUBDIR_RANGES, cfg)

    assert result.abort is True
    assert result.messages == []
    mock_advise.assert_not_called()
    mock_gen.assert_not_called()


def test_is_whole_file_diff_empty_ranges():
    assert _is_whole_file_diff([], 5) is False


def test_is_whole_file_diff_zero_lines():
    assert _is_whole_file_diff([(1, 3)], 0) is False


def test_is_whole_file_diff_gap():
    # Lines 1-2 and 4-5 — line 3 is missing.
    assert _is_whole_file_diff([(1, 2), (4, 5)], 5) is False


def test_is_whole_file_diff_doesnt_start_at_one():
    # Range starts at line 2 — line 1 is not covered.
    assert _is_whole_file_diff([(2, 5)], 5) is False


def test_is_whole_file_diff_partial_coverage():
    # Covers lines 1-3 but file has 5 lines.
    assert _is_whole_file_diff([(1, 3)], 5) is False


def test_is_whole_file_diff_exact_coverage():
    assert _is_whole_file_diff([(1, 5)], 5) is True


def test_is_whole_file_diff_multi_range_contiguous():
    # Two adjacent ranges that together cover 1..5.
    assert _is_whole_file_diff([(1, 3), (4, 5)], 5) is True


def test_is_whole_file_diff_overshoots():
    # Ranges cover more lines than n_lines — still counts as whole-file.
    assert _is_whole_file_diff([(1, 10)], 5) is True


def test_has_main_block_detects_dunder_main():
    src = "def foo():\n    pass\n\nif __name__ == '__main__':\n    foo()\n"
    assert _has_main_block(src) is True


def test_has_main_block_no_main():
    assert _has_main_block("def foo():\n    pass\n") is False


def test_has_main_block_syntax_error():
    assert _has_main_block("def (:\n") is False


@patch(_PATCH_CLASSIFY)
def test_runner_subdir_split_dir_exists_aborts(mock_classify, tmp_path):
    mock_classify.return_value = _make_classified()
    # Create a directory named 'service' alongside the source file.
    service_dir = tmp_path / "service"
    service_dir.mkdir()
    filepath = str(tmp_path / "service.py")

    source = "def foo():\n    pass\n"
    # Whole-file diff: ranges cover all 2 lines.
    cfg = CrispenConfig(file_limiter_subdir_split=True)
    result = run_file_limiter(filepath, source, source, [(1, 2)], cfg)

    assert result.abort is True
    assert result.new_files == {}
    assert any("already exists" in m for m in result.messages)
    assert any("service/" in m for m in result.messages)


@patch(_PATCH_CLASSIFY)
def test_runner_subdir_split_sibling_py_aborts(mock_classify, tmp_path):
    mock_classify.return_value = _make_classified()
    # Create a sibling 'service.py' alongside the source file — the intended
    # subdirectory 'service/' would shadow it.
    (tmp_path / "service.py").write_text("# helper\n")
    filepath = str(tmp_path / "test_service.py")

    source = "def test_foo():\n    pass\n"
    # Whole-file diff: ranges cover all 2 lines.
    cfg = CrispenConfig(file_limiter_subdir_split=True)
    result = run_file_limiter(filepath, source, source, [(1, 2)], cfg)

    assert result.abort is True
    assert result.new_files == {}
    assert any("shadow" in m for m in result.messages)
    assert any("service/" in m for m in result.messages)


@patch(_PATCH_CLASSIFY)
def test_runner_subdir_split_disabled(mock_classify, tmp_path):
    # file_limiter_subdir_split=False — subdir detection is skipped entirely.
    mock_classify.return_value = ClassifiedEntities(
        entities=[],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[],
        set_3_groups=[],
        abort=True,  # force early abort so advise is not called
    )
    filepath = str(tmp_path / "service.py")
    source = "def foo():\n    pass\n"
    cfg = CrispenConfig(file_limiter_subdir_split=False)
    result = run_file_limiter(filepath, source, source, [(1, 2)], cfg)

    # abort comes from classifier, not from subdir detection
    assert result.abort is True
    assert "already exists" not in " ".join(result.messages)


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_subdir_split_non_test_success(mock_classify, mock_advise, mock_gen):
    # Whole-file diff on a non-test file → placements get subdir prefix,
    # original_source is unchanged, and __init__.py carries the split content.
    source = "def foo():\n    pass\ndef bar():\n    pass\n"
    entity1 = _make_entity("foo", 1, 2)
    entity2 = _make_entity("bar", 3, 4)
    # Two groups required so the n_groups > 1 subdir guard doesn't fire.
    mock_classify.return_value = ClassifiedEntities(
        entities=[entity1, entity2],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[["foo"], ["bar"]],
        set_3_groups=[],
        abort=False,
    )
    # LLM returns flat filenames (no subdir prefix yet).
    mock_advise.return_value = FileLimiterPlan(
        set3_migrate=[],
        placements=[
            GroupPlacement(group=["foo"], target_file="utils.py"),
            GroupPlacement(group=["bar"], target_file="helpers.py"),
        ],
        abort=False,
    )
    mock_gen.return_value = SplitResult(
        new_files={
            "service/utils.py": "def foo():\n    pass",
            "service/helpers.py": "def bar():\n    pass",
        },
        original_source="# init content\n",
        abort=False,
    )

    cfg = CrispenConfig(file_limiter_subdir_split=True)
    result = run_file_limiter("service.py", source, source, [(1, 4)], cfg)

    assert result.abort is False
    # service/__init__.py carries the post-split original source.
    assert "service/__init__.py" in result.new_files
    assert result.new_files["service/__init__.py"] == "# init content\n"
    # original_source is reset to the input (so service.py is not modified).
    assert result.original_source == source
    assert result.subdir_name == "service"
    # The moved-message includes the prefixed target file.
    assert any("service/utils.py" in m for m in result.messages)


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_subdir_split_test_file_keeps_original(
    mock_classify, mock_advise, mock_gen
):
    # Whole-file diff on a test file → placements get subdir prefix but
    # original_source (re-export stubs in test_service.py) is written back.
    source = "def test_foo():\n    pass\ndef test_bar():\n    pass\n"
    entity1 = _make_entity("test_foo", 1, 2)
    entity2 = _make_entity("test_bar", 3, 4)
    # Two groups required so the n_groups > 1 subdir guard doesn't fire.
    mock_classify.return_value = ClassifiedEntities(
        entities=[entity1, entity2],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[["test_foo"], ["test_bar"]],
        set_3_groups=[],
        abort=False,
    )
    mock_advise.return_value = FileLimiterPlan(
        set3_migrate=[],
        placements=[
            GroupPlacement(group=["test_foo"], target_file="helpers.py"),
            GroupPlacement(group=["test_bar"], target_file="extras.py"),
        ],
        abort=False,
    )
    mock_gen.return_value = SplitResult(
        new_files={
            "service/test_helpers.py": "def test_foo():\n    pass",
            "service/test_extras.py": "def test_bar():\n    pass",
        },
        original_source="# re-export stubs\n",
        abort=False,
    )

    cfg = CrispenConfig(file_limiter_subdir_split=True)
    result = run_file_limiter("tests/test_service.py", source, source, [(1, 4)], cfg)

    assert result.abort is False
    # No __init__.py injected for test files.
    assert "service/__init__.py" not in result.new_files
    # original_source has the re-export stubs (NOT reset to input).
    assert result.original_source == "# re-export stubs\n"
    assert result.subdir_name == "service"


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_subdir_split_strips_test_prefix_from_stem(
    mock_classify, mock_advise, mock_gen
):
    # test_big.py → subdir "big/" (strip "test_" prefix from stem).
    source = "def test_foo():\n    pass\ndef test_bar():\n    pass\n"
    entity1 = _make_entity("test_foo", 1, 2)
    entity2 = _make_entity("test_bar", 3, 4)
    # Two groups required so the n_groups > 1 subdir guard doesn't fire.
    mock_classify.return_value = ClassifiedEntities(
        entities=[entity1, entity2],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[["test_foo"], ["test_bar"]],
        set_3_groups=[],
        abort=False,
    )
    mock_advise.return_value = FileLimiterPlan(
        set3_migrate=[],
        placements=[
            GroupPlacement(group=["test_foo"], target_file="helpers.py"),
            GroupPlacement(group=["test_bar"], target_file="extras.py"),
        ],
        abort=False,
    )
    mock_gen.return_value = SplitResult(
        new_files={
            "big/test_helpers.py": "def test_foo():\n    pass",
            "big/test_extras.py": "def test_bar():\n    pass",
        },
        original_source="# stubs\n",
        abort=False,
    )

    cfg = CrispenConfig(file_limiter_subdir_split=True)
    result = run_file_limiter("tests/test_big.py", source, source, [(1, 4)], cfg)

    assert result.abort is False
    assert result.subdir_name == "big"
    # "helpers.py" → test_ prefix → "test_helpers.py" → "big/test_helpers.py".
    assert any("big/test_helpers.py" in m for m in result.messages)


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_subdir_split_has_main_uses_lib_suffix(
    mock_classify, mock_advise, mock_gen, tmp_path
):
    # Non-test file with __main__: subdir uses "_lib" suffix, original_source
    # is the split content (re-export stubs + __main__), and has_main=True.
    # No blank lines between entities so entity ranges don't pick up leading \n.
    source = (
        "def foo():\n    pass\n"
        "def bar():\n    pass\n"
        "if __name__ == '__main__':\n    foo()\n"
    )
    entity1 = _make_entity("foo", 1, 2)
    entity2 = _make_entity("bar", 3, 4)
    mock_classify.return_value = ClassifiedEntities(
        entities=[entity1, entity2],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[["foo"], ["bar"]],
        set_3_groups=[],
        abort=False,
    )
    mock_advise.return_value = FileLimiterPlan(
        set3_migrate=[],
        placements=[
            GroupPlacement(group=["foo"], target_file="utils.py"),
            GroupPlacement(group=["bar"], target_file="helpers.py"),
        ],
        abort=False,
    )
    mock_gen.return_value = SplitResult(
        new_files={
            "service_lib/utils.py": "def foo():\n    pass",
            "service_lib/helpers.py": "def bar():\n    pass",
        },
        original_source=(
            "from service_lib.utils import foo\n\n"
            "if __name__ == '__main__':\n    foo()\n"
        ),
        abort=False,
    )

    cfg = CrispenConfig(file_limiter_subdir_split=True)
    filepath = str(tmp_path / "service.py")
    result = run_file_limiter(filepath, source, source, [(1, 6)], cfg)

    assert result.abort is False
    assert result.has_main is True
    assert result.subdir_name == "service_lib"
    # original_source keeps the split content (re-exports + __main__), not reset.
    assert "__main__" in result.original_source
    # No __init__.py injected: original file stays as the entry point.
    assert "service_lib/__init__.py" not in result.new_files
    assert any("service_lib/utils.py" in m for m in result.messages)


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_subdir_split_has_main_fallback_suffix(
    mock_classify, mock_advise, mock_gen, tmp_path
):
    # When service_lib/ already exists, fall back to the next suffix (_helpers).
    source = (
        "def foo():\n    pass\n"
        "def bar():\n    pass\n"
        "if __name__ == '__main__':\n    foo()\n"
    )
    (tmp_path / "service_lib").mkdir()
    entity1 = _make_entity("foo", 1, 2)
    entity2 = _make_entity("bar", 3, 4)
    mock_classify.return_value = ClassifiedEntities(
        entities=[entity1, entity2],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[["foo"], ["bar"]],
        set_3_groups=[],
        abort=False,
    )
    mock_advise.return_value = FileLimiterPlan(
        set3_migrate=[],
        placements=[
            GroupPlacement(group=["foo"], target_file="utils.py"),
            GroupPlacement(group=["bar"], target_file="helpers.py"),
        ],
        abort=False,
    )
    mock_gen.return_value = SplitResult(
        new_files={
            "service_helpers/utils.py": "def foo():\n    pass",
            "service_helpers/helpers.py": "def bar():\n    pass",
        },
        original_source="# stubs\n",
        abort=False,
    )

    cfg = CrispenConfig(file_limiter_subdir_split=True)
    filepath = str(tmp_path / "service.py")
    result = run_file_limiter(filepath, source, source, [(1, 6)], cfg)

    assert result.abort is False
    assert result.subdir_name == "service_helpers"
    assert result.has_main is True


@patch(_PATCH_CLASSIFY)
def test_runner_subdir_split_has_main_all_suffixes_conflict_aborts(
    mock_classify, tmp_path
):
    # All _lib/_helpers/etc. directories already exist → abort with a clear message.
    source = "def foo():\n    pass\n\nif __name__ == '__main__':\n    foo()\n"
    for suffix in _MAIN_SUBDIR_SUFFIXES:
        (tmp_path / f"service{suffix}").mkdir()
    mock_classify.return_value = _make_classified()

    cfg = CrispenConfig(file_limiter_subdir_split=True)
    filepath = str(tmp_path / "service.py")
    result = run_file_limiter(filepath, source, source, [(1, 5)], cfg)

    assert result.abort is True
    assert result.new_files == {}
    assert any("__main__" in m for m in result.messages)
    assert any("conflict" in m for m in result.messages)


@patch(_PATCH_CLASSIFY)
def test_runner_init_py_skips_subdir_split(mock_classify, tmp_path):
    """__init__.py with a whole-file diff must not trigger subdir-split detection.

    A subdir split for __init__.py would create an ``__init__/`` subdirectory,
    which is nonsensical.  Instead it should fall through to the normal in-place
    split (siblings in the same package directory).
    """
    # Classify returns abort so the LLM path is skipped; we only care that
    # subdir_name is NOT set on the result.
    mock_classify.return_value = ClassifiedEntities(
        entities=[],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[],
        set_3_groups=[],
        abort=True,
    )
    filepath = str(tmp_path / "__init__.py")
    # Make the source long enough to be a "whole-file diff".
    source = "".join(f"def func_{i}():\n    pass\n\n" for i in range(10))
    cfg = CrispenConfig(file_limiter_subdir_split=True)
    result = run_file_limiter(
        filepath, source, source, [(1, len(source.splitlines()))], cfg
    )

    # Abort comes from the classifier — subdir conflict detection was bypassed.
    assert result.abort is True
    assert result.subdir_name is None
    assert "already exists" not in " ".join(result.messages)
