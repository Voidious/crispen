from __future__ import annotations
from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import SplitResult
from crispen.file_limiter.runner import (
    _detect_naming_conflicts,
    _is_whole_file_diff,
    run_file_limiter,
)
from .test_verify import _make_entity
from ..test_runner import (
    _CONFIG_NO_RETRY,
    _PATCH_ADVISE,
    _PATCH_CLASSIFY,
    _PATCH_GEN,
    _PATCH_RESOLVE,
)


def _good_split(entity_name: str = "foo", target: str = "utils.py") -> SplitResult:
    return SplitResult(
        new_files={target: f"def {entity_name}():\n    pass"},
        original_source="# original updated\n",
        abort=False,
    )


def _plan_with(group: list, target: str) -> FileLimiterPlan:
    return FileLimiterPlan(
        set3_migrate=[],
        placements=[GroupPlacement(group=group, target_file=target)],
        abort=False,
    )


def _make_classified(entities=None) -> ClassifiedEntities:
    return ClassifiedEntities(
        entities=entities or [],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[],
        set_3_groups=[],
        abort=False,
    )


def test_detect_conflicts_no_conflicts():
    placements = [
        GroupPlacement(group=["foo"], target_file="utils.py"),
        GroupPlacement(group=["bar"], target_file="helpers.py"),
    ]
    assert _detect_naming_conflicts(placements, frozenset(), frozenset()) == []


def test_detect_conflicts_plan_vs_plan():
    # Plan contains both 'utils.py' and 'utils/io.py' → conflict on stem 'utils'.
    placements = [
        GroupPlacement(group=["foo"], target_file="utils.py"),
        GroupPlacement(group=["bar"], target_file="utils/io.py"),
    ]
    conflicts = _detect_naming_conflicts(placements, frozenset(), frozenset())
    assert len(conflicts) == 1
    assert "'utils.py'" in conflicts[0]
    assert "'utils/'" in conflicts[0]


def test_detect_conflicts_plan_file_vs_existing_dir():
    # Plan proposes 'models.py' but 'models' directory already exists on disk.
    placements = [GroupPlacement(group=["foo"], target_file="models.py")]
    conflicts = _detect_naming_conflicts(placements, frozenset(), frozenset({"models"}))
    assert len(conflicts) == 1
    assert "'models.py'" in conflicts[0]
    assert "'models/'" in conflicts[0]


def test_detect_conflicts_plan_dir_vs_existing_file():
    # Plan proposes 'helpers/io.py' but 'helpers.py' already exists on disk.
    placements = [GroupPlacement(group=["bar"], target_file="helpers/io.py")]
    conflicts = _detect_naming_conflicts(
        placements, frozenset({"helpers.py"}), frozenset()
    )
    assert len(conflicts) == 1
    assert "'helpers/'" in conflicts[0]
    assert "'helpers.py'" in conflicts[0]


def test_detect_conflicts_no_filesystem_conflict():
    # Proposed 'utils.py'; existing dir named 'other' — no overlap.
    placements = [GroupPlacement(group=["foo"], target_file="utils.py")]
    assert _detect_naming_conflicts(placements, frozenset(), frozenset({"other"})) == []


def test_detect_conflicts_multiple_conflicts():
    # Three separate conflicts in one plan.
    placements = [
        GroupPlacement(group=["a"], target_file="alpha.py"),  # vs alpha/ dir on disk
        GroupPlacement(group=["b"], target_file="beta/x.py"),  # vs beta.py on disk
        GroupPlacement(group=["c"], target_file="gamma.py"),  # vs gamma/ in plan
        GroupPlacement(group=["d"], target_file="gamma/y.py"),  # vs gamma.py in plan
    ]
    conflicts = _detect_naming_conflicts(
        placements, frozenset({"beta.py"}), frozenset({"alpha"})
    )
    assert len(conflicts) == 3  # alpha (disk dir), beta (disk file), gamma (plan)


def test_detect_conflicts_subdir_only_no_conflict():
    # All targets are in different subdirectories — no stem overlap.
    placements = [
        GroupPlacement(group=["a"], target_file="pkg/models.py"),
        GroupPlacement(group=["b"], target_file="pkg/helpers.py"),
    ]
    # Both land in 'pkg/' — that's fine; only 'pkg.py' vs 'pkg/' would conflict.
    assert _detect_naming_conflicts(placements, frozenset(), frozenset()) == []


@patch(_PATCH_GEN)
@patch(_PATCH_RESOLVE)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_naming_conflict_resolve_succeeds(
    mock_classify, mock_advise, mock_resolve, mock_gen
):
    # Conflict → resolve returns updated placements → generate called once, advise once.
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    conflicting_plan = FileLimiterPlan(
        set3_migrate=[],
        placements=[
            GroupPlacement(group=["foo"], target_file="utils.py"),
            GroupPlacement(group=["bar"], target_file="utils/helpers.py"),  # conflict!
        ],
        abort=False,
    )
    resolved_placements = [GroupPlacement(group=["foo"], target_file="models.py")]
    mock_advise.return_value = conflicting_plan
    mock_resolve.return_value = resolved_placements
    mock_gen.return_value = _good_split(entity_name="foo", target="models.py")

    result = run_file_limiter(
        "big.py", "", "def foo():\n    pass\n", [], _CONFIG_NO_RETRY
    )

    assert result.abort is False
    assert mock_advise.call_count == 1  # no outer retry needed
    assert mock_resolve.call_count == 1
    assert mock_gen.call_count == 1
    # No SKIP message — resolve handled the conflict without retrying advise.
    assert not any("naming conflicts" in m for m in result.messages)
    assert any("FileLimiter: moved" in m for m in result.messages)


@patch(_PATCH_GEN)
@patch(_PATCH_RESOLVE)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_naming_conflict_resolve_fails_then_outer_retry(
    mock_classify, mock_advise, mock_resolve, mock_gen
):
    # resolve returns None → outer retry → second advise succeeds.
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    conflicting_plan = FileLimiterPlan(
        set3_migrate=[],
        placements=[
            GroupPlacement(group=["foo"], target_file="pkg.py"),
            GroupPlacement(group=["bar"], target_file="pkg/mod.py"),  # conflict!
        ],
        abort=False,
    )
    mock_advise.side_effect = [conflicting_plan, _plan_with(["foo"], "models.py")]
    mock_resolve.return_value = None  # targeted rename fails
    mock_gen.return_value = _good_split(entity_name="foo", target="models.py")
    cfg = CrispenConfig(file_limiter_retries=1)

    result = run_file_limiter("big.py", "", "def foo():\n    pass\n", [], cfg)

    assert result.abort is False
    assert mock_advise.call_count == 2
    assert mock_resolve.call_count == 1
    assert any("naming conflicts" in m for m in result.messages)
    assert any("FileLimiter: moved" in m for m in result.messages)
    # Conflict description was forwarded as feedback for the second advise call.
    prev_pf = mock_advise.call_args_list[1].kwargs["prev_placement_failure"]
    assert "naming conflicts" in prev_pf


@patch(_PATCH_RESOLVE)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_naming_conflict_exhausts_all(mock_classify, mock_advise, mock_resolve):
    # resolve always fails, all retries exhausted → abort=True, 2 SKIP messages.
    mock_classify.return_value = _make_classified()
    conflicting_plan = FileLimiterPlan(
        set3_migrate=[],
        placements=[
            GroupPlacement(group=["foo"], target_file="pkg.py"),
            GroupPlacement(group=["bar"], target_file="pkg/mod.py"),  # conflict!
        ],
        abort=False,
    )
    mock_advise.return_value = conflicting_plan
    mock_resolve.return_value = None  # always fails
    cfg = CrispenConfig(file_limiter_retries=1)

    result = run_file_limiter("big.py", "", "def foo():\n    pass\n", [], cfg)

    assert result.abort is True
    assert mock_advise.call_count == 2
    assert mock_resolve.call_count == 2
    assert sum(1 for m in result.messages if "naming conflicts" in m) == 2


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
