from __future__ import annotations
from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.runner import _detect_naming_conflicts, run_file_limiter
from .test_runner_core import (
    _CONFIG_NO_RETRY,
    _PATCH_ADVISE,
    _PATCH_CLASSIFY,
    _PATCH_GEN,
    _PATCH_RESOLVE,
    _good_split,
    _make_classified,
    _make_entity,
    _plan_with,
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


def test_detect_conflicts_flat_target_in_existing_files():
    # Flat target whose filename is in existing_files (e.g. conftest.py) → conflict.
    placements = [GroupPlacement(group=["fix"], target_file="conftest.py")]
    conflicts = _detect_naming_conflicts(
        placements, frozenset({"conftest.py"}), frozenset()
    )
    assert len(conflicts) == 1
    assert "conftest.py" in conflicts[0]


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
