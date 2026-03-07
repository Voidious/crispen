from __future__ import annotations
from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import SplitResult
from crispen.file_limiter.entity_parser import Entity, EntityKind
from crispen.file_limiter.runner import _is_whole_file_diff, run_file_limiter


_CONFIG = CrispenConfig()
# Zero-retry config for tests that exercise a single-attempt failure path.
_CONFIG_NO_RETRY = CrispenConfig(file_limiter_retries=0)
_PATCH_CLASSIFY = "crispen.file_limiter.runner.classify_entities"
_PATCH_ADVISE = "crispen.file_limiter.runner.advise_file_limiter"
_PATCH_GEN = "crispen.file_limiter.runner.generate_file_splits"
_PATCH_RESOLVE = "crispen.file_limiter.runner.resolve_naming_conflicts"


def _make_entity(name: str, start: int, end: int) -> Entity:
    return Entity(EntityKind.FUNCTION, name, start, end, [name])


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


def _abort_plan() -> FileLimiterPlan:
    return FileLimiterPlan(set3_migrate=[], placements=[], abort=True)


def _empty_plan() -> FileLimiterPlan:
    return FileLimiterPlan(set3_migrate=[], placements=[], abort=False)


def _plan_with(group: list, target: str) -> FileLimiterPlan:
    return FileLimiterPlan(
        set3_migrate=[],
        placements=[GroupPlacement(group=group, target_file=target)],
        abort=False,
    )


def _classified_with_groups(entities=None) -> ClassifiedEntities:
    """Classified result with non-empty set_3_groups (triggers LLM advise)."""
    ents = entities or [_make_entity("foo", 1, 2)]
    return ClassifiedEntities(
        entities=ents,
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[],
        set_3_groups=[[e.name for e in ents]],
        abort=False,
    )


def _good_split(entity_name: str = "foo", target: str = "utils.py") -> SplitResult:
    return SplitResult(
        new_files={target: f"def {entity_name}():\n    pass"},
        original_source="# original updated\n",
        abort=False,
    )


@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_classifier_abort(mock_classify, mock_advise):
    # classified.abort=True → early return before LLM; advise never called.
    mock_classify.return_value = ClassifiedEntities(
        entities=[_make_entity("a", 1, 2), _make_entity("b", 3, 4)],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[],
        set_3_groups=[],
        abort=True,
        abort_reason="",
    )

    result = run_file_limiter("big.py", "", "def a(): b()\ndef b(): a()\n", [], _CONFIG)

    assert result.abort is True
    mock_advise.assert_not_called()
    assert any("cannot be split" in m for m in result.messages)


@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_classifier_abort_with_reason(mock_classify, mock_advise):
    mock_classify.return_value = ClassifiedEntities(
        entities=[_make_entity("a", 1, 2), _make_entity("b", 3, 4)],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[],
        set_3_groups=[],
        abort=True,
        abort_reason="all 2 top-level entities form one dependency cycle",
    )

    result = run_file_limiter("big.py", "", "def a(): b()\ndef b(): a()\n", [], _CONFIG)

    assert result.abort is True
    mock_advise.assert_not_called()
    assert any("dependency cycle" in m for m in result.messages)


@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_plan_abort(mock_classify, mock_advise):
    mock_classify.return_value = _make_classified()
    mock_advise.return_value = _abort_plan()

    result = run_file_limiter(
        "big.py", "", "def foo():\n    pass\n", [], _CONFIG_NO_RETRY
    )

    assert result.abort is True
    assert result.new_files == {}
    assert any("cannot be split" in m for m in result.messages)


@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_plan_abort_with_reason(mock_classify, mock_advise):
    mock_classify.return_value = _make_classified()
    mock_advise.return_value = FileLimiterPlan(
        set3_migrate=[], placements=[], abort=True, abort_reason="all 3 entities cycle"
    )

    result = run_file_limiter(
        "big.py", "", "def foo():\n    pass\n", [], _CONFIG_NO_RETRY
    )

    assert result.abort is True
    assert any("all 3 entities cycle" in m for m in result.messages)


@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_plan_abort_retries_and_fails(mock_classify, mock_advise):
    # retries=1: both attempts produce plan.abort (set-3 failure) → 2 SKIP messages.
    mock_classify.return_value = _make_classified()
    mock_advise.return_value = FileLimiterPlan(
        set3_migrate=[],
        placements=[],
        abort=True,
        abort_reason="LLM failed to plan set-3 groups",
    )
    cfg = CrispenConfig(file_limiter_retries=1)

    result = run_file_limiter("big.py", "", "def foo():\n    pass\n", [], cfg)

    assert result.abort is True
    assert mock_advise.call_count == 2
    assert sum(1 for m in result.messages if "cannot be split" in m) == 2


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_plan_abort_retries_and_succeeds(mock_classify, mock_advise, mock_gen):
    # retries=1: first plan.abort (placement failure), second succeeds.
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.side_effect = [
        FileLimiterPlan(
            set3_migrate=[],
            placements=[],
            abort=True,
            abort_reason="LLM failed to assign file placements",
        ),
        _plan_with(["foo"], "utils.py"),
    ]
    mock_gen.return_value = _good_split()
    cfg = CrispenConfig(file_limiter_retries=1)

    result = run_file_limiter("big.py", "", "def foo():\n    pass\n", [], cfg)

    assert result.abort is False
    assert mock_advise.call_count == 2
    # Failed attempt message is preserved alongside the success message.
    assert any("cannot be split" in m for m in result.messages)
    assert any("FileLimiter: moved" in m for m in result.messages)
    # Feedback was forwarded on the second call.
    assert mock_advise.call_args_list[1].kwargs["prev_placement_failure"] != ""


@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_no_placements(mock_classify, mock_advise):
    mock_classify.return_value = _make_classified()
    mock_advise.return_value = _empty_plan()

    source = "def foo():\n    pass\n"
    result = run_file_limiter("big.py", "", source, [], _CONFIG)

    assert result.abort is False
    assert result.new_files == {}
    assert result.original_source == source
    assert result.messages == []


@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_no_placements_with_groups(mock_classify, mock_advise):
    # When set_3_groups is non-empty but the LLM selects nothing to migrate,
    # runner should emit a SKIP message so the user knows the file was examined.
    mock_classify.return_value = _classified_with_groups()
    mock_advise.return_value = _empty_plan()

    source = "def foo():\n    pass\n"
    result = run_file_limiter("big.py", "", source, [], _CONFIG_NO_RETRY)

    assert result.abort is False
    assert result.new_files == {}
    assert any("no entities selected for migration" in m for m in result.messages)


@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_no_migration_retries_and_fails(mock_classify, mock_advise):
    # retries=1: both attempts → no entities selected → 2 SKIP msgs, abort=False.
    mock_classify.return_value = _classified_with_groups()
    mock_advise.return_value = _empty_plan()
    cfg = CrispenConfig(file_limiter_retries=1)

    result = run_file_limiter("big.py", "", "def foo():\n    pass\n", [], cfg)

    assert result.abort is False
    assert mock_advise.call_count == 2
    assert sum(1 for m in result.messages if "no entities selected" in m) == 2


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_no_migration_retries_and_succeeds(mock_classify, mock_advise, mock_gen):
    # retries=1: first attempt → no migration; second → placements and success.
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _classified_with_groups(entities=[entity])
    mock_advise.side_effect = [_empty_plan(), _plan_with(["foo"], "utils.py")]
    mock_gen.return_value = _good_split()
    cfg = CrispenConfig(file_limiter_retries=1)

    result = run_file_limiter("big.py", "", "def foo():\n    pass\n", [], cfg)

    assert result.abort is False
    assert mock_advise.call_count == 2
    # Failed attempt message is preserved alongside the success message.
    assert any("no entities selected" in m for m in result.messages)
    assert any("FileLimiter: moved" in m for m in result.messages)
    # Feedback about all-stay was forwarded on the second call.
    assert mock_advise.call_args_list[1].kwargs["prev_set3_failure"] != ""


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
    mock_classify.return_value = _make_classified()
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
    mock_classify.return_value = _make_classified(entities=[entity1, entity2])
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
def test_runner_single_group_one_file_subdir_allowed(
    mock_classify, mock_advise, mock_gen
):
    # Subdir split with only 1 group: 1 placement → 1 file is always valid.
    entity = _make_entity("foo", 1, 1)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["foo"], "utils.py")
    mock_gen.return_value = SplitResult(
        new_files={"big/utils.py": "x = 1"},
        original_source=_SUBDIR_SRC,
        abort=False,
    )
    cfg = CrispenConfig(file_limiter_retries=0)

    result = run_file_limiter("big.py", "", _SUBDIR_SRC, _SUBDIR_RANGES, cfg)

    assert result.abort is False
    assert not any("single file" in m for m in result.messages)
    mock_gen.assert_called_once()


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_verification_fails(mock_classify, mock_advise, mock_gen):
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["foo"], "utils.py")
    # Return a split where foo's source is NOT present anywhere.
    mock_gen.return_value = SplitResult(
        new_files={"utils.py": "# empty placeholder"},
        original_source="# empty original",
        abort=False,
    )

    result = run_file_limiter("big.py", "", source, [], _CONFIG)

    assert result.abort is True
    assert result.original_source == source
    assert any("verification failed" in m for m in result.messages)


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_success(mock_classify, mock_advise, mock_gen):
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["foo"], "utils.py")
    mock_gen.return_value = SplitResult(
        new_files={"utils.py": "def foo():\n    pass"},
        original_source="# original updated\n",
        abort=False,
    )

    result = run_file_limiter("big.py", "", source, [], _CONFIG)

    assert result.abort is False
    assert "utils.py" in result.new_files
    assert result.original_source == "# original updated\n"
    assert any("FileLimiter: moved" in m for m in result.messages)
    assert any("foo" in m for m in result.messages)
    assert any("utils.py" in m for m in result.messages)


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_split_aborts_on_cycle(mock_classify, mock_advise, mock_gen):
    # generate_file_splits detects a cycle and returns abort=True with no
    # new_files.  run_file_limiter must emit a SKIP message (not bogus "moved"
    # messages) and return abort=True so the engine skips the file.
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["foo"], "utils.py")
    mock_gen.return_value = SplitResult(
        new_files={},
        original_source=source,
        abort=True,
    )

    result = run_file_limiter("big.py", "", source, [], _CONFIG_NO_RETRY)

    assert result.abort is True
    assert result.new_files == {}
    assert result.original_source == source
    # Must not claim to have moved anything.
    assert not any("FileLimiter: moved" in m for m in result.messages)
    assert any("cannot be split" in m for m in result.messages)


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_split_aborts_with_reason(mock_classify, mock_advise, mock_gen):
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["foo"], "utils.py")
    mock_gen.return_value = SplitResult(
        new_files={},
        original_source=source,
        abort=True,
        abort_reason="proposed split would create circular file imports",
    )

    result = run_file_limiter("big.py", "", source, [], _CONFIG_NO_RETRY)

    assert result.abort is True
    assert any("circular file imports" in m for m in result.messages)


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_split_abort_retries_and_fails(mock_classify, mock_advise, mock_gen):
    # retries=1: both attempts produce split.abort → 2 SKIP messages, abort=True.
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["foo"], "utils.py")
    mock_gen.return_value = SplitResult(
        new_files={},
        original_source=source,
        abort=True,
        abort_reason="proposed split would create circular file imports",
    )
    cfg = CrispenConfig(file_limiter_retries=1)

    result = run_file_limiter("big.py", "", source, [], cfg)

    assert result.abort is True
    assert mock_advise.call_count == 2
    assert mock_gen.call_count == 2
    assert sum(1 for m in result.messages if "cannot be split" in m) == 2
    # Circular-import failure bumps target_files_bonus for the second call.
    bonus_on_retry = mock_advise.call_args_list[1].kwargs["target_files_bonus"]
    assert bonus_on_retry >= 1


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_split_abort_retries_and_succeeds(mock_classify, mock_advise, mock_gen):
    # retries=1: first split.abort, second succeeds → only "moved" message, no SKIP.
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["foo"], "utils.py")
    mock_gen.side_effect = [
        SplitResult(
            new_files={},
            original_source=source,
            abort=True,
            abort_reason="circular imports",
        ),
        _good_split(),
    ]
    cfg = CrispenConfig(file_limiter_retries=1)

    result = run_file_limiter("big.py", "", source, [], cfg)

    assert result.abort is False
    assert mock_advise.call_count == 2
    assert mock_gen.call_count == 2
    # Failed attempt message is preserved alongside the success message.
    assert any("cannot be split" in m for m in result.messages)
    assert any("FileLimiter: moved" in m for m in result.messages)
    # Circular-import feedback was forwarded on the second call.
    prev_pf = mock_advise.call_args_list[1].kwargs["prev_placement_failure"]
    assert "circular" in prev_pf
    # target_files_bonus was also increased for the retry.
    bonus_on_retry = mock_advise.call_args_list[1].kwargs["target_files_bonus"]
    assert bonus_on_retry >= 1


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_split_abort_bonus_forwarded(mock_classify, mock_advise, mock_gen):
    # On circular-import failure, bonus = max(1, round(0.25 * original_target)).
    # With original_target_files=8, per_step = max(1, round(2.0)) = 2.
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = FileLimiterPlan(
        set3_migrate=[],
        placements=[GroupPlacement(group=["foo"], target_file="utils.py")],
        abort=False,
        original_target_files=8,
    )
    mock_gen.side_effect = [
        SplitResult(
            new_files={},
            original_source=source,
            abort=True,
            abort_reason="circular imports",
        ),
        _good_split(),
    ]
    cfg = CrispenConfig(file_limiter_retries=1)

    run_file_limiter("big.py", "", source, [], cfg)

    bonus_on_retry = mock_advise.call_args_list[1].kwargs["target_files_bonus"]
    assert bonus_on_retry == 2  # max(1, round(0.25 * 8)) = 2


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_split_abort_bonus_accumulates(mock_classify, mock_advise, mock_gen):
    # Two consecutive circular-import retries accumulate the bonus additively.
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = FileLimiterPlan(
        set3_migrate=[],
        placements=[GroupPlacement(group=["foo"], target_file="utils.py")],
        abort=False,
        original_target_files=8,
    )
    mock_gen.side_effect = [
        SplitResult(
            new_files={},
            original_source=source,
            abort=True,
            abort_reason="circular imports",
        ),
        SplitResult(
            new_files={},
            original_source=source,
            abort=True,
            abort_reason="circular imports",
        ),
        _good_split(),
    ]
    cfg = CrispenConfig(file_limiter_retries=2)

    run_file_limiter("big.py", "", source, [], cfg)

    bonuses = [
        mock_advise.call_args_list[i].kwargs["target_files_bonus"] for i in range(3)
    ]
    # per_step = max(1, round(0.25 * 8)) = 2; accumulates each circular retry.
    assert bonuses == [0, 2, 4]


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_split_abort_bonus_retained_on_other_failure(
    mock_classify, mock_advise, mock_gen
):
    # First failure is circular import → bonus bumped.
    # Second failure is plan.abort (different reason) → bonus retained, not reset.
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.side_effect = [
        # Attempt 1: plan OK, original_target_files=8.
        FileLimiterPlan(
            set3_migrate=[],
            placements=[GroupPlacement(group=["foo"], target_file="utils.py")],
            abort=False,
            original_target_files=8,
        ),
        # Attempt 2: plan aborts (LLM error) — no split step, no bonus change.
        FileLimiterPlan(set3_migrate=[], placements=[], abort=True, abort_reason="x"),
        # Attempt 3: plan OK, success.
        FileLimiterPlan(
            set3_migrate=[],
            placements=[GroupPlacement(group=["foo"], target_file="utils.py")],
            abort=False,
            original_target_files=8,
        ),
    ]
    mock_gen.side_effect = [
        SplitResult(
            new_files={},
            original_source=source,
            abort=True,
            abort_reason="circular imports",
        ),
        _good_split(),
    ]
    cfg = CrispenConfig(file_limiter_retries=2)

    run_file_limiter("big.py", "", source, [], cfg)

    bonuses = [
        mock_advise.call_args_list[i].kwargs["target_files_bonus"] for i in range(3)
    ]
    # Attempt 1: no bonus. Attempt 2: circular-import bumped by 2. Attempt 3: plan.abort
    # did not change bonus, so still 2.
    assert bonuses == [0, 2, 2]


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_adds_test_prefix_to_new_files(mock_classify, mock_advise, mock_gen):
    # When the source file is test_*.py, target files in the same directory
    # must also have the test_ prefix so pytest can discover the moved tests.
    source = "def test_foo():\n    pass\n"
    entity = _make_entity("test_foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["test_foo"], "helpers.py")
    mock_gen.return_value = SplitResult(
        new_files={"test_helpers.py": "def test_foo():\n    pass"},
        original_source="# original\n",
        abort=False,
    )

    result = run_file_limiter("tests/test_big.py", "", source, [], _CONFIG)

    assert result.abort is False
    # The placement target passed to generate_file_splits must have been
    # normalised — verify via the success message.
    assert any("test_helpers.py" in m for m in result.messages)
    assert not any(
        "helpers.py" in m and "test_helpers.py" not in m for m in result.messages
    )


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_test_prefix_already_present(mock_classify, mock_advise, mock_gen):
    # Target file already starts with test_ → name is left unchanged.
    source = "def test_foo():\n    pass\n"
    entity = _make_entity("test_foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["test_foo"], "test_helpers.py")
    mock_gen.return_value = SplitResult(
        new_files={"test_helpers.py": "def test_foo():\n    pass"},
        original_source="# original\n",
        abort=False,
    )

    result = run_file_limiter("tests/test_big.py", "", source, [], _CONFIG)

    assert result.abort is False
    assert any("test_helpers.py" in m for m in result.messages)


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_no_test_prefix_for_helper_only_group(
    mock_classify, mock_advise, mock_gen
):
    # Source is test_*.py but the group contains only helper functions (no
    # test_ prefix) — the target file name should NOT get a test_ prefix.
    source = "def _helper():\n    pass\n"
    entity = _make_entity("_helper", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["_helper"], "helpers.py")
    mock_gen.return_value = SplitResult(
        new_files={"helpers.py": "def _helper():\n    pass"},
        original_source="# original\n",
        abort=False,
    )

    result = run_file_limiter("tests/test_big.py", "", source, [], _CONFIG)

    assert result.abort is False
    assert any("helpers.py" in m for m in result.messages)
    assert not any("test_helpers.py" in m for m in result.messages)


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_no_test_prefix_for_non_test_file(mock_classify, mock_advise, mock_gen):
    # Source file is NOT a test module — target file names are left as-is.
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["foo"], "helpers.py")
    mock_gen.return_value = SplitResult(
        new_files={"helpers.py": "def foo():\n    pass"},
        original_source="# original\n",
        abort=False,
    )

    result = run_file_limiter("big.py", "", source, [], _CONFIG)

    assert result.abort is False
    assert any("helpers.py" in m for m in result.messages)


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


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_verbose_prints_to_stderr(mock_classify, mock_advise, mock_gen, capsys):
    """verbose=True prints analysis/verification messages to stderr."""
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["foo"], "utils.py")
    mock_gen.return_value = SplitResult(
        new_files={"utils.py": "def foo():\n    pass"},
        original_source="# original updated\n",
        abort=False,
    )

    result = run_file_limiter("big.py", "", source, [], _CONFIG, verbose=True)

    assert result.abort is False
    err = capsys.readouterr().err
    assert "FileLimiter" in err
    assert "big.py" in err


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_success_with_class_entity(mock_classify, mock_advise, mock_gen):
    """Verification loop increments verified_classes for CLASS entities."""
    source = "class Foo:\n    pass\n"
    entity = Entity(EntityKind.CLASS, "Foo", 1, 2, ["Foo"])
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["Foo"], "models.py")
    mock_gen.return_value = SplitResult(
        new_files={"models.py": "class Foo:\n    pass"},
        original_source="# original\n",
        abort=False,
    )

    result = run_file_limiter("big.py", "", source, [], _CONFIG)

    assert result.abort is False
    assert result.verified_classes == 1
    assert result.verified_functions == 0
    assert result.verified_lines == 2


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_success_with_top_level_entity(mock_classify, mock_advise, mock_gen):
    """TOP_LEVEL entities are skipped in the verification count loop."""
    source = "import os\ndef foo():\n    pass\n"
    import_entity = Entity(EntityKind.TOP_LEVEL, "_block_0", 1, 1, ["os"])
    func_entity = _make_entity("foo", 2, 3)
    mock_classify.return_value = _make_classified(entities=[import_entity, func_entity])
    mock_advise.return_value = _plan_with(["foo"], "utils.py")
    mock_gen.return_value = SplitResult(
        new_files={"utils.py": "def foo():\n    pass"},
        original_source="import os\n",
        abort=False,
    )

    result = run_file_limiter("big.py", "", source, [], _CONFIG)

    assert result.abort is False
    # Only the function counts; TOP_LEVEL is skipped.
    assert result.verified_functions == 1
    assert result.verified_classes == 0


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_success_with_empty_entity_source(mock_classify, mock_advise, mock_gen):
    """Entities whose source is blank after rstrip are skipped in the count.

    Also covers verification of an entity that stays in the original file
    (stays_entity is verified and counted alongside migrated entities).
    """
    source = "def foo():\n    pass\n\ndef bar():\n    pass\n"
    # blank_entity has empty source → skipped. foo migrated; bar stays in original.
    blank_entity = _make_entity("_block_1", 3, 3)
    func_entity = _make_entity("foo", 1, 2)
    stays_entity = _make_entity("bar", 4, 5)
    mock_classify.return_value = _make_classified(
        entities=[func_entity, blank_entity, stays_entity]
    )
    mock_advise.return_value = _plan_with(["foo"], "utils.py")
    mock_gen.return_value = SplitResult(
        new_files={"utils.py": "def foo():\n    pass"},
        original_source="def bar():\n    pass\n",
        abort=False,
    )

    result = run_file_limiter("big.py", "", source, [], _CONFIG)

    assert result.abort is False
    # blank_entity: empty source → skipped. bar: stays in original → not counted.
    # Only foo (migrated) counts.
    assert result.verified_functions == 1
    assert result.verified_lines == 2
