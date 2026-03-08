from __future__ import annotations
from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import SplitResult
from crispen.file_limiter.entity_parser import Entity, EntityKind
from crispen.file_limiter.runner import _MAIN_SUBDIR_SUFFIXES, run_file_limiter
from .test_preservation import _make_entity
from ..test_runner import (
    _CONFIG,
    _CONFIG_NO_RETRY,
    _PATCH_ADVISE,
    _PATCH_CLASSIFY,
    _PATCH_GEN,
    _PATCH_RESOLVE,
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


@patch(_PATCH_CLASSIFY)
def test_runner_dashed_parent_dir_skips(mock_classify):
    # A filepath whose parent contains a dash must be skipped immediately,
    # before classify_entities is ever called.
    result = run_file_limiter(
        "tests/cross-engine/test_lever.py", "", "x = 1\n", [(1, 1)], _CONFIG
    )
    assert result.abort is True
    assert "cross-engine" in result.messages[0]
    assert "dash" in result.messages[0]
    mock_classify.assert_not_called()


@patch(_PATCH_CLASSIFY)
def test_runner_dashed_parent_dir_deep_skips(mock_classify):
    # Dash anywhere in the ancestor chain (not just the immediate parent).
    result = run_file_limiter(
        "src/my-pkg/sub/module.py", "", "x = 1\n", [(1, 1)], _CONFIG
    )
    assert result.abort is True
    assert "my-pkg" in result.messages[0]
    mock_classify.assert_not_called()


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
def test_runner_passes_pytest_conftest_to_generate(
    mock_classify, mock_advise, mock_gen
):
    # Verify config.file_limiter_pytest_conftest is forwarded to generate_file_splits.
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["foo"], "utils.py")
    mock_gen.return_value = _good_split()

    config_false = CrispenConfig(file_limiter_pytest_conftest=False)
    run_file_limiter("big.py", "", source, [], config_false)

    _, call_kwargs = mock_gen.call_args
    assert call_kwargs.get("pytest_conftest") is False


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
