from __future__ import annotations
from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import SplitResult
from crispen.file_limiter.entity_parser import Entity, EntityKind
from crispen.file_limiter.runner import run_file_limiter


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
def test_runner_nonexistent_parent_dir_existing_dirs_empty(mock_classify, mock_advise):
    """When source dir doesn't exist, iterdir raises FileNotFoundError → empty set."""
    mock_classify.return_value = ClassifiedEntities(
        entities=[],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[],
        set_3_groups=[],
        abort=True,
    )
    # /nonexistent/parent doesn't exist; iterdir() will raise FileNotFoundError.
    result = run_file_limiter(
        "/nonexistent/parent/module.py",
        "",
        "def foo(): pass\n",
        [(1, 1)],
        _CONFIG,
    )
    # Abort from classifier, but the FileNotFoundError branch was hit.
    assert result.abort is True


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
    # test_/Test* names) — the target file must NOT get a test_ prefix so
    # pytest does not try to collect it.
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
def test_runner_init_not_renamed_by_test_prefix_logic(
    mock_classify, mock_advise, mock_gen
):
    # Defence-in-depth: __init__.py placements must not get the test_ prefix.
    source = "def test_foo():\n    pass\n\ndef _setup():\n    pass\n"
    e1 = _make_entity("test_foo", 1, 2)
    e2 = _make_entity("_setup", 4, 5)
    mock_classify.return_value = _make_classified(entities=[e1, e2])
    mock_advise.return_value = FileLimiterPlan(
        set3_migrate=[],
        placements=[
            GroupPlacement(group=["test_foo"], target_file="cases.py"),
            GroupPlacement(group=["_setup"], target_file="__init__.py"),
        ],
        abort=False,
    )
    mock_gen.return_value = SplitResult(
        new_files={
            "test_cases.py": "def test_foo():\n    pass",
            "__init__.py": "def _setup():\n    pass",
        },
        original_source="# original\n",
        abort=False,
    )

    # tests/runner/ has no __init__.py so it won't appear in existing_files.
    result = run_file_limiter("tests/runner/test_big.py", "", source, [], _CONFIG)

    assert result.abort is False
    # cases.py → test_cases.py (has test_foo in group)
    assert any("test_cases.py" in m for m in result.messages)
    # __init__.py untouched
    assert any("__init__.py" in m for m in result.messages)
    assert not any("test___init__.py" in m for m in result.messages)


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
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_strips_tests_suffix_and_adds_prefix(
    mock_classify, mock_advise, mock_gen
):
    # LLM returns a filename ending with _tests.py — strip the suffix and add
    # the test_ prefix so pytest discovers the file.
    source = "class TestFoo:\n    def test_bar(self):\n        pass\n"
    entity = _make_entity("TestFoo", 1, 3)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["TestFoo"], "foo_tests.py")
    mock_gen.return_value = SplitResult(
        new_files={
            "test_foo.py": "class TestFoo:\n    def test_bar(self):\n        pass"
        },
        original_source="# original\n",
        abort=False,
    )

    result = run_file_limiter("tests/test_big.py", "", source, [], _CONFIG)

    assert result.abort is False
    assert any("test_foo.py" in m for m in result.messages)
    assert not any("foo_tests.py" in m for m in result.messages)


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_adds_prefix_for_test_class_group(mock_classify, mock_advise, mock_gen):
    # Group contains a Test-prefixed class (not test_ function) — must still
    # get the test_ file prefix.
    source = "class TestFoo:\n    def test_bar(self):\n        pass\n"
    entity = _make_entity("TestFoo", 1, 3)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["TestFoo"], "foo_cases.py")
    mock_gen.return_value = SplitResult(
        new_files={
            "test_foo_cases.py": "class TestFoo:\n    def test_bar(self):\n        pass"
        },
        original_source="# original\n",
        abort=False,
    )

    result = run_file_limiter("tests/test_big.py", "", source, [], _CONFIG)

    assert result.abort is False
    assert any("test_foo_cases.py" in m for m in result.messages)
    assert not any(
        "foo_cases.py" in m and "test_foo_cases.py" not in m for m in result.messages
    )


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


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_entity_to_target_populated_on_success(mock_classify, mock_advise, mock_gen):
    """On a successful run, entity_to_target maps entity names to target files."""
    source = "def foo():\n    pass\ndef bar():\n    pass\n"
    entity_foo = _make_entity("foo", 1, 2)
    entity_bar = _make_entity("bar", 3, 4)
    mock_classify.return_value = _make_classified(entities=[entity_foo, entity_bar])
    # Plan: foo → utils.py, bar → helpers.py

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
            "utils.py": "def foo():\n    pass",
            "helpers.py": "def bar():\n    pass",
        },
        original_source="# original updated\n",
        abort=False,
    )

    result = run_file_limiter("big.py", "", source, [], _CONFIG)

    assert result.abort is False
    assert result.entity_to_target == {
        "foo": "utils.py",
        "bar": "helpers.py",
    }


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_entity_to_target_empty_on_abort(mock_classify, mock_advise, mock_gen):
    """Abort result has empty entity_to_target."""
    mock_classify.return_value = ClassifiedEntities(
        entities=[],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[],
        set_3_groups=[],
        abort=True,
    )

    result = run_file_limiter("big.py", "", "x = 1\n", [], _CONFIG_NO_RETRY)

    assert result.abort is True
    assert result.entity_to_target == {}
