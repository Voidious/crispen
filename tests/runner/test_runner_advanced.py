from __future__ import annotations
from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import SplitResult
from crispen.file_limiter.runner import (
    _MAIN_SUBDIR_SUFFIXES,
    _detect_naming_conflicts,
    run_file_limiter,
)
from .test_runner_core import (
    _CONFIG,
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
