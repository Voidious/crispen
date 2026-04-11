from __future__ import annotations
from unittest.mock import patch
from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.code_gen import SplitResult
from crispen.file_limiter.runner import run_file_limiter
from .test_runner_misc import (
    _CONFIG,
    _PATCH_ADVISE,
    _PATCH_CLASSIFY,
    _PATCH_GEN,
    _make_classified,
    _make_entity,
    _plan_with,
)


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
    result = run_file_limiter("tests/noexist/test_big.py", "", source, [], _CONFIG)

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
