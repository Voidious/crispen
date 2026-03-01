"""Tests for file_limiter.runner — 100% branch coverage."""

from __future__ import annotations

from unittest.mock import patch

from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import SplitResult
from crispen.file_limiter.entity_parser import Entity, EntityKind
from crispen.file_limiter.runner import (
    _verify_preservation,
    run_file_limiter,
)

_CONFIG = CrispenConfig()
_PATCH_CLASSIFY = "crispen.file_limiter.runner.classify_entities"
_PATCH_ADVISE = "crispen.file_limiter.runner.advise_file_limiter"
_PATCH_GEN = "crispen.file_limiter.runner.generate_file_splits"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _verify_preservation
# ---------------------------------------------------------------------------


def test_verify_entity_source_in_original():
    post_source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    split = SplitResult(
        new_files={},
        original_source="def foo():\n    pass\n",
        abort=False,
    )
    assert _verify_preservation([entity], split, post_source) is True


def test_verify_entity_source_in_new_file():
    post_source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    split = SplitResult(
        new_files={"utils.py": "def foo():\n    pass"},
        original_source="# original\n",
        abort=False,
    )
    assert _verify_preservation([entity], split, post_source) is True


def test_verify_entity_source_missing():
    post_source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    split = SplitResult(
        new_files={},
        original_source="# nothing relevant\n",
        abort=False,
    )
    assert _verify_preservation([entity], split, post_source) is False


def test_verify_empty_entity_source_skipped():
    # Entity spanning only a blank line → rstrip → "" → falsy → skipped.
    post_source = "\n"
    entity = _make_entity("_block_1", 1, 1)
    split = SplitResult(
        new_files={},
        original_source="# completely different",
        abort=False,
    )
    assert _verify_preservation([entity], split, post_source) is True


# ---------------------------------------------------------------------------
# run_file_limiter — plan.abort path
# ---------------------------------------------------------------------------


@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_plan_abort(mock_classify, mock_advise):
    mock_classify.return_value = _make_classified()
    mock_advise.return_value = _abort_plan()

    result = run_file_limiter("big.py", "", "def foo():\n    pass\n", [], _CONFIG)

    assert result.abort is True
    assert result.new_files == {}
    assert any("cannot be split" in m for m in result.messages)


# ---------------------------------------------------------------------------
# run_file_limiter — no placements path
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# run_file_limiter — verification fails path
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# run_file_limiter — success path
# ---------------------------------------------------------------------------


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
