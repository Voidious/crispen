"""Tests for file_limiter.advisor — 100% branch coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crispen.config import CrispenConfig
from crispen.errors import CrispenAPIError
from crispen.file_limiter.advisor import (
    _PLACEMENT_CHUNK_SIZE,
    _find_conflicting_placement_indices,
    advise_file_limiter,
    GroupPlacement,
    resolve_naming_conflicts,
)
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.entity_parser import Entity, EntityKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(name: str, start: int, end: int) -> Entity:
    return Entity(EntityKind.FUNCTION, name, start, end, [name])


def _classified(
    *,
    entities=None,
    entity_class=None,
    set_1=None,
    set_2_groups=None,
    set_3_groups=None,
    abort=False,
) -> ClassifiedEntities:
    return ClassifiedEntities(
        entities=entities or [],
        entity_class=entity_class or {},
        graph={},
        set_1=set_1 or [],
        set_2_groups=set_2_groups or [],
        set_3_groups=set_3_groups or [],
        abort=abort,
    )


_CONFIG = CrispenConfig()
_PATCH_KEY = "crispen.file_limiter.advisor.get_api_key"
_PATCH_CLIENT = "crispen.file_limiter.advisor.make_client"
_PATCH_CALL = "crispen.file_limiter.advisor.call_with_tool"


# ---------------------------------------------------------------------------
# Early-exit paths (no LLM calls)
# ---------------------------------------------------------------------------


def test_plan_abort_when_classified_abort():
    """classified.abort=True → FileLimiterPlan(abort=True), no LLM calls."""
    c = _classified(abort=True)
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.abort is True
    assert plan.set3_migrate == []
    assert plan.placements == []


def test_plan_no_movable_groups():
    """set_2=[], set_3=[] → empty plan, no LLM calls."""
    c = _classified()
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.abort is False
    assert plan.placements == []


# ---------------------------------------------------------------------------
# API key error propagates
# ---------------------------------------------------------------------------


def test_plan_api_key_error_propagates(monkeypatch):
    """Missing API key raises CrispenAPIError before any LLM call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = _classified(
        entities=[_make_entity("foo", 1, 5)],
        set_2_groups=[["foo"]],
    )
    with pytest.raises(CrispenAPIError):
        advise_file_limiter(c, "src/big.py", _CONFIG)


# ---------------------------------------------------------------------------
# Set 2 only (skip Set 3 call)
# ---------------------------------------------------------------------------


def _setup_mock_and_classified(mock_key, mock_client, mock_call):
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "utils.py"}]
    }
    c = _classified(
        entities=[_make_entity("foo", 1, 10)],
        set_2_groups=[["foo"]],
    )
    return c, mock_call


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set2_only_skips_set3_call(mock_key, mock_client, mock_call):
    """set_2 groups only: call 1 is skipped, call 2 assigns placement."""
    c, mock_call = _setup_mock_and_classified(mock_key, mock_client, mock_call)
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)

    assert plan.abort is False
    assert plan.set3_migrate == []
    assert len(plan.placements) == 1
    assert plan.placements[0].group == ["foo"]
    assert plan.placements[0].target_file == "utils.py"
    assert mock_call.call_count == 1  # only placement call


# ---------------------------------------------------------------------------
# Set 3 — stay and migrate paths
# ---------------------------------------------------------------------------


def _create_plan_and_assert_stay(entity_name: str, file_path: str, config) -> None:
    c = _classified(
        entities=[_make_entity(entity_name, 1, 10)],
        set_3_groups=[[entity_name]],
    )
    plan = advise_file_limiter(c, file_path, config)

    assert plan.abort is False
    assert plan.set3_migrate == []
    assert plan.placements == []

    return plan


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_all_stay_no_placement(mock_key, mock_client, mock_call):
    """All Set 3 groups stay → no placement call, empty plan."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {"decisions": [{"group_id": 0, "action": "stay"}]}

    _create_plan_and_assert_stay("bar", "src/big.py", _CONFIG)
    assert mock_call.call_count == 1  # only set3 advice call


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_migrate(mock_key, mock_client, mock_call):
    """Set 3 group migrates → two LLM calls, placement returned."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        {"decisions": [{"group_id": 0, "action": "migrate"}]},
        {"placements": [{"group_id": 0, "target_file": "helpers.py"}]},
    ]
    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)

    assert plan.abort is False
    assert plan.set3_migrate == [["bar"]]
    assert len(plan.placements) == 1
    assert plan.placements[0].group == ["bar"]
    assert plan.placements[0].target_file == "helpers.py"
    assert mock_call.call_count == 2


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set2_and_set3_migrate(mock_key, mock_client, mock_call):
    """set_2 + migrating set_3 → both groups in placement call."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        {"decisions": [{"group_id": 0, "action": "migrate"}]},
        {
            "placements": [
                {"group_id": 0, "target_file": "new_stuff.py"},
                {"group_id": 1, "target_file": "changed.py"},
            ]
        },
    ]
    c = _classified(
        entities=[_make_entity("foo", 1, 5), _make_entity("bar", 6, 15)],
        set_2_groups=[["foo"]],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)

    assert plan.abort is False
    assert plan.set3_migrate == [["bar"]]
    assert len(plan.placements) == 2
    targets = {p.target_file for p in plan.placements}
    assert targets == {"new_stuff.py", "changed.py"}


# ---------------------------------------------------------------------------
# LLM returns None → abort
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_call_returns_none_aborts(mock_key, mock_client, mock_call):
    """Call 1 (set3 advice) returns None → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = None

    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.abort is True


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_call_returns_none_aborts(mock_key, mock_client, mock_call):
    """Placement chunk exhausts all per-chunk retries → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    # file_limiter_retries=0 → 1 attempt per chunk; set3 call + 1 placement attempt.
    mock_call.side_effect = [
        {"decisions": [{"group_id": 0, "action": "migrate"}]},
        None,
    ]
    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", CrispenConfig(file_limiter_retries=0))
    assert plan.abort is True


# ---------------------------------------------------------------------------
# Invalid LLM responses — set3 advice
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_invalid_group_id_treated_as_stay(mock_key, mock_client, mock_call):
    """Out-of-range group_id in set3 advice → skipped (treated as stay)."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "decisions": [
            {"group_id": 99, "action": "migrate"},  # invalid — out of range
            {"group_id": 0, "action": "stay"},
        ]
    }
    plan = _create_plan_and_assert_stay("bar", "src/big.py", _CONFIG)


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_non_int_group_id_treated_as_stay(mock_key, mock_client, mock_call):
    """Non-integer group_id in set3 advice → isinstance check fails → stay."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {"decisions": [{"group_id": "zero", "action": "migrate"}]}
    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.set3_migrate == []


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_unknown_action_treated_as_stay(mock_key, mock_client, mock_call):
    """Unknown action value in set3 advice → action != 'migrate' → stay."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "decisions": [{"group_id": 0, "action": "delete"}]  # not in enum
    }
    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.set3_migrate == []


# ---------------------------------------------------------------------------
# Invalid LLM responses — placement
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_incomplete_aborts(mock_key, mock_client, mock_call):
    """Placement missing some group_ids → len mismatch → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    # Two groups but only one placement returned.
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "utils.py"}]
    }
    c = _classified(
        entities=[_make_entity("foo", 1, 5), _make_entity("bar", 6, 10)],
        set_2_groups=[["foo"], ["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.abort is True


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_duplicate_group_id_aborts(mock_key, mock_client, mock_call):
    """Duplicate group_id in placement → only first counted → len mismatch → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "utils.py"},
            {"group_id": 0, "target_file": "other.py"},  # duplicate
        ]
    }
    c = _classified(
        entities=[_make_entity("foo", 1, 5), _make_entity("bar", 6, 10)],
        set_2_groups=[["foo"], ["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.abort is True


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_empty_target_aborts(mock_key, mock_client, mock_call):
    """Empty target_file → falsy check fails → treated as missing → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {"placements": [{"group_id": 0, "target_file": ""}]}
    c = _classified(
        entities=[_make_entity("foo", 1, 5)],
        set_2_groups=[["foo"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.abort is True


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_out_of_range_group_id_aborts(mock_key, mock_client, mock_call):
    """Out-of-range group_id in placement → skipped → len mismatch → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 99, "target_file": "utils.py"}]
    }
    c = _classified(
        entities=[_make_entity("foo", 1, 5)],
        set_2_groups=[["foo"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.abort is True


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_non_int_group_id_aborts(mock_key, mock_client, mock_call):
    """Non-integer group_id in placement → isinstance check fails → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": "zero", "target_file": "utils.py"}]
    }
    c = _classified(
        entities=[_make_entity("foo", 1, 5)],
        set_2_groups=[["foo"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.abort is True


# ---------------------------------------------------------------------------
# Placement targets an existing file → abort
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_targets_existing_file_aborts(mock_key, mock_client, mock_call):
    """LLM suggests a target that already exists on disk → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "existing.py"}]
    }
    c = _classified(
        entities=[_make_entity("foo", 1, 5)],
        set_2_groups=[["foo"]],
    )
    plan = advise_file_limiter(
        c, "src/big.py", _CONFIG, existing_files=frozenset({"existing.py"})
    )
    assert plan.abort is True


# ---------------------------------------------------------------------------
# _group_summary: entity not in map
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_entity_not_in_entity_map(mock_key, mock_client, mock_call):
    """Group contains name absent from entity list → falls back to name-only display."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "utils.py"}]
    }
    # "ghost" is not in entities list, so entity_map lookup fails.
    c = _classified(
        entities=[],
        set_2_groups=[["ghost"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.abort is False
    assert plan.placements[0].target_file == "utils.py"


# ---------------------------------------------------------------------------
# prev_failure feedback propagation
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_prev_failure_appended_to_prompt(mock_key, mock_client, mock_call):
    """prev_set3_failure is appended to the set3 advice prompt."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {"decisions": [{"group_id": 0, "action": "stay"}]}

    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    advise_file_limiter(c, "src/big.py", _CONFIG, prev_set3_failure="sentinel text")

    # messages is positional arg index 6 in call_with_tool
    messages = mock_call.call_args[0][6]
    assert "sentinel text" in messages[0]["content"]


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_prev_failure_appended_to_prompt(
    mock_key, mock_client, mock_call
):
    """prev_placement_failure is appended to the placement prompt."""
    c, mock_call = _setup_mock_and_classified(mock_key, mock_client, mock_call)
    advise_file_limiter(
        c, "src/big.py", _CONFIG, prev_placement_failure="sentinel text"
    )

    # set_2_groups only: only the placement call fires (call_count == 1)
    assert mock_call.call_count == 1
    messages = mock_call.call_args[0][6]
    assert "sentinel text" in messages[0]["content"]


# ---------------------------------------------------------------------------
# Chunked placement calls (>_PLACEMENT_CHUNK_SIZE groups)
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_chunked_placement_makes_multiple_calls(mock_key, mock_client, mock_call):
    """More than _PLACEMENT_CHUNK_SIZE set-2 groups → multiple placement calls."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()

    # Build _PLACEMENT_CHUNK_SIZE + 1 groups so two chunks are needed.
    n = _PLACEMENT_CHUNK_SIZE + 1
    entities = [_make_entity(f"f{i}", i * 2 + 1, i * 2 + 2) for i in range(n)]
    groups = [[f"f{i}"] for i in range(n)]

    # First chunk returns placements for group_ids 0..CHUNK_SIZE-1.
    first_chunk_response = {
        "placements": [
            {"group_id": j, "target_file": f"file_{j}.py"}
            for j in range(_PLACEMENT_CHUNK_SIZE)
        ]
    }
    # Second chunk has 1 group (group_id 0).
    second_chunk_response = {"placements": [{"group_id": 0, "target_file": "last.py"}]}

    mock_call.side_effect = [first_chunk_response, second_chunk_response]

    c = _classified(entities=entities, set_2_groups=groups)
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)

    assert plan.abort is False
    assert len(plan.placements) == n
    # Two placement calls were made (no set-3 call since set_3_groups is empty).
    assert mock_call.call_count == 2


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_chunked_placement_second_chunk_fails_aborts(
    mock_key, mock_client, mock_call
):
    """Second chunk exhausts all per-chunk retries → placement returns None → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()

    n = _PLACEMENT_CHUNK_SIZE + 1
    entities = [_make_entity(f"f{i}", i * 2 + 1, i * 2 + 2) for i in range(n)]
    groups = [[f"f{i}"] for i in range(n)]

    first_chunk_response = {
        "placements": [
            {"group_id": j, "target_file": f"file_{j}.py"}
            for j in range(_PLACEMENT_CHUNK_SIZE)
        ]
    }

    cfg = CrispenConfig(file_limiter_retries=1)  # 2 attempts per chunk
    # First chunk succeeds; second chunk fails both attempts.
    mock_call.side_effect = [first_chunk_response, None, None]

    c = _classified(entities=entities, set_2_groups=groups)
    plan = advise_file_limiter(c, "src/big.py", cfg)

    assert plan.abort is True
    assert "LLM failed to assign file placements" in plan.abort_reason
    assert mock_call.call_count == 3  # chunk 1 (1 call) + chunk 2 (2 failed attempts)


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_chunked_placement_chunk_retry_succeeds(mock_key, mock_client, mock_call):
    """A chunk that fails once is retried; on success the full plan is returned."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()

    n = _PLACEMENT_CHUNK_SIZE + 1
    entities = [_make_entity(f"f{i}", i * 2 + 1, i * 2 + 2) for i in range(n)]
    groups = [[f"f{i}"] for i in range(n)]

    first_chunk_response = {
        "placements": [
            {"group_id": j, "target_file": f"file_{j}.py"}
            for j in range(_PLACEMENT_CHUNK_SIZE)
        ]
    }
    second_chunk_response = {"placements": [{"group_id": 0, "target_file": "last.py"}]}

    cfg = CrispenConfig(file_limiter_retries=1)  # 2 attempts per chunk
    # First chunk succeeds; second chunk fails once then succeeds.
    mock_call.side_effect = [first_chunk_response, None, second_chunk_response]

    c = _classified(entities=entities, set_2_groups=groups)
    plan = advise_file_limiter(c, "src/big.py", cfg)

    assert plan.abort is False
    assert len(plan.placements) == n
    assert mock_call.call_count == 3  # chunk 1 (1) + chunk 2 (1 fail + 1 succeed)


# ---------------------------------------------------------------------------
# _find_conflicting_placement_indices
# ---------------------------------------------------------------------------


def test_find_conflicting_idx_plan_vs_plan():
    """Flat file + subdir with same stem both appear → both indices returned."""
    placements = [
        GroupPlacement(group=["foo"], target_file="utils.py"),
        GroupPlacement(group=["bar"], target_file="utils/io.py"),
        GroupPlacement(group=["baz"], target_file="helpers.py"),
    ]
    idxs = _find_conflicting_placement_indices(placements, frozenset(), frozenset())
    assert idxs == [0, 1]


def test_find_conflicting_idx_file_vs_existing_dir():
    """Flat .py target whose stem matches an existing directory → index returned."""
    placements = [GroupPlacement(group=["foo"], target_file="models.py")]
    idxs = _find_conflicting_placement_indices(
        placements, frozenset(), frozenset({"models"})
    )
    assert idxs == [0]


def test_find_conflicting_idx_subdir_vs_existing_file():
    """Subdir target whose top matches an existing .py file → index returned."""
    placements = [GroupPlacement(group=["bar"], target_file="helpers/io.py")]
    idxs = _find_conflicting_placement_indices(
        placements, frozenset({"helpers.py"}), frozenset()
    )
    assert idxs == [0]


def test_find_conflicting_idx_no_conflict():
    """Clean plan with no conflicts → empty list."""
    placements = [
        GroupPlacement(group=["foo"], target_file="utils.py"),
        GroupPlacement(group=["bar"], target_file="helpers.py"),
    ]
    assert (
        _find_conflicting_placement_indices(placements, frozenset(), frozenset()) == []
    )


# ---------------------------------------------------------------------------
# resolve_naming_conflicts — helpers shared by the block below
# ---------------------------------------------------------------------------

_CONFLICTING_PLACEMENTS = [
    GroupPlacement(group=["foo"], target_file="utils.py"),  # plan-vs-plan conflict
    GroupPlacement(group=["bar"], target_file="utils/io.py"),  # plan-vs-plan conflict
    GroupPlacement(group=["baz"], target_file="helpers.py"),  # not conflicting
]

_CLEAN_PLACEMENTS = [
    GroupPlacement(group=["foo"], target_file="utils.py"),
    GroupPlacement(group=["bar"], target_file="helpers.py"),
]


# ---------------------------------------------------------------------------
# resolve_naming_conflicts — tests
# ---------------------------------------------------------------------------


def test_resolve_no_conflicts_returns_unchanged():
    """No conflicts → returns a copy of the input list; no LLM calls needed."""
    c = _classified(entities=[_make_entity("foo", 1, 5), _make_entity("bar", 6, 10)])
    result = resolve_naming_conflicts(
        _CLEAN_PLACEMENTS, c, "src/big.py", frozenset(), frozenset(), _CONFIG
    )
    assert result == _CLEAN_PLACEMENTS
    assert result is not _CLEAN_PLACEMENTS


def test_resolve_api_key_error_propagates(monkeypatch):
    """Missing API key raises CrispenAPIError before any LLM call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = _classified(entities=[_make_entity("foo", 1, 5)])
    with pytest.raises(CrispenAPIError):
        resolve_naming_conflicts(
            _CONFLICTING_PLACEMENTS, c, "src/big.py", frozenset(), frozenset(), _CONFIG
        )


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_success(mock_key, mock_client, mock_call):
    """Happy path: forbidden_dir_stems and existing_file_stems both non-empty;
    prev_failure is False on the first (successful) attempt."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "models.py"},
            {"group_id": 1, "target_file": "services.py"},
        ]
    }
    c = _classified(
        entities=[_make_entity("foo", 1, 5), _make_entity("bar", 6, 10)],
    )
    result = resolve_naming_conflicts(
        _CONFLICTING_PLACEMENTS,
        c,
        "src/big.py",
        existing_files=frozenset({"other.py"}),  # non-empty → existing_file_stems
        existing_dirs=frozenset({"mydir"}),  # non-empty → forbidden_dir_stems
        config=_CONFIG,
    )
    assert result is not None
    assert result[0].target_file == "models.py"
    assert result[1].target_file == "services.py"
    assert result[2].target_file == "helpers.py"  # non-conflicting, unchanged
    assert mock_call.call_count == 1


def _call_resolve_naming_conflicts_and_assert_none():
    c = _classified()
    result = resolve_naming_conflicts(
        _CONFLICTING_PLACEMENTS,
        c,
        "src/big.py",
        frozenset(),
        frozenset(),
        CrispenConfig(file_limiter_retries=0),
    )
    assert result is None


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_llm_none_returns_none(mock_key, mock_client, mock_call):
    """LLM returns None → resolve returns None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = None
    _call_resolve_naming_conflicts_and_assert_none()


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_forbidden_target_returns_none(mock_key, mock_client, mock_call):
    """LLM picks a target that is in forbidden_files → resolve returns None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    # "helpers.py" is a non-conflicting target → included in forbidden_files.
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "helpers.py"},
            {"group_id": 1, "target_file": "services.py"},
        ]
    }
    _call_resolve_naming_conflicts_and_assert_none()


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_incomplete_response_returns_none(mock_key, mock_client, mock_call):
    """LLM returns fewer placements than groups → len mismatch → None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "models.py"}]  # only 1 of 2
    }
    _call_resolve_naming_conflicts_and_assert_none()


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_retry_succeeds(mock_key, mock_client, mock_call):
    """First attempt None, second succeeds; covers if prev_failure: True branch."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        None,
        {
            "placements": [
                {"group_id": 0, "target_file": "models.py"},
                {"group_id": 1, "target_file": "services.py"},
            ]
        },
    ]
    c = _classified()
    result = resolve_naming_conflicts(
        _CONFLICTING_PLACEMENTS,
        c,
        "src/big.py",
        frozenset(),
        frozenset(),
        CrispenConfig(file_limiter_retries=1),
    )
    assert result is not None
    assert result[0].target_file == "models.py"
    assert mock_call.call_count == 2


def _setup_mock_call_and_classified(mock_key, mock_client, mock_call):
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "models.py"},
            {"group_id": 1, "target_file": "services.py"},
        ]
    }
    c = _classified()
    return c


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_empty_forbidden_dir_stems(mock_key, mock_client, mock_call):
    """existing_dirs empty → forbidden_dir_stems empty → branch False."""
    c = _setup_mock_call_and_classified(mock_key, mock_client, mock_call)
    result = resolve_naming_conflicts(
        _CONFLICTING_PLACEMENTS,
        c,
        "src/big.py",
        existing_files=frozenset({"other.py"}),  # file_stems non-empty
        existing_dirs=frozenset(),  # dir_stems empty
        config=_CONFIG,
    )
    assert result is not None


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_empty_existing_file_stems(mock_key, mock_client, mock_call):
    """existing_files=frozenset() → file_stems empty → if existing_file_stems: False."""
    c = _setup_mock_call_and_classified(mock_key, mock_client, mock_call)
    result = resolve_naming_conflicts(
        _CONFLICTING_PLACEMENTS,
        c,
        "src/big.py",
        existing_files=frozenset(),  # file_stems empty
        existing_dirs=frozenset({"mydir"}),  # dir_stems non-empty
        config=_CONFIG,
    )
    assert result is not None


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_non_int_group_id(mock_key, mock_client, mock_call):
    """Non-integer group_id → isinstance check fails → skipped → len mismatch → None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": "zero", "target_file": "models.py"},
            {"group_id": 1, "target_file": "services.py"},
        ]
    }
    _call_resolve_naming_conflicts_and_assert_none()


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_out_of_range_group_id(mock_key, mock_client, mock_call):
    """Out-of-range group_id → range check fails → skipped → len mismatch → None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 99, "target_file": "models.py"},
            {"group_id": 1, "target_file": "services.py"},
        ]
    }
    _call_resolve_naming_conflicts_and_assert_none()


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_duplicate_group_id(mock_key, mock_client, mock_call):
    """Duplicate group_id → second entry skipped → len mismatch → None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "models.py"},
            {"group_id": 0, "target_file": "other.py"},  # duplicate
        ]
    }
    _call_resolve_naming_conflicts_and_assert_none()


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_empty_target(mock_key, mock_client, mock_call):
    """Empty target_file → falsy check fails → skipped → len mismatch → None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": ""},
            {"group_id": 1, "target_file": "services.py"},
        ]
    }
    _call_resolve_naming_conflicts_and_assert_none()
