"""Tests for file_limiter.advisor — 100% branch coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crispen.config import CrispenConfig
from crispen.errors import CrispenAPIError
from crispen.file_limiter.advisor import (
    _PLACEMENT_CHUNK_SIZE,
    _advise_set3,
    _assign_placements_chunk,
    _build_group_mermaid,
    _compute_projected_lines,
    _find_conflicting_placement_indices,
    _group_summary,
    _propose_files_step,
    _refine_merge_tiny,
    advise_file_limiter,
    GroupPlacement,
    resolve_naming_conflicts,
)
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.entity_parser import Entity, EntityKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(
    name: str,
    start: int,
    end: int,
    *,
    docstring=None,
    params=None,
) -> Entity:
    return Entity(
        EntityKind.FUNCTION,
        name,
        start,
        end,
        [name],
        docstring=docstring,
        params=params or [],
    )


def _classified(
    *,
    entities=None,
    entity_class=None,
    graph=None,
    set_1=None,
    set_2_groups=None,
    set_3_groups=None,
    abort=False,
) -> ClassifiedEntities:
    return ClassifiedEntities(
        entities=entities or [],
        entity_class=entity_class or {},
        graph=graph if graph is not None else {},
        set_1=set_1 or [],
        set_2_groups=set_2_groups or [],
        set_3_groups=set_3_groups or [],
        abort=abort,
    )


def _propose_ok(*filenames: str) -> dict:
    """Return a valid propose_output_files LLM response for the given filenames."""
    return {
        "files": [{"filename": f, "description": "auto-generated"} for f in filenames]
    }


_CONFIG = CrispenConfig()
_PATCH_KEY = "crispen.file_limiter.advisor.planning_workflow.get_api_key"
_PATCH_CLIENT = "crispen.file_limiter.advisor.planning_workflow.make_client"
_PATCH_CALL = "crispen.file_limiter.advisor.planning_workflow.call_with_tool"


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


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set2_only_skips_set3_call(mock_key, mock_client, mock_call):
    """set_2 groups only: no set3 call; propose + assign = 2 LLM calls."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        _propose_ok("utils.py"),
        {"placements": [{"group_id": 0, "target_file": "utils.py"}]},
    ]
    c = _classified(
        entities=[_make_entity("foo", 1, 10)],
        set_2_groups=[["foo"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)

    assert plan.abort is False
    assert plan.set3_migrate == []
    assert len(plan.placements) == 1
    assert plan.placements[0].group == ["foo"]
    assert plan.placements[0].target_file == "utils.py"
    assert (
        mock_call.call_count == 2
    )  # propose + assign (no refinement: only 1 tiny file)


# ---------------------------------------------------------------------------
# Set 3 — stay and migrate paths
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_all_stay_no_placement(mock_key, mock_client, mock_call):
    """All Set 3 groups stay → no propose/assign call, empty plan."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {"decisions": [{"group_id": 0, "action": "stay"}]}

    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)

    assert plan.abort is False
    assert plan.set3_migrate == []
    assert plan.placements == []
    assert mock_call.call_count == 1  # only set3 advice call


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_migrate(mock_key, mock_client, mock_call):
    """Set 3 group migrates → set3 + propose + assign = 3 LLM calls."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        {"decisions": [{"group_id": 0, "action": "migrate"}]},
        _propose_ok("helpers.py"),
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
    assert mock_call.call_count == 3


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set2_and_set3_migrate(mock_key, mock_client, mock_call):
    """set_2 + migrating set_3 → both groups in placement call."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        {"decisions": [{"group_id": 0, "action": "migrate"}]},
        _propose_ok("new_stuff.py", "changed.py"),
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
    """Propose succeeds then assignment chunk exhausts retries → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    # file_limiter_retries=0 → 1 attempt for propose, 1 attempt for assign.
    mock_call.side_effect = [
        {"decisions": [{"group_id": 0, "action": "migrate"}]},
        _propose_ok("helpers.py"),
        None,  # assignment fails
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
    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.abort is False
    assert plan.set3_migrate == []
    assert plan.placements == []


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
# Invalid LLM responses — placement assignment
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_incomplete_aborts(mock_key, mock_client, mock_call):
    """Placement missing some group_ids → len mismatch → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    # Two groups but only one placement returned; retries=0 → immediate abort.
    mock_call.side_effect = [
        _propose_ok("utils.py"),
        {"placements": [{"group_id": 0, "target_file": "utils.py"}]},
    ]
    c = _classified(
        entities=[_make_entity("foo", 1, 5), _make_entity("bar", 6, 10)],
        set_2_groups=[["foo"], ["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", CrispenConfig(file_limiter_retries=0))
    assert plan.abort is True


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_duplicate_group_id_aborts(mock_key, mock_client, mock_call):
    """Duplicate group_id in placement → only first counted → len mismatch → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        _propose_ok("utils.py", "other.py"),
        {
            "placements": [
                {"group_id": 0, "target_file": "utils.py"},
                {"group_id": 0, "target_file": "other.py"},  # duplicate
            ]
        },
    ]
    c = _classified(
        entities=[_make_entity("foo", 1, 5), _make_entity("bar", 6, 10)],
        set_2_groups=[["foo"], ["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", CrispenConfig(file_limiter_retries=0))
    assert plan.abort is True


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_empty_target_aborts(mock_key, mock_client, mock_call):
    """Empty target_file → falsy check fails → treated as missing → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        _propose_ok("utils.py"),
        {"placements": [{"group_id": 0, "target_file": ""}]},
    ]
    c = _classified(
        entities=[_make_entity("foo", 1, 5)],
        set_2_groups=[["foo"]],
    )
    plan = advise_file_limiter(c, "src/big.py", CrispenConfig(file_limiter_retries=0))
    assert plan.abort is True


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_out_of_range_group_id_aborts(mock_key, mock_client, mock_call):
    """Out-of-range group_id in placement → skipped → len mismatch → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        _propose_ok("utils.py"),
        {"placements": [{"group_id": 99, "target_file": "utils.py"}]},
    ]
    c = _classified(
        entities=[_make_entity("foo", 1, 5)],
        set_2_groups=[["foo"]],
    )
    plan = advise_file_limiter(c, "src/big.py", CrispenConfig(file_limiter_retries=0))
    assert plan.abort is True


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_non_int_group_id_aborts(mock_key, mock_client, mock_call):
    """Non-integer group_id in placement → isinstance check fails → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        _propose_ok("utils.py"),
        {"placements": [{"group_id": "zero", "target_file": "utils.py"}]},
    ]
    c = _classified(
        entities=[_make_entity("foo", 1, 5)],
        set_2_groups=[["foo"]],
    )
    plan = advise_file_limiter(c, "src/big.py", CrispenConfig(file_limiter_retries=0))
    assert plan.abort is True


# ---------------------------------------------------------------------------
# Placement target not in proposed list → abort
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_targets_outside_proposed_aborts(
    mock_key, mock_client, mock_call
):
    """LLM returns target not in proposed list → constrained check fails → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    # Propose "utils.py" but assignment tries to use "existing.py" (not proposed).
    mock_call.side_effect = [
        _propose_ok("utils.py"),
        {"placements": [{"group_id": 0, "target_file": "existing.py"}]},
    ]
    c = _classified(
        entities=[_make_entity("foo", 1, 5)],
        set_2_groups=[["foo"]],
    )
    plan = advise_file_limiter(
        c,
        "src/big.py",
        CrispenConfig(file_limiter_retries=0),
        existing_files=frozenset({"existing.py"}),
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
    mock_call.side_effect = [
        _propose_ok("utils.py"),
        {"placements": [{"group_id": 0, "target_file": "utils.py"}]},
    ]
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
    """prev_placement_failure is appended to the assignment prompt (last call)."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        _propose_ok("utils.py"),
        {"placements": [{"group_id": 0, "target_file": "utils.py"}]},
    ]

    c = _classified(
        entities=[_make_entity("foo", 1, 10)],
        set_2_groups=[["foo"]],
    )
    advise_file_limiter(
        c, "src/big.py", _CONFIG, prev_placement_failure="sentinel text"
    )

    assert mock_call.call_count == 2  # propose + assign
    # The assign call is the last call; it receives prev_placement_failure.
    messages = mock_call.call_args[0][6]
    assert "sentinel text" in messages[0]["content"]


# ---------------------------------------------------------------------------
# Chunked placement calls (>_PLACEMENT_CHUNK_SIZE groups)
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_chunked_placement_makes_multiple_calls(mock_key, mock_client, mock_call):
    """More than _PLACEMENT_CHUNK_SIZE groups → propose + multiple assign calls."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()

    # Build _PLACEMENT_CHUNK_SIZE + 1 groups so two chunks are needed.
    n = _PLACEMENT_CHUNK_SIZE + 1
    entities = [_make_entity(f"f{i}", i * 2 + 1, i * 2 + 2) for i in range(n)]
    groups = [[f"f{i}"] for i in range(n)]

    # First chunk returns placements for group_ids 0..CHUNK_SIZE-1.
    first_chunk_response = {
        "placements": [
            {"group_id": j, "target_file": "file_a.py"}
            for j in range(_PLACEMENT_CHUNK_SIZE)
        ]
    }
    # Second chunk has 1 group (group_id 0) → goes to file_b.py (tiny, 2 lines).
    second_chunk_response = {
        "placements": [{"group_id": 0, "target_file": "file_b.py"}]
    }
    # Refinement: file_b.py is tiny (2 lines < 200), reassign to file_a.py.
    refine_response = {"placements": [{"group_id": 0, "target_file": "file_a.py"}]}

    mock_call.side_effect = [
        _propose_ok("file_a.py", "file_b.py"),
        first_chunk_response,
        second_chunk_response,
        refine_response,
    ]

    c = _classified(entities=entities, set_2_groups=groups)
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)

    assert plan.abort is False
    assert len(plan.placements) == n
    # propose + chunk1 + chunk2 + refine = 4 calls.
    assert mock_call.call_count == 4


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
            {"group_id": j, "target_file": "file_a.py"}
            for j in range(_PLACEMENT_CHUNK_SIZE)
        ]
    }

    cfg = CrispenConfig(file_limiter_retries=1)  # 2 attempts per chunk
    # propose + chunk 1 (1 call) + chunk 2 (2 failed attempts) = 4 calls.
    mock_call.side_effect = [
        _propose_ok("file_a.py", "file_b.py"),
        first_chunk_response,
        None,
        None,
    ]

    c = _classified(entities=entities, set_2_groups=groups)
    plan = advise_file_limiter(c, "src/big.py", cfg)

    assert plan.abort is True
    assert "LLM failed to assign file placements" in plan.abort_reason
    assert mock_call.call_count == 4  # propose + chunk1 + 2 failed chunk2 attempts


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
            {"group_id": j, "target_file": "file_a.py"}
            for j in range(_PLACEMENT_CHUNK_SIZE)
        ]
    }
    second_chunk_response = {
        "placements": [{"group_id": 0, "target_file": "file_b.py"}]
    }
    # Refinement: file_b.py is tiny, reassign to file_a.py.
    refine_response = {"placements": [{"group_id": 0, "target_file": "file_a.py"}]}

    cfg = CrispenConfig(file_limiter_retries=1)  # 2 attempts per chunk
    # propose + chunk1 + chunk2 (fail) + chunk2 (succeed) + refine = 5 calls.
    mock_call.side_effect = [
        _propose_ok("file_a.py", "file_b.py"),
        first_chunk_response,
        None,
        second_chunk_response,
        refine_response,
    ]

    c = _classified(entities=entities, set_2_groups=groups)
    plan = advise_file_limiter(c, "src/big.py", cfg)

    assert plan.abort is False
    assert len(plan.placements) == n
    assert mock_call.call_count == 5


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_chunked_placement_zero_total_lines(mock_key, mock_client, mock_call):
    """Groups whose names are absent from entity_map → total_lines==0 → fallback."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()

    # Groups reference names not present in entity_map (entities=[]).
    # Projected lines = 0 for all files → no tiny files → no refinement.
    groups = [["orphan_a"], ["orphan_b"]]
    mock_call.side_effect = [
        _propose_ok("a.py", "b.py"),
        {
            "placements": [
                {"group_id": 0, "target_file": "a.py"},
                {"group_id": 1, "target_file": "b.py"},
            ]
        },
    ]

    c = _classified(entities=[], set_2_groups=groups)
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)

    assert plan.abort is False
    assert len(plan.placements) == 2
    assert mock_call.call_count == 2  # propose + assign (no refinement)


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


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_llm_none_returns_none(mock_key, mock_client, mock_call):
    """LLM returns None → resolve returns None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = None
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
def test_resolve_incomplete_response_returns_none(mock_key, mock_client, mock_call):
    """LLM returns fewer placements than groups → len mismatch → None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "models.py"}]  # only 1 of 2
    }
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


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_empty_forbidden_dir_stems(mock_key, mock_client, mock_call):
    """existing_dirs empty → forbidden_dir_stems empty → branch False."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "models.py"},
            {"group_id": 1, "target_file": "services.py"},
        ]
    }
    c = _classified()
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
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "models.py"},
            {"group_id": 1, "target_file": "services.py"},
        ]
    }
    c = _classified()
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


# ---------------------------------------------------------------------------
# _group_summary — enriched descriptions
# ---------------------------------------------------------------------------


def test_group_summary_with_docstring_and_params():
    """Entity with docstring and params → both appear in summary."""
    ent = _make_entity(
        "foo",
        1,
        10,
        docstring="Parse the config file. More details here.",
        params=["path: str", "strict: bool"],
    )
    summary = _group_summary(["foo"], {"foo": ent})
    assert "foo (10 lines)" in summary
    assert '"Parse the config file."' in summary
    assert "params: path: str, strict: bool" in summary


def test_group_summary_with_params_only():
    """Entity with params but no docstring → params appear, no docstring quote."""
    ent = _make_entity("bar", 1, 5, params=["x: int", "y"])
    summary = _group_summary(["bar"], {"bar": ent})
    assert "params: x: int, y" in summary
    assert '"' not in summary


def test_group_summary_docstring_no_period():
    """Docstring with no period → full text used as first sentence."""
    ent = _make_entity("baz", 1, 3, docstring="No period here")
    summary = _group_summary(["baz"], {"baz": ent})
    assert '"No period here"' in summary


# ---------------------------------------------------------------------------
# _build_group_mermaid
# ---------------------------------------------------------------------------


def test_build_group_mermaid_no_edges():
    """Empty graph → no inter-group deps → returns empty string."""
    c = _classified(entities=[], set_2_groups=[["foo"], ["bar"]])
    result = _build_group_mermaid([["foo"], ["bar"]], c)
    assert result == ""


def test_build_group_mermaid_with_inter_group_dep():
    """G0 depends on G1 → Mermaid text with that edge is returned."""
    c = _classified(graph={"foo": {"bar"}, "bar": set()})
    result = _build_group_mermaid([["foo"], ["bar"]], c)
    assert "```mermaid" in result
    assert "G0 --> G1" in result


def test_build_group_mermaid_dep_outside_chunk():
    """Dep to entity outside chunk → dep_gid is None → not added → empty."""
    c = _classified(graph={"foo": {"external"}, "bar": set()})
    result = _build_group_mermaid([["foo"], ["bar"]], c)
    assert result == ""


def test_build_group_mermaid_intra_group_dep():
    """Dep within same SCC group → dep_gid == gid → not added as edge."""
    # foo and baz are in the same group; foo depends on baz (intra-SCC edge)
    c = _classified(graph={"foo": {"baz"}, "baz": {"foo"}, "bar": set()})
    result = _build_group_mermaid([["foo", "baz"], ["bar"]], c)
    assert result == ""


# ---------------------------------------------------------------------------
# Mermaid appears in assignment prompt when deps exist
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_placement_prompt_includes_mermaid_when_deps_exist(
    mock_key, mock_client, mock_call
):
    """Inter-group deps exist → Mermaid diagram included in the assignment prompt."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        _propose_ok("utils.py", "models.py"),
        {
            "placements": [
                {"group_id": 0, "target_file": "utils.py"},
                {"group_id": 1, "target_file": "models.py"},
            ]
        },
    ]
    c = _classified(
        entities=[_make_entity("foo", 1, 5), _make_entity("bar", 6, 10)],
        set_2_groups=[["foo"], ["bar"]],
        graph={"foo": {"bar"}, "bar": set()},
    )
    advise_file_limiter(c, "src/big.py", _CONFIG)

    # The assignment call is the last call; it has the Mermaid diagram.
    messages = mock_call.call_args[0][6]
    assert "```mermaid" in messages[0]["content"]
    assert "G0 --> G1" in messages[0]["content"]


# ---------------------------------------------------------------------------
# verbose=True paths (covers print statements and _counter increment)
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_advise_verbose_set3_and_placement(mock_key, mock_client, mock_call, capsys):
    """verbose=True exercises the print + _counter branches in _advise_set3,
    _propose_files_step, and _assign_placements_chunk."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    # set3 call → propose call → assign call.
    mock_call.side_effect = [
        {"decisions": [{"group_id": 0, "action": "migrate"}]},
        _propose_ok("utils.py"),
        {"placements": [{"group_id": 0, "target_file": "utils.py"}]},
    ]
    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG, verbose=True)

    assert plan.abort is False
    assert plan.llm_calls == 3
    err = capsys.readouterr().err
    assert "set-3 group" in err
    assert "file placements" in err


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_resolve_verbose(mock_client, mock_call, capsys):
    """verbose=True exercises the print + _counter branches in
    _rename_conflicting_chunk (with _counter passed to cover the increment)."""
    mock_client.return_value = MagicMock()
    # Both placements conflict (utils.py vs utils/io.py share stem "utils"),
    # so the chunk sent to LLM has 2 groups; return both renamed.
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "models.py"},
            {"group_id": 1, "target_file": "helpers.py"},
        ]
    }
    entity = _make_entity("foo", 1, 5)
    c = _classified(entities=[entity])
    placements = [
        GroupPlacement(group=["foo"], target_file="utils.py"),
        GroupPlacement(group=["bar"], target_file="utils/io.py"),  # conflict
    ]
    counter = [0]
    result = resolve_naming_conflicts(
        placements,
        c,
        "src/big.py",
        frozenset(),
        frozenset(),
        _CONFIG,
        verbose=True,
        _counter=counter,
    )

    assert result is not None
    assert counter[0] == 1  # one LLM call was counted
    err = capsys.readouterr().err
    assert "naming conflicts" in err


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_advise_set3_no_counter(mock_client, mock_call):
    """_advise_set3 called without _counter covers the None-counter branch."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {"decisions": [{"group_id": 0, "action": "migrate"}]}
    c = _classified(
        entities=[_make_entity("foo", 1, 5)],
        set_3_groups=[["foo"]],
    )
    result = _advise_set3(c, "big.py", mock_client(), _CONFIG)
    assert result == [["foo"]]


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_assign_placements_chunk_no_counter(mock_client, mock_call):
    """_assign_placements_chunk without _counter covers the None-counter branch."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "utils.py"}]
    }
    c = _classified(entities=[_make_entity("foo", 1, 5)])
    result = _assign_placements_chunk(
        [["foo"]], c, "big.py", frozenset(), mock_client(), _CONFIG
    )
    assert result is not None
    assert result[0].target_file == "utils.py"


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_assign_placements_chunk_subdir_name(mock_client, mock_call):
    """subdir_name is included in the prompt and suppresses the plain directory rule."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "detection_flow.py"}]
    }
    c = _classified(entities=[_make_entity("foo", 1, 5)])
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "tests/test_duplicate_extractor.py",
        frozenset(),
        mock_client(),
        _CONFIG,
        subdir_name="duplicate_extractor",
    )
    assert result is not None
    assert result[0].target_file == "detection_flow.py"
    # The prompt should mention the subdirectory and warn against repeating its name.
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "duplicate_extractor/" in prompt
    assert "do not repeat" in prompt.lower()


# ---------------------------------------------------------------------------
# _assign_placements_chunk — constrained mode (proposed_files provided)
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_assign_placements_chunk_constrained_success(mock_client, mock_call):
    """Constrained mode: target in proposed_filenames → placement accepted."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "utils.py"}]
    }
    c = _classified(entities=[_make_entity("foo", 1, 5)])
    proposed = [("utils.py", "general utilities"), ("models.py", "data models")]
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "src/big.py",
        frozenset(),
        mock_client(),
        _CONFIG,
        proposed_files=proposed,
    )
    assert result is not None
    assert result[0].target_file == "utils.py"
    # Prompt should list proposed files and instruct constrained choice.
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "Proposed output files" in prompt
    assert "utils.py" in prompt
    assert "models.py" in prompt


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_assign_placements_chunk_constrained_invalid_target(mock_client, mock_call):
    """Constrained mode: target not in proposed_filenames → immediate None return."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "rogue_file.py"}]
    }
    c = _classified(entities=[_make_entity("foo", 1, 5)])
    proposed = [("utils.py", "general utilities")]
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "src/big.py",
        frozenset(),
        mock_client(),
        _CONFIG,
        proposed_files=proposed,
    )
    assert result is None


# ---------------------------------------------------------------------------
# _propose_files_step — unit tests
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_propose_files_step_success(mock_client, mock_call):
    """Basic success: valid filenames are returned."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "files": [
            {"filename": "utils.py", "description": "utility functions"},
            {"filename": "models.py", "description": "data models"},
        ]
    }
    c = _classified(entities=[_make_entity("foo", 1, 50)])
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), mock_client(), _CONFIG
    )
    assert result is not None
    assert len(result) == 2
    assert result[0] == ("utils.py", "utility functions")
    assert result[1] == ("models.py", "data models")


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_propose_files_step_llm_none(mock_client, mock_call):
    """call_with_tool returns None → _propose_files_step returns None."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = None
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), mock_client(), _CONFIG
    )
    assert result is None


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_propose_files_step_empty_files_list(mock_client, mock_call):
    """LLM returns empty files list → returns None (not proposed)."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {"files": []}
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), mock_client(), _CONFIG
    )
    assert result is None


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_propose_files_step_strips_existing_files(mock_client, mock_call):
    """Filename in existing_files is stripped; remaining valid ones returned."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "files": [
            {"filename": "taken.py", "description": "already exists"},
            {"filename": "utils.py", "description": "new file"},
        ]
    }
    c = _classified()
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset({"taken.py"}),
        mock_client(),
        _CONFIG,
    )
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "utils.py"


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_propose_files_step_all_in_existing_files(mock_client, mock_call):
    """All proposed filenames are in existing_files → stripped → returns None."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "files": [{"filename": "taken.py", "description": "existing"}]
    }
    c = _classified()
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset({"taken.py"}),
        mock_client(),
        _CONFIG,
    )
    assert result is None


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_propose_files_step_strips_duplicates(mock_client, mock_call):
    """Duplicate filenames are stripped; only first occurrence kept."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "files": [
            {"filename": "utils.py", "description": "first"},
            {"filename": "utils.py", "description": "duplicate"},
        ]
    }
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), mock_client(), _CONFIG
    )
    assert result is not None
    assert len(result) == 1
    assert result[0] == ("utils.py", "first")


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_propose_files_step_strips_empty_filename(mock_client, mock_call):
    """Empty filename string is skipped; valid ones returned."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "files": [
            {"filename": "", "description": "empty"},
            {"filename": "utils.py", "description": "valid"},
        ]
    }
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), mock_client(), _CONFIG
    )
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "utils.py"


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_propose_files_step_verbose(mock_client, mock_call, capsys):
    """verbose=True prints propose message to stderr and increments counter."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "files": [{"filename": "utils.py", "description": "utilities"}]
    }
    c = _classified()
    counter = [0]
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset(),
        mock_client(),
        _CONFIG,
        verbose=True,
        _counter=counter,
    )
    assert result is not None
    assert counter[0] == 1
    err = capsys.readouterr().err
    assert "propose" in err.lower()


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_propose_files_step_no_counter(mock_client, mock_call):
    """_counter=None covers the None-counter branch (no increment)."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "files": [{"filename": "utils.py", "description": "utilities"}]
    }
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), mock_client(), _CONFIG
    )
    assert result is not None


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_propose_files_step_subdir_name(mock_client, mock_call):
    """subdir_name triggers the subdir placement_rule branch in the prompt."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "files": [{"filename": "handlers.py", "description": "request handlers"}]
    }
    c = _classified()
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/service.py",
        2,
        frozenset(),
        mock_client(),
        _CONFIG,
        subdir_name="service",
    )
    assert result is not None
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "service/" in prompt


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_propose_files_step_no_existing_files(mock_client, mock_call):
    """existing_files=frozenset() → exclude_section empty (branch False)."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "files": [{"filename": "utils.py", "description": "utilities"}]
    }
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), mock_client(), _CONFIG
    )
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "already exist" not in prompt
    assert result is not None


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_propose_files_step_with_existing_files(mock_client, mock_call):
    """existing_files non-empty → exclude_section added to prompt (branch True)."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "files": [{"filename": "utils.py", "description": "utilities"}]
    }
    c = _classified()
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset({"other.py"}),
        mock_client(),
        _CONFIG,
    )
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "already exist" in prompt
    assert result is not None


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_propose_files_step_prev_failure(mock_client, mock_call):
    """prev_failure is appended to the propose prompt."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "files": [{"filename": "utils.py", "description": "utilities"}]
    }
    c = _classified()
    _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset(),
        mock_client(),
        _CONFIG,
        prev_failure="sentinel_propose_failure",
    )
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "sentinel_propose_failure" in prompt


# ---------------------------------------------------------------------------
# _compute_projected_lines — unit tests
# ---------------------------------------------------------------------------


def test_compute_projected_lines_basic():
    """Entities found in map → lines counted per target file."""
    entity_a = _make_entity("func_a", 1, 50)  # 50 lines
    entity_b = _make_entity("func_b", 51, 100)  # 50 lines
    entity_map = {"func_a": entity_a, "func_b": entity_b}
    placements = [
        GroupPlacement(group=["func_a"], target_file="utils.py"),
        GroupPlacement(group=["func_b"], target_file="utils.py"),
    ]
    projected = _compute_projected_lines(placements, entity_map)
    assert projected == {"utils.py": 100}


def test_compute_projected_lines_unknown_entity():
    """Entity name not in map → no lines added for that entity (skipped)."""
    entity_map = {}  # nothing in the map
    placements = [GroupPlacement(group=["ghost"], target_file="utils.py")]
    projected = _compute_projected_lines(placements, entity_map)
    assert projected == {}


def test_compute_projected_lines_multiple_files():
    """Entities across multiple target files → separate line counts."""
    entity_a = _make_entity("func_a", 1, 100)  # 100 lines
    entity_b = _make_entity("func_b", 101, 200)  # 100 lines
    entity_map = {"func_a": entity_a, "func_b": entity_b}
    placements = [
        GroupPlacement(group=["func_a"], target_file="module_a.py"),
        GroupPlacement(group=["func_b"], target_file="module_b.py"),
    ]
    projected = _compute_projected_lines(placements, entity_map)
    assert projected == {"module_a.py": 100, "module_b.py": 100}


# ---------------------------------------------------------------------------
# _refine_merge_tiny — unit tests
# ---------------------------------------------------------------------------


def test_refine_merge_tiny_no_tiny_files():
    """All projected files are above the tiny threshold → no merge, return unchanged."""
    # Entity with 300 lines is well above min_size (200 for 1000-line limit).
    entity = _make_entity("large_func", 1, 300)
    c = _classified(entities=[entity])
    placements = [GroupPlacement(group=["large_func"], target_file="utils.py")]
    proposed_files = [("utils.py", "large functions"), ("models.py", "models")]

    result = _refine_merge_tiny(
        placements, proposed_files, c, "src/big.py", MagicMock(), _CONFIG
    )
    assert result == placements
    assert result is not placements


def test_refine_merge_tiny_no_ok_proposed():
    """All proposed files are tiny → ok_proposed is empty → return unchanged."""
    # Two tiny entities, both below threshold.
    entity_a = _make_entity("tiny_a", 1, 10)
    entity_b = _make_entity("tiny_b", 11, 20)
    c = _classified(entities=[entity_a, entity_b])
    placements = [
        GroupPlacement(group=["tiny_a"], target_file="a.py"),
        GroupPlacement(group=["tiny_b"], target_file="b.py"),
    ]
    proposed_files = [("a.py", "tiny a"), ("b.py", "tiny b")]
    # Both files are tiny (10 and 10 lines < 200); no ok_proposed → no merge.

    result = _refine_merge_tiny(
        placements, proposed_files, c, "src/big.py", MagicMock(), _CONFIG
    )
    assert result == placements


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_refine_merge_tiny_success(mock_client, mock_call):
    """Tiny file group is merged into a larger file successfully."""
    mock_client.return_value = MagicMock()
    # LLM reassigns the tiny group to the large file.
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "large.py"}]
    }

    entity_small = _make_entity("small_func", 1, 10)  # 10 lines (tiny)
    entity_large = _make_entity("large_func", 11, 310)  # 300 lines (not tiny)
    c = _classified(entities=[entity_small, entity_large])

    placements = [
        GroupPlacement(group=["small_func"], target_file="small.py"),
        GroupPlacement(group=["large_func"], target_file="large.py"),
    ]
    proposed_files = [("small.py", "small"), ("large.py", "large")]
    # small.py: 10 lines (tiny <200); large.py: 300 lines (not tiny).
    # ok_proposed = [("large.py", "large")]; refinement merges small.py into large.py.

    result = _refine_merge_tiny(
        placements, proposed_files, c, "src/big.py", mock_client(), _CONFIG
    )
    assert len(result) == 2
    # small_func should now be in large.py.
    small_placement = next(r for r in result if "small_func" in r.group)
    assert small_placement.target_file == "large.py"
    # large_func remains in large.py.
    large_placement = next(r for r in result if "large_func" in r.group)
    assert large_placement.target_file == "large.py"


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_refine_merge_tiny_llm_fails(mock_client, mock_call):
    """Reassignment LLM returns None → original placements returned (best-effort)."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = None  # LLM fails

    entity_small = _make_entity("small_func", 1, 10)
    entity_large = _make_entity("large_func", 11, 310)
    c = _classified(entities=[entity_small, entity_large])

    placements = [
        GroupPlacement(group=["small_func"], target_file="small.py"),
        GroupPlacement(group=["large_func"], target_file="large.py"),
    ]
    proposed_files = [("small.py", "small"), ("large.py", "large")]

    result = _refine_merge_tiny(
        placements, proposed_files, c, "src/big.py", mock_client(), _CONFIG
    )
    # Best-effort: return original placements unchanged.
    assert len(result) == 2
    assert result[0].target_file == "small.py"
    assert result[1].target_file == "large.py"


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_refine_merge_tiny_verbose(mock_client, mock_call, capsys):
    """verbose=True prints the refining message to stderr."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "large.py"}]
    }

    entity_small = _make_entity("small_func", 1, 10)
    entity_large = _make_entity("large_func", 11, 310)
    c = _classified(entities=[entity_small, entity_large])

    placements = [
        GroupPlacement(group=["small_func"], target_file="small.py"),
        GroupPlacement(group=["large_func"], target_file="large.py"),
    ]
    proposed_files = [("small.py", "small"), ("large.py", "large")]

    _refine_merge_tiny(
        placements,
        proposed_files,
        c,
        "src/big.py",
        mock_client(),
        _CONFIG,
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "refining" in err.lower()


# ---------------------------------------------------------------------------
# Coverage gap: free-form _assign_placements_chunk with existing_files
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_assign_placements_chunk_existing_files_exclude_section(mock_client, mock_call):
    """Free-form mode with non-empty existing_files builds the exclude section."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "helpers.py"}]
    }
    entity = _make_entity("foo", 1, 10)
    c = _classified(entities=[entity], set_2_groups=[["foo"]])
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "src/big.py",
        frozenset({"existing.py"}),  # non-empty existing_files
        mock_client(),
        _CONFIG,
        proposed_files=None,  # free-form mode
    )
    assert result is not None
    assert result[0].target_file == "helpers.py"


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_assign_placements_chunk_target_in_existing_files_returns_none(
    mock_client, mock_call
):
    """Free-form mode: target_file in existing_files → return None (line 589)."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "existing.py"}]
    }
    entity = _make_entity("foo", 1, 10)
    c = _classified(entities=[entity], set_2_groups=[["foo"]])
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "src/big.py",
        frozenset({"existing.py"}),  # target collides with existing file
        mock_client(),
        _CONFIG,
        proposed_files=None,  # free-form mode
    )
    assert result is None


# ---------------------------------------------------------------------------
# Coverage gap: propose retry loop in _assign_placements
# ---------------------------------------------------------------------------


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_propose_retry_succeeds_on_second_attempt(mock_key, mock_client, mock_call):
    """Propose returns None once, then succeeds on retry (lines 862->882, 878)."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    entity = _make_entity("foo", 1, 50)
    c = _classified(entities=[entity], set_2_groups=[["foo"]])
    mock_call.side_effect = [
        None,  # propose fails first attempt
        _propose_ok("helpers.py"),  # propose succeeds on retry
        {"placements": [{"group_id": 0, "target_file": "helpers.py"}]},  # assign
        # no refinement: 50 lines is not tiny (>= 200 is fine, 50 < 200 but only file)
    ]
    plan = advise_file_limiter(
        c,
        "src/big.py",
        CrispenConfig(file_limiter_retries=1),  # allow 1 retry
    )
    assert plan.abort is False
    assert plan.placements[0].target_file == "helpers.py"


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_propose_all_retries_exhausted_aborts(mock_key, mock_client, mock_call):
    """All propose retries fail → _assign_placements returns None → abort (line 883)."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    entity = _make_entity("foo", 1, 50)
    c = _classified(entities=[entity], set_2_groups=[["foo"]])
    mock_call.return_value = None  # propose always fails
    plan = advise_file_limiter(
        c,
        "src/big.py",
        CrispenConfig(file_limiter_retries=0),  # no retries
    )
    assert plan.abort is True
