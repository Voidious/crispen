from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.config import CrispenConfig
from crispen.errors import CrispenAPIError
from crispen.file_limiter.advisor import _PLACEMENT_CHUNK_SIZE, advise_file_limiter
import pytest
from .test_helpers import (
    _CONFIG,
    _PATCH_CALL,
    _PATCH_CLIENT,
    _PATCH_KEY,
    _classified,
    _make_entity,
    _make_llm_result,
    _propose_ok,
)


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


def test_plan_api_key_error_propagates(monkeypatch):
    """Missing API key raises CrispenAPIError before any LLM call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = _classified(
        entities=[_make_entity("foo", 1, 5)],
        set_2_groups=[["foo"]],
    )
    with pytest.raises(CrispenAPIError):
        advise_file_limiter(c, "src/big.py", _CONFIG)


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set2_only_skips_set3_call(mock_key, mock_client, mock_call):
    """set_2 groups only: no set3 call; propose + assign = 2 LLM calls."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        _propose_ok("utils.py"),
        _make_llm_result({"placements": [{"group_id": 0, "target_file": "utils.py"}]}),
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


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_all_stay_no_placement(mock_key, mock_client, mock_call):
    """All Set 3 groups stay → no propose/assign call, empty plan."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = _make_llm_result(
        {"decisions": [{"group_id": 0, "action": "stay"}]}
    )

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
        _make_llm_result({"decisions": [{"group_id": 0, "action": "migrate"}]}),
        _propose_ok("helpers.py"),
        _make_llm_result(
            {"placements": [{"group_id": 0, "target_file": "helpers.py"}]}
        ),
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
def test_plan_set3_test_subdir_skips_advise_call(mock_key, mock_client, mock_call):
    """Test-file subdir split: set-3 groups migrate without an LLM advice call."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    # Only propose + assign calls; no set3-advice call.
    mock_call.side_effect = [
        _propose_ok("test_helpers.py"),
        _make_llm_result(
            {"placements": [{"group_id": 0, "target_file": "test_helpers.py"}]}
        ),
    ]
    c = _classified(
        entities=[_make_entity("test_bar", 1, 10)],
        set_3_groups=[["test_bar"]],
    )
    plan = advise_file_limiter(c, "tests/test_big.py", _CONFIG, subdir_name="big")

    assert plan.abort is False
    assert plan.set3_migrate == [["test_bar"]]
    assert len(plan.placements) == 1
    assert plan.placements[0].target_file == "test_helpers.py"
    assert mock_call.call_count == 2  # no set3-advice call


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set2_and_set3_migrate(mock_key, mock_client, mock_call):
    """set_2 + migrating set_3 → both groups in placement call."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        _make_llm_result({"decisions": [{"group_id": 0, "action": "migrate"}]}),
        _propose_ok("new_stuff.py", "changed.py"),
        _make_llm_result(
            {
                "placements": [
                    {"group_id": 0, "target_file": "new_stuff.py"},
                    {"group_id": 1, "target_file": "changed.py"},
                ]
            }
        ),
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


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_call_returns_none_aborts(mock_key, mock_client, mock_call):
    """Call 1 (set3 advice) returns None → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = _make_llm_result(None)

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
        _make_llm_result({"decisions": [{"group_id": 0, "action": "migrate"}]}),
        _propose_ok("helpers.py"),
        _make_llm_result(None),  # assignment fails
    ]
    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", CrispenConfig(file_limiter_retries=0))
    assert plan.abort is True


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_invalid_group_id_treated_as_stay(mock_key, mock_client, mock_call):
    """Out-of-range group_id in set3 advice → skipped (treated as stay)."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = _make_llm_result(
        {
            "decisions": [
                {"group_id": 99, "action": "migrate"},  # invalid — out of range
                {"group_id": 0, "action": "stay"},
            ]
        }
    )
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
    mock_call.return_value = _make_llm_result(
        {"decisions": [{"group_id": "zero", "action": "migrate"}]}
    )
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
    mock_call.return_value = _make_llm_result(
        {"decisions": [{"group_id": 0, "action": "delete"}]}  # not in enum
    )
    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.set3_migrate == []


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
        _make_llm_result({"placements": [{"group_id": 0, "target_file": "utils.py"}]}),
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
        _make_llm_result(
            {
                "placements": [
                    {"group_id": 0, "target_file": "utils.py"},
                    {"group_id": 0, "target_file": "other.py"},  # duplicate
                ]
            }
        ),
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
        _make_llm_result({"placements": [{"group_id": 0, "target_file": ""}]}),
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
        _make_llm_result({"placements": [{"group_id": 99, "target_file": "utils.py"}]}),
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
        _make_llm_result(
            {"placements": [{"group_id": "zero", "target_file": "utils.py"}]}
        ),
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
def test_plan_placement_targets_outside_proposed_aborts(
    mock_key, mock_client, mock_call
):
    """LLM returns target not in proposed list → constrained check fails → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    # Propose "utils.py" but assignment tries to use "existing.py" (not proposed).
    mock_call.side_effect = [
        _propose_ok("utils.py"),
        _make_llm_result(
            {"placements": [{"group_id": 0, "target_file": "existing.py"}]}
        ),
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


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_entity_not_in_entity_map(mock_key, mock_client, mock_call):
    """Group contains name absent from entity list → falls back to name-only display."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        _propose_ok("utils.py"),
        _make_llm_result({"placements": [{"group_id": 0, "target_file": "utils.py"}]}),
    ]
    # "ghost" is not in entities list, so entity_map lookup fails.
    c = _classified(
        entities=[],
        set_2_groups=[["ghost"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.abort is False
    assert plan.placements[0].target_file == "utils.py"


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_prev_failure_appended_to_prompt(mock_key, mock_client, mock_call):
    """prev_set3_failure is appended to the set3 advice prompt."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = _make_llm_result(
        {"decisions": [{"group_id": 0, "action": "stay"}]}
    )

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
        _make_llm_result({"placements": [{"group_id": 0, "target_file": "utils.py"}]}),
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
    first_chunk_response = _make_llm_result(
        {
            "placements": [
                {"group_id": j, "target_file": "file_a.py"}
                for j in range(_PLACEMENT_CHUNK_SIZE)
            ]
        }
    )
    # Second chunk has 1 group (group_id 0) → goes to file_b.py (tiny, 2 lines).
    second_chunk_response = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "file_b.py"}]}
    )
    # Refinement: file_b.py is tiny (2 lines < 200), reassign to file_a.py.
    refine_response = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "file_a.py"}]}
    )

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
    mock_key, mock_client, mock_call, capsys
):
    """Second chunk exhausts all per-chunk retries → placement returns None → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()

    n = _PLACEMENT_CHUNK_SIZE + 1
    entities = [_make_entity(f"f{i}", i * 2 + 1, i * 2 + 2) for i in range(n)]
    groups = [[f"f{i}"] for i in range(n)]

    first_chunk_response = _make_llm_result(
        {
            "placements": [
                {"group_id": j, "target_file": "file_a.py"}
                for j in range(_PLACEMENT_CHUNK_SIZE)
            ]
        }
    )

    cfg = CrispenConfig(file_limiter_retries=1)  # 2 attempts per chunk
    # propose + chunk 1 (1 call) + chunk 2 (2 failed attempts) = 4 calls.
    mock_call.side_effect = [
        _propose_ok("file_a.py", "file_b.py"),
        first_chunk_response,
        _make_llm_result(None),
        _make_llm_result(None),
    ]

    c = _classified(entities=entities, set_2_groups=groups)
    plan = advise_file_limiter(c, "src/big.py", cfg, verbose=True)

    assert plan.abort is True
    assert "LLM failed to assign file placements" in plan.abort_reason
    assert mock_call.call_count == 4  # propose + chunk1 + 2 failed chunk2 attempts
    assert "failed to assign file placements" in capsys.readouterr().err


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

    first_chunk_response = _make_llm_result(
        {
            "placements": [
                {"group_id": j, "target_file": "file_a.py"}
                for j in range(_PLACEMENT_CHUNK_SIZE)
            ]
        }
    )
    second_chunk_response = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "file_b.py"}]}
    )
    # Refinement: file_b.py is tiny, reassign to file_a.py.
    refine_response = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "file_a.py"}]}
    )

    cfg = CrispenConfig(file_limiter_retries=1)  # 2 attempts per chunk
    # propose + chunk1 + chunk2 (fail) + chunk2 (succeed) + refine = 5 calls.
    mock_call.side_effect = [
        _propose_ok("file_a.py", "file_b.py"),
        first_chunk_response,
        _make_llm_result(None),
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
        _make_llm_result(
            {
                "placements": [
                    {"group_id": 0, "target_file": "a.py"},
                    {"group_id": 1, "target_file": "b.py"},
                ]
            }
        ),
    ]

    c = _classified(entities=[], set_2_groups=groups)
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)

    assert plan.abort is False
    assert len(plan.placements) == 2
    assert mock_call.call_count == 2  # propose + assign (no refinement)


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
        _make_llm_result(
            {
                "placements": [
                    {"group_id": 0, "target_file": "utils.py"},
                    {"group_id": 1, "target_file": "models.py"},
                ]
            }
        ),
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
