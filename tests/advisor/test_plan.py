from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.config import CrispenConfig
from crispen.errors import CrispenAPIError
from crispen.file_limiter.advisor import (
    _PLACEMENT_CHUNK_SIZE,
    _advise_set3,
    _assign_placements_chunk,
    advise_file_limiter,
)
from .test_utils import _CONFIG, _classified, _make_entity, _propose_ok
import pytest

_PATCH_KEY = "crispen.file_limiter.advisor.conflict_resolution.get_api_key"
_PATCH_CLIENT = "crispen.file_limiter.advisor.conflict_resolution.make_client"
_PATCH_CALL = "crispen.file_limiter.advisor.placement.call_with_tool"


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
def test_advise_set3_with_dep_graph(mock_client, mock_call):
    """_advise_set3 with inter-group dependencies includes mermaid graph in prompt."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {"decisions": [{"group_id": 0, "action": "migrate"}]}
    # graph["foo"] = {"bar"} means foo depends on bar → two groups have an edge
    c = _classified(
        entities=[_make_entity("foo", 1, 5), _make_entity("bar", 6, 10)],
        graph={"foo": {"bar"}},
        set_3_groups=[["foo"], ["bar"]],
    )
    counter = [0]
    result = _advise_set3(c, "big.py", mock_client(), _CONFIG, _counter=counter)
    assert result == [["foo"]]
    # Verify the mermaid graph was injected into the prompt.
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "graph TD" in prompt


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


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
def test_assign_placements_chunk_strips_subdir_prefix(mock_client, mock_call):
    """LLM returns 'subdir/file.py' — the leading subdir/ should be stripped."""
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "duplicate_extractor/detection_flow.py"}
        ]
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
