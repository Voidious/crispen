from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import _PLACEMENT_CHUNK_SIZE, advise_file_limiter
from .test_advisor_plan_core import (
    _CONFIG,
    _PATCH_CALL,
    _PATCH_CLIENT,
    _PATCH_KEY,
    _classified,
    _make_entity,
)


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


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_chunked_placement_zero_total_lines(mock_key, mock_client, mock_call):
    """Groups whose names are absent from entity_map → total_lines==0 → fallback."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()

    # Groups reference names not present in entity_map (entities=[]).
    groups = [["orphan_a"], ["orphan_b"]]
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "a.py"},
            {"group_id": 1, "target_file": "b.py"},
        ]
    }

    c = _classified(entities=[], set_2_groups=groups)
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)

    assert plan.abort is False
    assert len(plan.placements) == 2
