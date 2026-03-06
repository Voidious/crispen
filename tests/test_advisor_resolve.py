from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from crispen.config import CrispenConfig
from crispen.errors import CrispenAPIError
from crispen.file_limiter.advisor import GroupPlacement, resolve_naming_conflicts
from tests.test_advisor_plan_utils import (
    _CONFIG,
    _PATCH_CALL,
    _PATCH_CLIENT,
    _PATCH_KEY,
    _classified,
    _make_entity,
)

_CONFLICTING_PLACEMENTS = [
    GroupPlacement(group=["foo"], target_file="utils.py"),  # plan-vs-plan conflict
    GroupPlacement(group=["bar"], target_file="utils/io.py"),  # plan-vs-plan conflict
    GroupPlacement(group=["baz"], target_file="helpers.py"),  # not conflicting
]

_CLEAN_PLACEMENTS = [
    GroupPlacement(group=["foo"], target_file="utils.py"),
    GroupPlacement(group=["bar"], target_file="helpers.py"),
]


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
