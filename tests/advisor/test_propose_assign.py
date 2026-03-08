from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.file_limiter.advisor import _assign_placements_chunk, _propose_files_step
from .test_plan_core import (
    _CONFIG,
    _PATCH_CALL,
    _PATCH_CLIENT,
    _classified,
    _make_entity,
)


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
