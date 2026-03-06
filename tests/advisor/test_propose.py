from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import _propose_files_step, advise_file_limiter
from .test_utils import _CONFIG, _classified, _make_entity

_PATCH_KEY = "crispen.file_limiter.advisor.placement_planner.get_api_key"
_PATCH_CLIENT = "crispen.file_limiter.advisor.placement_planner.make_client"
_PATCH_CALL = "crispen.file_limiter.advisor.placement_planner.call_with_tool"


def _propose_ok(*filenames: str) -> dict:
    """Return a valid propose_output_files LLM response for the given filenames."""
    return {
        "files": [{"filename": f, "description": "auto-generated"} for f in filenames]
    }


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
