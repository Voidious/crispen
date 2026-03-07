from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.file_limiter.advisor import (
    _assign_placements_chunk,
    _compute_projected_lines,
    _propose_files_step,
    _refine_merge_tiny,
    GroupPlacement,
)
from .test_plan import _CONFIG, _PATCH_CALL, _PATCH_CLIENT, _classified, _make_entity


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
