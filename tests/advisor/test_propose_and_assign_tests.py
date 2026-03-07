from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import (
    _advise_set3,
    _assign_placements_chunk,
    _propose_files_step,
    _refine_merge_tiny,
    advise_file_limiter,
    GroupPlacement,
)
from .propose_and_assign_tests import _propose_ok
from .utils_and_helpers_tests import (
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
