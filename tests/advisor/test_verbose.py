from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import (
    GroupPlacement,
    _LLMAccumulator,
    _advise_set3,
    _assign_placements_chunk,
    _propose_files_step,
    advise_file_limiter,
    resolve_naming_conflicts,
)
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
        _make_llm_result({"decisions": [{"group_id": 0, "action": "migrate"}]}),
        _propose_ok("utils.py"),
        _make_llm_result({"placements": [{"group_id": 0, "target_file": "utils.py"}]}),
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
@patch(_PATCH_KEY)
def test_resolve_verbose(mock_key, mock_client, mock_call, capsys):
    """verbose=True exercises the print + _counter branches in
    _rename_conflicting_chunk (with _counter passed to cover the increment)."""
    mock_key.return_value = "key"
    # Both placements conflict (utils.py vs utils/io.py share stem "utils"),
    # so the chunk sent to LLM has 2 groups; return both renamed.
    mock_call.return_value = _make_llm_result(
        {
            "placements": [
                {"group_id": 0, "target_file": "models.py"},
                {"group_id": 1, "target_file": "helpers.py"},
            ]
        }
    )
    entity = _make_entity("foo", 1, 5)
    c = _classified(entities=[entity])
    placements = [
        GroupPlacement(group=["foo"], target_file="utils.py"),
        GroupPlacement(group=["bar"], target_file="utils/io.py"),  # conflict
    ]
    acc = _LLMAccumulator()
    result = resolve_naming_conflicts(
        placements,
        c,
        "src/big.py",
        frozenset(),
        frozenset(),
        _CONFIG,
        verbose=True,
        _acc=acc,
    )

    assert result is not None
    assert acc.calls == 1  # one LLM call was counted
    err = capsys.readouterr().err
    assert "naming conflicts" in err


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_advise_verbose_detailed_timing_prints(
    mock_key, mock_client, mock_call, capsys
):
    """timing='detailed' prints per-call → done lines for set3, propose, and assign."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    # set3 call → propose call → assign call.
    mock_call.side_effect = [
        _make_llm_result({"decisions": [{"group_id": 0, "action": "migrate"}]}),
        _propose_ok("utils.py"),
        _make_llm_result({"placements": [{"group_id": 0, "target_file": "utils.py"}]}),
    ]
    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(
        c, "src/big.py", _CONFIG, verbose=True, timing="detailed"
    )

    assert plan.abort is False
    err = capsys.readouterr().err
    assert "→ done [" in err


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_verbose_detailed_timing_print(
    mock_key, mock_client, mock_call, capsys
):
    """timing='detailed' prints per-call → done line in resolve_naming_conflicts."""
    mock_key.return_value = "key"
    mock_call.return_value = _make_llm_result(
        {
            "placements": [
                {"group_id": 0, "target_file": "models.py"},
                {"group_id": 1, "target_file": "helpers.py"},
            ]
        }
    )
    entity = _make_entity("foo", 1, 5)
    c = _classified(entities=[entity])
    placements = [
        GroupPlacement(group=["foo"], target_file="utils.py"),
        GroupPlacement(group=["bar"], target_file="utils/io.py"),  # conflict
    ]
    acc = _LLMAccumulator()
    result = resolve_naming_conflicts(
        placements,
        c,
        "src/big.py",
        frozenset(),
        frozenset(),
        _CONFIG,
        verbose=True,
        timing="detailed",
        _acc=acc,
    )

    assert result is not None
    err = capsys.readouterr().err
    assert "→ done [" in err


@patch(_PATCH_CALL)
def test_advise_set3_no_counter(mock_call):
    """_advise_set3 called without _counter covers the None-counter branch."""
    mock_call.return_value = _make_llm_result(
        {"decisions": [{"group_id": 0, "action": "migrate"}]}
    )
    c = _classified(
        entities=[_make_entity("foo", 1, 5)],
        set_3_groups=[["foo"]],
    )
    result = _advise_set3(c, "big.py", MagicMock(), _CONFIG)
    assert result == [["foo"]]


@patch(_PATCH_CALL)
def test_advise_set3_with_dep_graph(mock_call):
    """_advise_set3 with inter-group dependencies includes mermaid graph in prompt."""
    mock_call.return_value = _make_llm_result(
        {"decisions": [{"group_id": 0, "action": "migrate"}]}
    )
    # graph["foo"] = {"bar"} means foo depends on bar → two groups have an edge
    c = _classified(
        entities=[_make_entity("foo", 1, 5), _make_entity("bar", 6, 10)],
        graph={"foo": {"bar"}},
        set_3_groups=[["foo"], ["bar"]],
    )
    result = _advise_set3(c, "big.py", MagicMock(), _CONFIG)
    assert result == [["foo"]]
    # Verify the mermaid graph was injected into the prompt.
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "graph TD" in prompt


@patch(_PATCH_CALL)
def test_assign_placements_chunk_no_counter(mock_call):
    """_assign_placements_chunk without _counter covers the None-counter branch."""
    mock_call.return_value = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "utils.py"}]}
    )
    c = _classified(entities=[_make_entity("foo", 1, 5)])
    result = _assign_placements_chunk(
        [["foo"]], c, "big.py", frozenset(), MagicMock(), _CONFIG
    )
    assert result is not None
    assert result[0].target_file == "utils.py"


@patch(_PATCH_CALL)
def test_assign_placements_chunk_subdir_name(mock_call):
    """subdir_name is included in the prompt and suppresses the plain directory rule."""
    mock_call.return_value = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "detection_flow.py"}]}
    )
    c = _classified(entities=[_make_entity("foo", 1, 5)])
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "tests/test_duplicate_extractor.py",
        frozenset(),
        MagicMock(),
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
def test_assign_placements_chunk_strips_subdir_prefix(mock_call):
    """LLM returns 'subdir/file.py' — the leading subdir/ should be stripped."""
    mock_call.return_value = _make_llm_result(
        {
            "placements": [
                {"group_id": 0, "target_file": "duplicate_extractor/detection_flow.py"}
            ]
        }
    )
    c = _classified(entities=[_make_entity("foo", 1, 5)])
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "tests/test_duplicate_extractor.py",
        frozenset(),
        MagicMock(),
        _CONFIG,
        subdir_name="duplicate_extractor",
    )
    assert result is not None
    assert result[0].target_file == "detection_flow.py"


@patch(_PATCH_CALL)
def test_assign_placements_chunk_constrained_success(mock_call):
    """Constrained mode: target in proposed_filenames → placement accepted."""
    mock_call.return_value = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "utils.py"}]}
    )
    c = _classified(entities=[_make_entity("foo", 1, 5)])
    proposed = [("utils.py", "general utilities"), ("models.py", "data models")]
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "src/big.py",
        frozenset(),
        MagicMock(),
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
def test_assign_placements_chunk_constrained_invalid_target(mock_call):
    """Constrained mode: target not in proposed_filenames → immediate None return."""
    mock_call.return_value = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "rogue_file.py"}]}
    )
    c = _classified(entities=[_make_entity("foo", 1, 5)])
    proposed = [("utils.py", "general utilities")]
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "src/big.py",
        frozenset(),
        MagicMock(),
        _CONFIG,
        proposed_files=proposed,
    )
    assert result is None


@patch(_PATCH_CALL)
def test_propose_files_step_success(mock_call):
    """Basic success: valid filenames are returned."""
    mock_call.return_value = _make_llm_result(
        {
            "files": [
                {"filename": "utils.py", "description": "utility functions"},
                {"filename": "models.py", "description": "data models"},
            ]
        }
    )
    c = _classified(entities=[_make_entity("foo", 1, 50)])
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), MagicMock(), _CONFIG
    )
    assert result is not None
    assert len(result) == 2
    assert result[0] == ("utils.py", "utility functions")
    assert result[1] == ("models.py", "data models")


@patch(_PATCH_CALL)
def test_propose_files_step_llm_none(mock_call):
    """call_with_tool returns None → _propose_files_step returns None."""
    mock_call.return_value = _make_llm_result(None)
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), MagicMock(), _CONFIG
    )
    assert result is None


@patch(_PATCH_CALL)
def test_propose_files_step_empty_files_list(mock_call):
    """LLM returns empty files list → returns None (not proposed)."""
    mock_call.return_value = _make_llm_result({"files": []})
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), MagicMock(), _CONFIG
    )
    assert result is None


@patch(_PATCH_CALL)
def test_propose_files_step_strips_existing_files(mock_call):
    """Filename in existing_files is stripped; remaining valid ones returned."""
    mock_call.return_value = _make_llm_result(
        {
            "files": [
                {"filename": "taken.py", "description": "already exists"},
                {"filename": "utils.py", "description": "new file"},
            ]
        }
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset({"taken.py"}),
        MagicMock(),
        _CONFIG,
    )
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "utils.py"


@patch(_PATCH_CALL)
def test_propose_files_step_all_in_existing_files(mock_call):
    """All proposed filenames are in existing_files → stripped → returns None."""
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "taken.py", "description": "existing"}]}
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset({"taken.py"}),
        MagicMock(),
        _CONFIG,
    )
    assert result is None


@patch(_PATCH_CALL)
def test_propose_files_step_strips_duplicates(mock_call):
    """Duplicate filenames are stripped; only first occurrence kept."""
    mock_call.return_value = _make_llm_result(
        {
            "files": [
                {"filename": "utils.py", "description": "first"},
                {"filename": "utils.py", "description": "duplicate"},
            ]
        }
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), MagicMock(), _CONFIG
    )
    assert result is not None
    assert len(result) == 1
    assert result[0] == ("utils.py", "first")


@patch(_PATCH_CALL)
def test_propose_files_step_strips_empty_filename(mock_call):
    """Empty filename string is skipped; valid ones returned."""
    mock_call.return_value = _make_llm_result(
        {
            "files": [
                {"filename": "", "description": "empty"},
                {"filename": "utils.py", "description": "valid"},
            ]
        }
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), MagicMock(), _CONFIG
    )
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "utils.py"


@patch(_PATCH_CALL)
def test_propose_files_step_verbose(mock_call, capsys):
    """verbose=True prints propose message to stderr and increments counter."""
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "utils.py", "description": "utilities"}]}
    )
    c = _classified()
    acc = _LLMAccumulator()
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset(),
        MagicMock(),
        _CONFIG,
        verbose=True,
        _acc=acc,
    )
    assert result is not None
    assert acc.calls == 1
    err = capsys.readouterr().err
    assert "propose" in err.lower()


@patch(_PATCH_CALL)
def test_propose_files_step_no_counter(mock_call):
    """_counter=None covers the None-counter branch (no increment)."""
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "utils.py", "description": "utilities"}]}
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), MagicMock(), _CONFIG
    )
    assert result is not None


@patch(_PATCH_CALL)
def test_propose_files_step_subdir_name(mock_call):
    """subdir_name triggers the subdir placement_rule branch in the prompt."""
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "handlers.py", "description": "request handlers"}]}
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/service.py",
        2,
        frozenset(),
        MagicMock(),
        _CONFIG,
        subdir_name="service",
    )
    assert result is not None
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "service/" in prompt


@patch(_PATCH_CALL)
def test_propose_files_step_no_existing_files(mock_call):
    """existing_files=frozenset() → exclude_section empty (branch False)."""
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "utils.py", "description": "utilities"}]}
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), MagicMock(), _CONFIG
    )
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "already exist" not in prompt
    assert result is not None


@patch(_PATCH_CALL)
def test_propose_files_step_with_existing_files(mock_call):
    """existing_files non-empty → exclude_section added to prompt (branch True)."""
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "utils.py", "description": "utilities"}]}
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset({"other.py"}),
        MagicMock(),
        _CONFIG,
    )
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "already exist" in prompt
    assert result is not None


@patch(_PATCH_CALL)
def test_propose_files_step_prev_failure(mock_call):
    """prev_failure is appended to the propose prompt."""
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "utils.py", "description": "utilities"}]}
    )
    c = _classified()
    _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset(),
        MagicMock(),
        _CONFIG,
        prev_failure="sentinel_propose_failure",
    )
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "sentinel_propose_failure" in prompt


@patch(_PATCH_CALL)
def test_assign_placements_chunk_existing_files_exclude_section(mock_call):
    """Free-form mode with non-empty existing_files builds the exclude section."""
    mock_call.return_value = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "helpers.py"}]}
    )
    entity = _make_entity("foo", 1, 10)
    c = _classified(entities=[entity], set_2_groups=[["foo"]])
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "src/big.py",
        frozenset({"existing.py"}),  # non-empty existing_files
        MagicMock(),
        _CONFIG,
        proposed_files=None,  # free-form mode
    )
    assert result is not None
    assert result[0].target_file == "helpers.py"


@patch(_PATCH_CALL)
def test_assign_placements_chunk_target_in_existing_files_returns_none(mock_call):
    """Free-form mode: target_file in existing_files → return None (line 589)."""
    mock_call.return_value = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "existing.py"}]}
    )
    entity = _make_entity("foo", 1, 10)
    c = _classified(entities=[entity], set_2_groups=[["foo"]])
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "src/big.py",
        frozenset({"existing.py"}),  # target collides with existing file
        MagicMock(),
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
        _make_llm_result(None),  # propose fails first attempt
        _propose_ok("helpers.py"),  # propose succeeds on retry
        _make_llm_result(
            {"placements": [{"group_id": 0, "target_file": "helpers.py"}]}
        ),  # assign
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
    mock_call.return_value = _make_llm_result(None)  # propose always fails
    plan = advise_file_limiter(
        c,
        "src/big.py",
        CrispenConfig(file_limiter_retries=0),  # no retries
    )
    assert plan.abort is True


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_propose_no_tool_call_verbose(mock_key, mock_client, mock_call, capsys):
    """tool_input=None + verbose=True → logs 'no tool call in response'."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    entity = _make_entity("foo", 1, 50)
    c = _classified(entities=[entity], set_2_groups=[["foo"]])
    mock_call.return_value = _make_llm_result(None)
    plan = advise_file_limiter(
        c, "src/big.py", CrispenConfig(file_limiter_retries=0), verbose=True
    )
    assert plan.abort is True
    assert "no tool call" in capsys.readouterr().err


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_propose_empty_files_list_verbose(mock_key, mock_client, mock_call, capsys):
    """tool_input={"files": []} + verbose=True → logs 'empty files list'."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    entity = _make_entity("foo", 1, 50)
    c = _classified(entities=[entity], set_2_groups=[["foo"]])
    mock_call.return_value = _make_llm_result({"files": []})
    plan = advise_file_limiter(
        c, "src/big.py", CrispenConfig(file_limiter_retries=0), verbose=True
    )
    assert plan.abort is True
    assert "empty files list" in capsys.readouterr().err


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_propose_all_filenames_filtered_verbose(
    mock_key, mock_client, mock_call, capsys
):
    """All proposed filenames in existing_files + verbose=True → logs filtered names."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    entity = _make_entity("foo", 1, 50)
    c = _classified(entities=[entity], set_2_groups=[["foo"]])
    # Propose "taken.py" which is already in existing_files.
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "taken.py", "description": "existing"}]}
    )
    plan = advise_file_limiter(
        c,
        "src/big.py",
        CrispenConfig(file_limiter_retries=0),
        existing_files=frozenset({"taken.py"}),
        verbose=True,
    )
    assert plan.abort is True
    assert "filtered" in capsys.readouterr().err
