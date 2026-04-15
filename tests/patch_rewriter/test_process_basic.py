from __future__ import annotations
from unittest.mock import MagicMock, patch as mock_patch
from crispen.config import CrispenConfig
from crispen.llm_client import LLMCallResult
from crispen.patch_rewriter import (
    RewriteAccumulator,
    _FLContext,
    _build_context_message,
    _process_file_source,
)
from .context_builders import (
    _CFG,
    _CFG_NO_LLM_VERIFY,
    _CLASSIFY_NO_CHANGE,
    _CLASSIFY_RENAME,
    _FORKING_PATHS,
    _PATCH_CALL_TOOL,
    _REWRITE_VERIFY_OK,
    _SRC_WITH_PATCH,
    _VERIFY_OK,
    _VERIFY_REJECT,
    _VERIFY_REJECT_WITH_CORRECTIONS,
    _ok,
)
from .process_rewrite import _VALID_REWRITE
from . import process_basic
from .process_basic import _PATCH_CALL_PR, _PATCH_GET_KEY_PR, _make_process_cfg


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_functions(mock_call):
    src = "def test_f(): pass\n"
    result, changed, cross = _process_file_source(
        src, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert result == src
    assert changed is False
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL, return_value=_ok(None))
def test_process_classify_tool_none(mock_call):
    # Classify returns tool_input=None → break, no update.
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_needed(mock_call):
    # Classify returns empty renames → verify confirms no-change → no update.
    mock_call.side_effect = [_ok(_CLASSIFY_NO_CHANGE), _ok(_VERIFY_OK)]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False
    assert mock_call.call_count == 2  # classify + verify


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_verify_none_accept(mock_call):
    # Classify says no change; verify returns None → accept no-change.
    mock_call.side_effect = [_ok(_CLASSIFY_NO_CHANGE), _ok(None)]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False
    assert mock_call.call_count == 2


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_verify_rejects_then_accepts(mock_call):
    # No-change verify rejects first; classify+verify accepted on retry.
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT),
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is True
    assert "crispen.after.X" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_verify_retries_exhausted(mock_call):
    # llm_verify_retries=0: no escalation, accept no-change immediately.

    cfg = CrispenConfig(patch_update_retries=1, llm_verify_retries=0)
    mock_call.side_effect = [_ok(_CLASSIFY_NO_CHANGE), _ok(_VERIFY_REJECT)]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), cfg, 1
    )
    assert changed is False
    assert mock_call.call_count == 2


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_exhausted_escalates_to_rewrite(mock_call):
    # When llm_verify_retries>0 and no-change retries are exhausted, escalate
    # to the full rewrite path seeded with the verifier's explanation.

    cfg = CrispenConfig(patch_update_retries=3, llm_verify_retries=1)
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),  # classify → no change
        _ok(_VERIFY_REJECT),  # verify → reject
        _ok(_CLASSIFY_NO_CHANGE),  # classify (retry) → no change again
        _ok(_VERIFY_REJECT),  # verify → reject (retries exhausted → escalate)
        _ok({"rewritten_function": _VALID_REWRITE}),  # rewrite (escalated)
        _ok(_REWRITE_VERIFY_OK),  # verify rewrite → accept
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), cfg, 3
    )
    assert changed is True
    assert mock_call.call_count == 6


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_exhausted_escalate_verbose(mock_call, capsys):
    # verbose=True prints 'escalating to rewrite' when escalation is triggered.

    cfg = CrispenConfig(patch_update_retries=3, llm_verify_retries=1)
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT),
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        cfg,
        3,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "escalating to rewrite" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_applied(mock_call):
    # No-change verify returns corrections → corrections-verify accepts → apply.
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),
        _ok(_VERIFY_OK),  # corrections-verify accepts
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is True
    assert "crispen.after.X" in result
    assert mock_call.call_count == 3


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_verify_none_accept(mock_call):
    # Corrections-verify returns tool_input=None → accept.
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),
        _ok(None),  # corrections-verify returns None → accept
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is True
    assert "crispen.after.X" in result
    assert mock_call.call_count == 3


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_verify_fails_retry(mock_call):
    # Corrections-verify rejects → retries left → retry classify which succeeds.

    cfg = CrispenConfig(patch_update_retries=3, llm_verify_retries=1)
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),  # classify → no change
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),  # verify → reject + corrections
        _ok(_VERIFY_REJECT),  # corrections-verify → rejected
        _ok(_CLASSIFY_RENAME),  # classify (retry) → rename
        _ok(_VERIFY_OK),  # rename verify → accept
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), cfg, 3
    )
    assert changed is True
    assert "crispen.after.X" in result
    assert mock_call.call_count == 5


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_verbose(mock_call, capsys):
    # verbose=True prints 'verifying corrections for' and 'ACCEPTED'.
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),
        _ok(_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        2,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "verifying corrections for" in err
    assert "ACCEPTED" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_verbose_reject(mock_call, capsys):
    # verbose=True prints 'REJECTED' and issue when corrections-verify rejects.

    cfg = CrispenConfig(patch_update_retries=3, llm_verify_retries=1)
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),
        _ok({"correct": False, "issue": "correction still wrong", "corrections": {}}),
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        cfg,
        3,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "corrections verify REJECTED" in err
    assert "correction still wrong" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_timing_detailed(mock_call, capsys):
    # timing='detailed' prints elapsed/token info after corrections-verify call.

    cfg = CrispenConfig(patch_update_retries=2, timing="detailed")
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_NO_CHANGE,
            elapsed=0.5,
            input_tokens=100,
            output_tokens=10,
        ),
        LLMCallResult(
            tool_input=_VERIFY_REJECT_WITH_CORRECTIONS,
            elapsed=0.4,
            input_tokens=90,
            output_tokens=20,
        ),
        LLMCallResult(
            tool_input=_VERIFY_OK,
            elapsed=0.3,
            input_tokens=80,
            output_tokens=5,
        ),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        cfg,
        2,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "0.30s" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_acc(mock_call):
    # _acc accumulates calls from classify, no-change verify, and corrections-verify.
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_NO_CHANGE,
            elapsed=0.5,
            input_tokens=100,
            output_tokens=10,
        ),
        LLMCallResult(
            tool_input=_VERIFY_REJECT_WITH_CORRECTIONS,
            elapsed=0.4,
            input_tokens=90,
            output_tokens=20,
        ),
        LLMCallResult(
            tool_input=_VERIFY_OK,
            elapsed=0.3,
            input_tokens=80,
            output_tokens=5,
        ),
    ]
    acc = RewriteAccumulator()
    _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2, _acc=acc
    )
    assert acc.calls == 3
    assert abs(acc.elapsed - 1.2) < 1e-9
    assert acc.input_tokens == 270
    assert acc.output_tokens == 35


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_no_splice(mock_call, tmp_path):
    # Corrections-verify accepts; function uses const ref → no splice; const updated.
    src = (
        'TARGET = "crispen.before.X"\n\n@patch(TARGET)\ndef test_f(mock_x):\n    pass\n'
    )
    scan = str(tmp_path / "test_foo.py")
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_REJECT_WITH_CORRECTIONS),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src, {"crispen.before.X"}, "ctx", MagicMock(), _CFG, 2, scan_file=scan
    )
    assert changed is True
    assert "crispen.after.X" in result
    assert mock_call.call_count == 3


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_name_invariant_filtered(mock_call):
    # Verifier proposes corrections that rename the patched name itself
    # (e.g. X → Y).  These must be filtered out; with an empty corrections set
    # the no-change result falls through to retry logic — here retries=1 so
    # the second classify call is made and returns no-change confirmed by verify.
    verify_name_change_correction = {
        "correct": False,
        "issue": "module moved",
        "corrections": {"crispen.before.X": "crispen.before.Y"},  # name changed!
    }
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(verify_name_change_correction),
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_OK),
    ]

    cfg = CrispenConfig(patch_update_retries=3, llm_verify_retries=1)
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), cfg, 3
    )
    # Correction was filtered (name changed X→Y) — no change applied.
    assert "crispen.before.Y" not in result
    assert mock_call.call_count == 4


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_corrections_still_imported_guard(mock_call):
    # Verifier proposes corrections that move a name listed as still-imported in
    # the context message to a non-submodule path.  The second still-imported
    # filter drops the correction; with empty corrections the retry loop resumes
    # and accepts no-change on verify.
    still_imported_ctx = (
        "Names still externally imported in the modified original (check):\n" "- `X`\n"
    )
    verify_still_imported_correction = {
        "correct": False,
        "issue": "hallucinated move",
        "corrections": {"crispen.before.X": "crispen.sub.X"},  # X is still in orig
    }
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(verify_still_imported_correction),
        _ok(_CLASSIFY_NO_CHANGE),
        _ok(_VERIFY_OK),
    ]

    cfg = CrispenConfig(patch_update_retries=3, llm_verify_retries=1)
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        still_imported_ctx,
        MagicMock(),
        cfg,
        3,
        still_imported={"X"},
    )
    # Correction was filtered (X still imported, non-submodule target) —
    # no change applied; retry loop accepted no-change on subsequent verify.
    assert "crispen.sub.X" not in result
    assert mock_call.call_count == 4


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_verify_verbose(mock_call, capsys):
    # verbose=True prints 'verifying no-change' and 'ACCEPTED'.
    mock_call.side_effect = [_ok(_CLASSIFY_NO_CHANGE), _ok(_VERIFY_OK)]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "verifying no-change" in err
    assert "ACCEPTED" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_verify_verbose_reject(mock_call, capsys):
    # verbose=True prints 'REJECTED' and the issue when no-change verify rejects.
    mock_call.side_effect = [
        _ok(_CLASSIFY_NO_CHANGE),
        _ok({"correct": False, "issue": "patch still points to old module"}),
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        2,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "REJECTED" in err
    assert "patch still points to old module" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_verify_timing_detailed(mock_call, capsys):
    # timing='detailed' appends elapsed/token info after the no-change verify call.

    cfg = CrispenConfig(patch_update_retries=1, timing="detailed")
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_NO_CHANGE,
            elapsed=0.5,
            input_tokens=100,
            output_tokens=10,
        ),
        LLMCallResult(
            tool_input=_VERIFY_OK,
            elapsed=0.3,
            input_tokens=80,
            output_tokens=5,
        ),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        cfg,
        1,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "→ done" in err
    assert "0.30s" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_change_acc_accumulates(mock_call):
    # _acc accumulates calls from both classify and no-change verify; no_change counted.
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_NO_CHANGE,
            elapsed=0.5,
            input_tokens=100,
            output_tokens=10,
        ),
        LLMCallResult(
            tool_input=_VERIFY_OK,
            elapsed=0.3,
            input_tokens=80,
            output_tokens=5,
        ),
    ]
    acc = RewriteAccumulator()
    _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1, _acc=acc
    )
    assert acc.calls == 2
    assert abs(acc.elapsed - 0.8) < 1e-9
    assert acc.input_tokens == 180
    assert acc.output_tokens == 15
    assert acc.no_change == 1
    assert acc.rename == 0
    assert acc.rewrite == 0
    assert acc.edit_failures == 0


@mock_patch(_PATCH_CALL_TOOL, return_value=_ok(None))
def test_process_acc_edit_failure_on_classify_none(mock_call):
    # Classify returns tool_input=None → edit_failures incremented.
    acc = RewriteAccumulator()
    _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1, _acc=acc
    )
    assert acc.edit_failures == 1
    assert acc.no_change == 0
    assert acc.rename == 0
    assert acc.rewrite == 0


@mock_patch(
    _PATCH_CALL_TOOL,
    return_value=_ok(
        {
            "needs_rewrite": False,
            "patch_renames": {"crispen.before.X": "crispen.before.X"},
        }
    ),
)
def test_process_same_path_filtered_out(mock_call):
    # Rename where old == new → filtered to empty → triggers no-change verify.
    # return_value repeats for both calls; verify gets wrong type → rejects; retries
    # exhaust → accept no-change.
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False


@mock_patch(
    _PATCH_CALL_TOOL,
    return_value=_ok({"needs_rewrite": False, "patch_renames": "not-a-dict"}),
)
def test_process_patch_renames_not_dict(mock_call):
    # patch_renames is not a dict → treated as empty, no change.
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False


@mock_patch(
    _PATCH_CALL_TOOL,
    return_value=_ok(
        {"needs_rewrite": False, "patch_renames": {42: "crispen.after.X"}}
    ),
)
def test_process_patch_renames_non_string_key(mock_call):
    # Non-string key in patch_renames → filtered out.
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_patch_renames_name_invariant_filtered(mock_call):
    # LLM proposes renaming crispen.before.X → crispen.before.Y (name changed from
    # X to Y).  A file split never renames an entity — only its module path changes.
    # The rename must be filtered out, leaving no renames → triggers no-change verify.
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.before.Y"},
            }
        ),
        _ok(_VERIFY_OK),  # no-change verify confirms
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False
    assert mock_call.call_count == 2  # classify + no-change verify


@mock_patch(_PATCH_CALL_TOOL)
def test_process_string_swap_verify_accepts(mock_call):
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is True
    assert "crispen.after.X" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verify_none_accept(mock_call):
    # Verify call returns tool_input=None → accept proposed renames.
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(None),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is True
    assert "crispen.after.X" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verify_none_accept_no_splice(mock_call, tmp_path):
    # Verify returns None; function uses const ref → new_text == orig_text → no splice.
    src = (
        'TARGET = "crispen.before.X"\n\n@patch(TARGET)\ndef test_f(mock_x):\n    pass\n'
    )
    scan = str(tmp_path / "test_foo.py")
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(None),
    ]
    result, changed, cross = _process_file_source(
        src, {"crispen.before.X"}, "ctx", MagicMock(), _CFG, 1, scan_file=scan
    )
    # No splice but const should be updated via same_file_const_map.
    assert changed is True
    assert "crispen.after.X" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verify_rejected_then_accept(mock_call):
    # First verify rejects; second classify+verify is accepted.
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_REJECT),
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is True


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verify_rejected_exhausted(mock_call):
    # Verify rejects with llm_verify_retries=0 → function skipped.
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_REJECT),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG_NO_LLM_VERIFY, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verify_rejected_exhausted_escalates_to_rewrite(mock_call):
    # When llm_verify_retries>0 and rename verify retries are exhausted,
    # escalate to the full rewrite path seeded with the verifier's explanation.

    cfg = CrispenConfig(patch_update_retries=3, llm_verify_retries=1)
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),  # classify → rename
        _ok(_VERIFY_REJECT),  # verify → reject
        _ok(_CLASSIFY_RENAME),  # classify (retry) → rename again
        _ok(_VERIFY_REJECT),  # verify → reject (retries exhausted → escalate)
        _ok({"rewritten_function": _VALID_REWRITE}),  # rewrite (escalated)
        _ok(_REWRITE_VERIFY_OK),  # verify rewrite → accept
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), cfg, 3
    )
    assert changed is True
    assert mock_call.call_count == 6


@mock_patch(_PATCH_CALL_TOOL)
def test_process_per_function_different_renames(mock_call):
    """Two functions with the same @patch string can receive different renames.

    This is the forking case: test_a tests an entity that moved to mod1,
    test_b tests an entity that moved to mod2.  Each gets classified and
    renamed independently.
    """
    src = (
        '@patch("crispen.before.X")\ndef test_a(m):\n    call_a()\n\n'
        '@patch("crispen.before.X")\ndef test_b(m):\n    call_b()\n'
    )
    mock_call.side_effect = [
        # test_a: classify → rename to mod1
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.mod1.X"},
            }
        ),
        _ok(_VERIFY_OK),
        # test_b: classify → rename to mod2
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.mod2.X"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src, {"crispen.before.X"}, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is True
    assert "crispen.mod1.X" in result
    assert "crispen.mod2.X" in result
    assert mock_call.call_count == 4


@mock_patch(_PATCH_CALL_TOOL)
def test_process_per_function_both_updated(mock_call):
    """Two functions with the same @patch string both get the same rename."""
    src = (
        '@patch("crispen.before.X")\ndef test_a(m):\n    pass\n\n'
        '@patch("crispen.before.X")\ndef test_b(m):\n    pass\n'
    )
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is True
    assert result.count("crispen.after.X") == 2


@mock_patch(_PATCH_CALL_TOOL)
def test_process_acc_accumulates(mock_call):
    """_process_file_source accumulates calls, elapsed, and tokens into _acc."""
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_RENAME,
            elapsed=1.2,
            input_tokens=200,
            output_tokens=40,
        ),
        LLMCallResult(
            tool_input=_VERIFY_OK,
            elapsed=0.3,
            input_tokens=150,
            output_tokens=5,
        ),
    ]
    acc = RewriteAccumulator()
    _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1, _acc=acc
    )
    assert acc.calls == 2
    assert abs(acc.elapsed - 1.5) < 1e-9
    assert acc.input_tokens == 350
    assert acc.output_tokens == 45


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_prints_to_stderr(mock_call, capsys):
    """verbose=True emits per-call messages to stderr."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "patch_rewriter" in err
    assert "classifying" in err
    assert "verifying renames" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_detailed_timing(mock_call, capsys):
    """timing='detailed' appends elapsed/token info after each call."""
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_RENAME,
            elapsed=1.23,
            input_tokens=100,
            output_tokens=20,
        ),
        LLMCallResult(
            tool_input=_VERIFY_OK,
            elapsed=0.45,
            input_tokens=80,
            output_tokens=5,
        ),
    ]

    cfg = CrispenConfig(patch_update_retries=1, timing="detailed")
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        cfg,
        1,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "→ done" in err
    assert "1.23s" in err
    assert "0.45s" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_retry_label(mock_call, capsys):
    """Retry attempts include '(retry)' in the verbose message."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_REJECT),
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        2,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "(retry)" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_verify_accepted(mock_call, capsys):
    """verbose=True prints 'ACCEPTED' when verify succeeds."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "ACCEPTED" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_verify_rejected_prints_issue(mock_call, capsys):
    """verbose=True prints 'REJECTED' and the issue when verify rejects."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_RENAME),
        _ok({"correct": False, "issue": "wrong module path"}),
        _ok(_CLASSIFY_RENAME),
        _ok(_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        2,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "REJECTED" in err
    assert "wrong module path" in err
    assert "ACCEPTED" in err


@mock_patch(_PATCH_CALL_PR)
@mock_patch(process_basic._PATCH_MAKE_CLIENT)
@mock_patch(_PATCH_GET_KEY_PR, return_value="key")
def test_process_file_source_candidates_reject_no_change(
    mock_key, mock_client, mock_call
):
    # LLM proposes no change but candidates exist → reject and retry.
    # First classify: no rename → rejected by candidates check.
    # Second classify: correct rename in candidates → verify → accepted.
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    ctx = _FLContext(
        filepath="/repo/pkg/big.py",
        old_module="pkg.big",
        original_source="from external import A\ndef f(): A()\n",
        modified_source="from .sub_a import f\n",
        new_files={"sub_a.py": "from external import A\ndef f(): A()\n"},
        new_module_paths={"sub_a.py": "pkg.sub_a"},
        entity_to_target={"f": "sub_a.py"},
        forking_old_paths={"pkg.big.A"},
    )
    context_msg = _build_context_message([ctx])
    mock_call.side_effect = [
        # First classify: no rename (LLM says no change needed)
        LLMCallResult(
            tool_input={"needs_rewrite": False, "patch_renames": {}},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Second classify (after candidates rejection): correct rename
        LLMCallResult(
            tool_input={
                "needs_rewrite": False,
                "patch_renames": {"pkg.big.A": "pkg.sub_a.A"},
            },
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Verify rename
        LLMCallResult(
            tool_input={"correct": True, "corrections": {}, "issue": ""},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
    ]
    cg_candidates = {"test_f": {"pkg.big.A": ["pkg.sub_a.A"]}}
    new_src, changed, _ = _process_file_source(
        src,
        {"pkg.big.A"},
        context_msg,
        mock_client.return_value,
        _make_process_cfg(),
        max_attempts=2,
        cg_candidates=cg_candidates,
    )
    assert changed
    assert "pkg.sub_a.A" in new_src
    # Two classify calls + one verify call = 3
    assert mock_call.call_count == 3


@mock_patch(_PATCH_CALL_PR)
@mock_patch(process_basic._PATCH_MAKE_CLIENT)
@mock_patch(_PATCH_GET_KEY_PR, return_value="key")
def test_process_file_source_candidates_reject_verbose(
    mock_key, mock_client, mock_call, capsys
):
    # verbose=True prints 'candidates check rejected' when cand_issue fires.
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    ctx = _FLContext(
        filepath="/repo/pkg/big.py",
        old_module="pkg.big",
        original_source="from external import A\ndef f(): A()\n",
        modified_source="from .sub_a import f\n",
        new_files={"sub_a.py": "from external import A\ndef f(): A()\n"},
        new_module_paths={"sub_a.py": "pkg.sub_a"},
        entity_to_target={"f": "sub_a.py"},
        forking_old_paths={"pkg.big.A"},
    )
    context_msg = _build_context_message([ctx])
    mock_call.side_effect = [
        # First classify: no rename → rejected by candidates check.
        LLMCallResult(
            tool_input={"needs_rewrite": False, "patch_renames": {}},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Second classify: correct rename
        LLMCallResult(
            tool_input={
                "needs_rewrite": False,
                "patch_renames": {"pkg.big.A": "pkg.sub_a.A"},
            },
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Verify rename
        LLMCallResult(
            tool_input={"correct": True, "corrections": {}, "issue": ""},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
    ]
    cg_candidates = {"test_f": {"pkg.big.A": ["pkg.sub_a.A"]}}
    _process_file_source(
        src,
        {"pkg.big.A"},
        context_msg,
        mock_client.return_value,
        _make_process_cfg(),
        max_attempts=2,
        cg_candidates=cg_candidates,
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "candidates check rejected" in err


@mock_patch(_PATCH_CALL_PR)
@mock_patch(process_basic._PATCH_MAKE_CLIENT)
@mock_patch(_PATCH_GET_KEY_PR, return_value="key")
def test_process_file_source_candidates_reject_bad_rename(
    mock_key, mock_client, mock_call
):
    # LLM proposes a rename not in candidates → rejected → retry with correct answer.
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    ctx = _FLContext(
        filepath="/repo/pkg/big.py",
        old_module="pkg.big",
        original_source="from external import A\ndef f(): A()\n",
        modified_source="from .sub_a import f\n",
        new_files={
            "sub_a.py": "from external import A\ndef f(): A()\n",
            "sub_b.py": "from external import A\ndef g(): A()\n",
        },
        new_module_paths={"sub_a.py": "pkg.sub_a", "sub_b.py": "pkg.sub_b"},
        entity_to_target={"f": "sub_a.py"},
        forking_old_paths={"pkg.big.A"},
    )
    context_msg = _build_context_message([ctx])
    mock_call.side_effect = [
        # First classify: wrong rename (not in candidates)
        LLMCallResult(
            tool_input={
                "needs_rewrite": False,
                "patch_renames": {"pkg.big.A": "pkg.sub_b.A"},
            },
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Second classify: correct rename
        LLMCallResult(
            tool_input={
                "needs_rewrite": False,
                "patch_renames": {"pkg.big.A": "pkg.sub_a.A"},
            },
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Verify
        LLMCallResult(
            tool_input={"correct": True, "corrections": {}, "issue": ""},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
    ]
    cg_candidates = {"test_f": {"pkg.big.A": ["pkg.sub_a.A"]}}
    new_src, changed, _ = _process_file_source(
        src,
        {"pkg.big.A"},
        context_msg,
        mock_client.return_value,
        _make_process_cfg(),
        max_attempts=2,
        cg_candidates=cg_candidates,
    )
    assert changed
    assert "pkg.sub_a.A" in new_src
    assert mock_call.call_count == 3
