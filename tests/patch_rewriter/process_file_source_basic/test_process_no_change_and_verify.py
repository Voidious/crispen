from __future__ import annotations
from unittest.mock import MagicMock, patch as mock_patch
from crispen.config import CrispenConfig
from crispen.llm_client import LLMCallResult
from crispen.patch_rewriter import RewriteAccumulator, _compiles, _process_file_source
from ..helpers import (
    _CFG,
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
from ..test_process_file_source_rewrite_path import _VALID_REWRITE


def test_compiles_valid():
    assert _compiles("x = 1\n") is True


def test_compiles_invalid():
    assert _compiles("def f(:\n    pass\n") is False


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
