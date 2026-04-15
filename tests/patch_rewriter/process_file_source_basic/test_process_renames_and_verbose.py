from __future__ import annotations
from unittest.mock import MagicMock, patch as mock_patch
from crispen.config import CrispenConfig
from crispen.llm_client import LLMCallResult
from crispen.patch_rewriter import RewriteAccumulator, _process_file_source
from ..helpers import (
    _CFG,
    _CFG_NO_LLM_VERIFY,
    _CLASSIFY_RENAME,
    _FORKING_PATHS,
    _PATCH_CALL_TOOL,
    _REWRITE_VERIFY_OK,
    _SRC_WITH_PATCH,
    _VERIFY_OK,
    _VERIFY_REJECT,
    _ok,
)
from ..test_process_file_source_rewrite_path import _VALID_REWRITE


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
