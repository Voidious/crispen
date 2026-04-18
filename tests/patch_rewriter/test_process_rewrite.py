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
from .helpers import (
    _CFG,
    _CFG_NO_LLM_VERIFY,
    _CLASSIFY_REWRITE,
    _FORKING_PATHS,
    _PATCH_CALL_PR,
    _PATCH_CALL_TOOL,
    _PATCH_GET_KEY_PR,
    _REWRITE_VERIFY_OK,
    _REWRITE_VERIFY_REJECT,
    _SRC_WITH_PATCH,
    _VALID_REWRITE,
    _make_process_cfg,
    _ok,
    _truncated_ok,
)
from . import helpers


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_success(mock_call):
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is True
    assert "crispen.after.X" in result
    assert "crispen.after.Y" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_tool_none(mock_call):
    # Rewrite call returns tool_input=None → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok(None),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_empty_text(mock_call):
    # Rewrite returns empty string → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": ""}),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_non_string(mock_call):
    # Rewrite returns non-string value → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": 42}),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_compile_error_retry(mock_call):
    # First rewrite has syntax error; second is valid.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": "def f(:\n    pass\n"}),  # invalid
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is True
    assert "crispen.after.X" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_compile_error_exhausted(mock_call):
    # Both rewrite attempts fail to compile → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": "def f(:\n    pass\n"}),  # invalid
        _ok({"rewritten_function": "def f(:\n    pass\n"}),  # still invalid
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_verify_none_accept(mock_call):
    # Verify returns tool_input=None → accept the rewrite.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(None),  # verify returns None → accept
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is True
    assert "crispen.after.X" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_verify_truncated_reject(mock_call):
    # Rewrite verify truncated → treated as rejection, rewrite not accepted.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _truncated_ok(),  # verify truncated → reject
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG_NO_LLM_VERIFY, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False
    assert mock_call.call_count == 3


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_verify_rejected_then_accept(mock_call):
    # Verify rejects first rewrite; second rewrite+verify is accepted.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_REJECT),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is True


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_verify_rejected_exhausted(mock_call):
    # Verify rejects with llm_verify_retries=0 → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_REJECT),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG_NO_LLM_VERIFY, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_acc_rewrite_accumulates(mock_call):
    """Full rewrite path accumulates classify, rewrite, and verify calls."""
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_REWRITE,
            elapsed=0.5,
            input_tokens=100,
            output_tokens=10,
        ),
        LLMCallResult(
            tool_input={"rewritten_function": _VALID_REWRITE},
            elapsed=1.5,
            input_tokens=300,
            output_tokens=60,
        ),
        LLMCallResult(
            tool_input=_REWRITE_VERIFY_OK,
            elapsed=0.2,
            input_tokens=80,
            output_tokens=5,
        ),
    ]
    acc = RewriteAccumulator()
    _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1, _acc=acc
    )
    assert acc.calls == 3
    assert abs(acc.elapsed - 2.2) < 1e-9


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_rewrite_path(mock_call, capsys):
    """verbose=True prints 'rewriting', 'verifying rewrite', and 'rewrote'."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
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
    assert "rewriting" in err
    assert "verifying rewrite" in err
    assert "rewrote" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_rewrite_verify_rejected(mock_call, capsys):
    """verbose=True prints 'REJECTED' and issue when rewrite verify fails."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_REJECT),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
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
    assert "wrong mock setup" in err
    assert "ACCEPTED" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_rewrite_compile_retry(mock_call, capsys):
    """verbose=True prints '(retry)' when rewrite compile fails."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": "def f(:\n    pass\n"}),  # invalid
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    cfg = CrispenConfig(patch_update_retries=1, timing="detailed")
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
    assert "rewriting" in err
    assert "(retry)" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_rewrite_restores_unchanged_const_ref(mock_call):
    """After full rewrite, @patch("value") left unchanged → reverted to @patch(NAME)."""
    src = (
        'STABLE = "pkg.stable.X"\n'
        'TARGET = "pkg.big.A"\n\n'
        "@patch(STABLE)\n"
        "@patch(TARGET)\n"
        "def test_f(mock_stable, mock_target):\n"
        "    pass\n"
    )
    # LLM updates TARGET but leaves STABLE's substituted literal unchanged.
    rewritten = (
        '@patch("pkg.stable.X")\n'
        '@patch("pkg.sub_a.A")\n'
        "def test_f(mock_stable, mock_target):\n"
        "    pass\n"
    )
    mock_call.side_effect = [
        _ok({"needs_rewrite": True}),
        _ok({"rewritten_function": rewritten}),
        _ok({"correct": True, "issue": ""}),
    ]
    result, changed, _ = _process_file_source(
        src,
        {"pkg.big.A"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file="tests/test_foo.py",
    )
    assert changed is True
    # STABLE decorator reverted to named constant form.
    assert "@patch(STABLE)" in result
    assert '@patch("pkg.stable.X")' not in result
    # TARGET decorator keeps the LLM's updated literal value.
    assert '@patch("pkg.sub_a.A")' in result


@mock_patch(_PATCH_CALL_PR)
@mock_patch(helpers._PATCH_MAKE_CLIENT)
@mock_patch(_PATCH_GET_KEY_PR, return_value="key")
def test_process_file_source_rewrite_candidates_reject_and_retry(
    mock_key, mock_client, mock_call
):
    # Rewrite returns old path still present → rewrite candidates check rejects
    # without calling verify → retry; second rewrite uses valid candidate → accepted.
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    complex_logic()\n'
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
    bad_rewrite = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    complex_logic()\n'
    good_rewrite = '@patch("pkg.sub_a.A")\ndef test_f(mock_a):\n    complex_logic()\n'
    mock_call.side_effect = [
        # classify → needs rewrite
        LLMCallResult(
            tool_input={"needs_rewrite": True},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # rewrite 1: old path still present → rejected by _rewrite_candidates_check
        LLMCallResult(
            tool_input={"rewritten_function": bad_rewrite},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # rewrite 2: valid candidate
        LLMCallResult(
            tool_input={"rewritten_function": good_rewrite},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # verify
        LLMCallResult(
            tool_input={"correct": True, "issue": ""},
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
    assert mock_call.call_count == 4  # classify + bad_rw + good_rw + verify


@mock_patch(_PATCH_CALL_PR)
@mock_patch(helpers._PATCH_MAKE_CLIENT)
@mock_patch(_PATCH_GET_KEY_PR, return_value="key")
def test_process_file_source_candidates_all_retries_escalates_to_rewrite(
    mock_key, mock_client, mock_call, capsys
):
    # All classify retries exhausted with persistent candidates check rejections →
    # escalate to full rewrite rather than silently leaving the test broken.
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
    good_rewrite = '@patch("pkg.sub_a.A")\ndef test_f(mock_a):\n    pass\n'
    mock_call.side_effect = [
        # First classify: no rename → rejected by candidates check (not last attempt).
        LLMCallResult(
            tool_input={"needs_rewrite": False, "patch_renames": {}},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Second classify: still no rename → last attempt → escalate to rewrite.
        LLMCallResult(
            tool_input={"needs_rewrite": False, "patch_renames": {}},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Rewrite (escalated from candidates check failure):
        LLMCallResult(
            tool_input={"rewritten_function": good_rewrite},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Rewrite verify:
        LLMCallResult(
            tool_input={"correct": True, "issue": ""},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
    ]
    # Two candidates → ambiguous → LLM keeps returning no_change.
    cg_candidates = {"test_f": {"pkg.big.A": ["pkg.sub_a.A", "pkg.sub_b.A"]}}
    new_src, changed, _ = _process_file_source(
        src,
        {"pkg.big.A"},
        context_msg,
        mock_client.return_value,
        _make_process_cfg(),
        max_attempts=2,
        cg_candidates=cg_candidates,
        verbose=True,
    )
    assert changed
    assert "pkg.sub_a.A" in new_src
    assert mock_call.call_count == 4  # classify x2 + rewrite + verify
    err = capsys.readouterr().err
    assert "candidates check retries exhausted" in err
