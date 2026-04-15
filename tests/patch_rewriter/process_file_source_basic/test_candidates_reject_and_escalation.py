from __future__ import annotations
from unittest.mock import patch as mock_patch
from crispen.llm_client import LLMCallResult
from crispen.patch_rewriter import (
    _FLContext,
    _build_context_message,
    _process_file_source,
)
from ..helpers import _PATCH_CALL_PR, _PATCH_GET_KEY_PR, _make_process_cfg
from .. import helpers


@mock_patch(_PATCH_CALL_PR)
@mock_patch(helpers._PATCH_MAKE_CLIENT)
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
@mock_patch(helpers._PATCH_MAKE_CLIENT)
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
@mock_patch(helpers._PATCH_MAKE_CLIENT)
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
