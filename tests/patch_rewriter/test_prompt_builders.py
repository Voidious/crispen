from __future__ import annotations
from crispen.patch_rewriter import (
    _CG_CANDIDATES_LLM_THRESHOLD,
    _build_classify_prompt,
    _build_context_message,
    _build_func_verify_prompt,
    _build_no_change_verify_prompt,
    _build_rewrite_func_prompt,
    _build_rewrite_verify_prompt,
)
from .helpers import _make_fl_ctx, _make_fl_ctx_simple
from .test_patch_lookup_and_rename_guard import _make_ctx_with_ext_imports


def _ctx_msg() -> str:
    return _build_context_message([_make_fl_ctx()])


def test_build_classify_prompt_no_prev():
    prompt = _build_classify_prompt(
        _ctx_msg(), "def test_f(): pass", ["crispen.before.X"]
    )
    assert "crispen.before.X" in prompt
    assert "Previous attempt was rejected" not in prompt
    assert "patch_renames" in prompt
    assert "Entity migration (quick reference)" in prompt


def test_build_classify_prompt_with_prev():
    prompt = _build_classify_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        ["crispen.before.X"],
        prev_issue="wrong module",
        prev_proposed="{'crispen.before.X': 'bad.mod.X'}",
    )
    assert "Previous attempt was rejected" in prompt
    assert "wrong module" in prompt
    assert "bad.mod.X" in prompt


def test_build_classify_prompt_multiple_paths():
    prompt = _build_classify_prompt(
        _ctx_msg(), "def test_f(): pass", ["crispen.before.X", "crispen.before.Y"]
    )
    assert "crispen.before.X" in prompt
    assert "crispen.before.Y" in prompt


def test_build_classify_prompt_with_lookup():
    # When the context has a patch target lookup, it appears in the classify prompt
    # and the simplified lookup-based algorithm is used.
    ctx_msg = _build_context_message([_make_ctx_with_ext_imports()])
    prompt = _build_classify_prompt(
        ctx_msg, "def test_f(): pass", ["pkg.big.call_with_tool"]
    )
    assert "Patch target lookup" in prompt
    assert "call_with_tool" in prompt
    assert "pkg.llm_planning" in prompt
    assert "patch_renames" in prompt
    assert "Entity migration (quick reference)" in prompt


def test_build_classify_prompt_with_stable_paths():
    # stable_patch_paths appear in a separate "already correct" section and
    # the forking path remains in the "needs updating" section.
    prompt = _build_classify_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        ["crispen.before.X"],
        stable_patch_paths=["crispen.after.Y"],
    )
    assert "crispen.before.X" in prompt
    assert "crispen.after.Y" in prompt
    assert "already correct" in prompt
    assert "do not modify" in prompt


def test_build_func_verify_prompt_basic():
    prompt = _build_func_verify_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        {"crispen.before.X": "crispen.after.X"},
    )
    assert "crispen.before.X" in prompt
    assert "crispen.after.X" in prompt
    assert "correct" in prompt


def test_build_func_verify_prompt_multiple_renames():
    prompt = _build_func_verify_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        {"crispen.before.X": "crispen.after.X", "crispen.before.Y": "crispen.after.Y"},
    )
    assert "crispen.before.X" in prompt
    assert "crispen.before.Y" in prompt
    assert "crispen.after.X" in prompt
    assert "crispen.after.Y" in prompt


def test_build_func_verify_prompt_includes_patch_lookup():
    # When the context has a patch lookup section, it should be repeated near
    # the verify instructions.
    ctx_msg = _build_context_message([_make_ctx_with_ext_imports()])
    prompt = _build_func_verify_prompt(
        ctx_msg,
        "def test_f(): pass",
        {"pkg.old.call_with_tool": "pkg.llm_planning.call_with_tool"},
    )
    assert "Patch target lookup" in prompt


def test_build_no_change_verify_prompt_includes_migration_reminder():
    # Prompt built with a context that has migration entries should include
    # the migration quick-reference block near the instructions.
    prompt = _build_no_change_verify_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        ["crispen.before.X"],
    )
    assert "crispen.before.X" in prompt
    assert "Entity migration" in prompt


def test_build_no_change_verify_prompt_includes_patch_lookup():
    # When the context has a patch lookup section, it should be repeated near
    # the verify instructions so the model doesn't have to scan the full context.
    ctx_msg = _build_context_message([_make_ctx_with_ext_imports()])
    prompt = _build_no_change_verify_prompt(
        ctx_msg,
        "def test_f(): pass",
        ["pkg.old.call_with_tool"],
    )
    assert "Patch target lookup" in prompt


def test_build_no_change_verify_prompt_with_stable_paths():
    # stable_patch_paths appear in a separate "already correct" section and
    # the instruction tells the verifier not to include them in corrections.
    prompt = _build_no_change_verify_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        ["crispen.before.X"],
        stable_patch_paths=["crispen.after.Y"],
    )
    assert "crispen.before.X" in prompt
    assert "crispen.after.Y" in prompt
    assert "already correct" in prompt
    assert "do not include" in prompt


def test_build_rewrite_func_prompt_no_error():
    prompt = _build_rewrite_func_prompt(
        _ctx_msg(), "def test_f(): pass", ["crispen.before.X"]
    )
    assert "crispen.before.X" in prompt
    assert "Previous rewrite" not in prompt
    assert "Rewrite the complete function" in prompt


def test_build_rewrite_func_prompt_with_error():
    prompt = _build_rewrite_func_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        ["crispen.before.X"],
        prev_error="SyntaxError on line 3",
    )
    assert "Previous rewrite was invalid" in prompt
    assert "SyntaxError on line 3" in prompt


def test_build_rewrite_func_prompt_with_stable_paths():
    # stable_patch_paths appear in a separate "already correct" section and
    # the instruction tells the LLM not to modify them.
    prompt = _build_rewrite_func_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        ["crispen.before.X"],
        stable_patch_paths=["crispen.after.Y"],
    )
    assert "crispen.before.X" in prompt
    assert "crispen.after.Y" in prompt
    assert "already correct" in prompt
    assert "do not modify" in prompt.lower()


def test_build_rewrite_verify_prompt_basic():
    prompt = _build_rewrite_verify_prompt(
        _ctx_msg(),
        "def test_f(): pass",
        '@patch("crispen.after.X")\ndef test_f(mock_x):\n    pass\n',
    )
    assert "Original test function" in prompt
    assert "Rewritten test function" in prompt
    assert "crispen.after.X" in prompt
    assert "correct" in prompt


def test_build_classify_prompt_with_candidates():
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    prompt = _build_classify_prompt(
        context_msg,
        "def test_f(): pass\n",
        ["pkg.big.A"],
        candidates_per_path={"pkg.big.A": ["pkg.sub_a.A", "pkg.sub_b.A"]},
    )
    assert "Call-graph candidate paths" in prompt
    assert "pkg.sub_a.A" in prompt
    assert "pkg.sub_b.A" in prompt


def test_build_classify_prompt_candidates_above_threshold():
    # Candidates count > threshold → section not included.
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    many_cands = [f"pkg.sub_{i}.A" for i in range(_CG_CANDIDATES_LLM_THRESHOLD + 1)]
    prompt = _build_classify_prompt(
        context_msg,
        "def test_f(): pass\n",
        ["pkg.big.A"],
        candidates_per_path={"pkg.big.A": many_cands},
    )
    assert "Call-graph candidate paths" not in prompt


def test_build_func_verify_prompt_with_candidates():
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    prompt = _build_func_verify_prompt(
        context_msg,
        "def test_f(): pass\n",
        {"pkg.big.A": "pkg.sub_a.A"},
        candidates_per_path={"pkg.big.A": ["pkg.sub_a.A", "pkg.sub_b.A"]},
    )
    assert "Call-graph candidate paths" in prompt
    assert "pkg.sub_a.A" in prompt


def test_build_func_verify_prompt_candidates_above_threshold():
    # All candidate lists exceed the threshold → section not included.
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    many_cands = [f"pkg.sub_{i}.A" for i in range(_CG_CANDIDATES_LLM_THRESHOLD + 1)]
    prompt = _build_func_verify_prompt(
        context_msg,
        "def test_f(): pass\n",
        {"pkg.big.A": "pkg.sub_a.A"},
        candidates_per_path={"pkg.big.A": many_cands},
    )
    assert "Call-graph candidate paths" not in prompt


def test_build_no_change_verify_prompt_with_candidates():
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    prompt = _build_no_change_verify_prompt(
        context_msg,
        "def test_f(): pass\n",
        ["pkg.big.A"],
        candidates_per_path={"pkg.big.A": ["pkg.sub_a.A"]},
    )
    assert "Call-graph candidate paths" in prompt
    assert "pkg.sub_a.A" in prompt


def test_build_no_change_verify_prompt_candidates_above_threshold():
    # All candidate lists exceed the threshold → section not included.
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    many_cands = [f"pkg.sub_{i}.A" for i in range(_CG_CANDIDATES_LLM_THRESHOLD + 1)]
    prompt = _build_no_change_verify_prompt(
        context_msg,
        "def test_f(): pass\n",
        ["pkg.big.A"],
        candidates_per_path={"pkg.big.A": many_cands},
    )
    assert "Call-graph candidate paths" not in prompt


def test_build_rewrite_func_prompt_with_candidates():
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    prompt = _build_rewrite_func_prompt(
        context_msg,
        "def test_f(): pass\n",
        ["pkg.big.A"],
        candidates_per_path={"pkg.big.A": ["pkg.sub_a.A", "pkg.helpers.A"]},
    )
    assert "Call-graph candidate paths" in prompt
    assert "pkg.sub_a.A" in prompt
    assert "pkg.helpers.A" in prompt


def test_build_rewrite_func_prompt_candidates_above_threshold():
    # All candidate lists exceed the threshold → section not included.
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    many_cands = [f"pkg.sub_{i}.A" for i in range(_CG_CANDIDATES_LLM_THRESHOLD + 1)]
    prompt = _build_rewrite_func_prompt(
        context_msg,
        "def test_f(): pass\n",
        ["pkg.big.A"],
        candidates_per_path={"pkg.big.A": many_cands},
    )
    assert "Call-graph candidate paths" not in prompt


def test_build_rewrite_verify_prompt_with_candidates():
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    prompt = _build_rewrite_verify_prompt(
        context_msg,
        "def test_f(): pass\n",
        "def test_f(): pass\n",
        candidates_per_path={"pkg.big.A": ["pkg.sub_a.A", "pkg.sub_b.A"]},
    )
    assert "Call-graph candidate paths" in prompt
    assert "pkg.sub_a.A" in prompt


def test_build_rewrite_verify_prompt_candidates_above_threshold():
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    many_cands = [f"pkg.sub_{i}.A" for i in range(_CG_CANDIDATES_LLM_THRESHOLD + 1)]
    prompt = _build_rewrite_verify_prompt(
        context_msg,
        "def test_f(): pass\n",
        "def test_f(): pass\n",
        candidates_per_path={"pkg.big.A": many_cands},
    )
    assert "Call-graph candidate paths" not in prompt


def test_build_rewrite_verify_prompt_no_candidates():
    ctx = _make_fl_ctx_simple()
    context_msg = _build_context_message([ctx])
    prompt = _build_rewrite_verify_prompt(
        context_msg,
        "def test_f(): pass\n",
        "def test_f(): pass\n",
    )
    assert "Call-graph candidate paths" not in prompt
    assert "Verify that the rewrite is correct" in prompt
