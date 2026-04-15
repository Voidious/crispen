from __future__ import annotations
from crispen.config import CrispenConfig
from crispen.llm_client import LLMCallResult
from crispen.patch_rewriter import _FLContext


def _ok(tool_input=None) -> LLMCallResult:
    return LLMCallResult(
        tool_input=tool_input, elapsed=0.0, input_tokens=0, output_tokens=0
    )


def _make_fl_ctx(**kwargs) -> _FLContext:
    defaults = dict(
        filepath="/proj/pkg/big.py",
        old_module="pkg.big",
        original_source="class A: pass\nclass B: pass\n",
        modified_source="from .sub_a import A\nfrom .sub_b import B\n",
        new_files={"sub_a.py": "class A: pass\n", "sub_b.py": "class B: pass\n"},
        new_module_paths={"sub_a.py": "pkg.sub_a", "sub_b.py": "pkg.sub_b"},
        entity_to_target={"A": "sub_a.py", "B": "sub_b.py"},
        forking_old_paths={"pkg.big.A", "pkg.big.B"},
    )
    defaults.update(kwargs)
    return _FLContext(**defaults)


_CFG = CrispenConfig(patch_update_retries=1)
_CFG_NO_LLM_VERIFY = CrispenConfig(patch_update_retries=1, llm_verify_retries=0)
_FORKING_PATHS = {"crispen.before.X"}
_SRC_WITH_PATCH = '@patch("crispen.before.X")\ndef test_f(mock_x):\n    pass\n'

_PATCH_GET_KEY = "crispen.patch_rewriter.get_api_key"
_PATCH_MAKE_CLIENT = "crispen.patch_rewriter.make_client"
_PATCH_CALL_TOOL = "crispen.patch_rewriter.call_with_tool"

# Shorthand classify tool_inputs.
_CLASSIFY_RENAME = {
    "needs_rewrite": False,
    "patch_renames": {"crispen.before.X": "crispen.after.X"},
}
_CLASSIFY_NO_CHANGE = {"needs_rewrite": False, "patch_renames": {}}
_CLASSIFY_REWRITE = {"needs_rewrite": True}
_VERIFY_OK = {"correct": True, "issue": ""}
_VERIFY_REJECT = {"correct": False, "issue": "wrong path"}
_VERIFY_REJECT_WITH_CORRECTIONS = {
    "correct": False,
    "issue": "wrong path",
    "corrections": {"crispen.before.X": "crispen.after.X"},
}
_REWRITE_VERIFY_OK = {"correct": True, "issue": ""}
_REWRITE_VERIFY_REJECT = {"correct": False, "issue": "wrong mock setup"}
