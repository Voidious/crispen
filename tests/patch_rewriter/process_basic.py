from __future__ import annotations
from crispen.config import CrispenConfig


_PATCH_MAKE_CLIENT = "crispen.patch_rewriter.make_client"
_PATCH_GET_KEY_PR = "crispen.patch_rewriter.get_api_key"
_PATCH_CALL_PR = "crispen.patch_rewriter.call_with_tool"


def _make_process_cfg():
    return CrispenConfig(patch_update_retries=1, llm_verify_retries=0)
