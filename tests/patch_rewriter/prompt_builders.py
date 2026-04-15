from __future__ import annotations
from crispen.patch_rewriter import _FLContext, _build_context_message
from .context_builders import _make_fl_ctx


def _ctx_msg() -> str:
    return _build_context_message([_make_fl_ctx()])


def _make_fl_ctx_simple():
    """Minimal FLContext for prompt builder tests."""
    return _FLContext(
        filepath="/repo/pkg/big.py",
        old_module="pkg.big",
        original_source="from external import A\ndef f(): A()\n",
        modified_source="from .sub_a import f\n",
        new_files={"sub_a.py": "from external import A\ndef f(): A()\n"},
        new_module_paths={"sub_a.py": "pkg.sub_a"},
        entity_to_target={"f": "sub_a.py"},
        forking_old_paths={"pkg.big.A"},
    )
