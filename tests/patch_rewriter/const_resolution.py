from __future__ import annotations
from crispen.patch_rewriter import _ConstRef


def _make_ref(const_name: str, resolved_value: str) -> _ConstRef:
    return _ConstRef(
        const_name=const_name,
        source_file="/proj/tests/helpers.py",
        resolved_value=resolved_value,
        patch_dec_idx=0,
    )


_SRC_WITH_CONST = (
    'TARGET = "crispen.before.X"\n\n'
    "@patch(TARGET)\n"
    "def test_f(mock_x):\n"
    "    pass\n"
)
