from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import run_engine
from crispen.file_limiter.runner import FileLimiterResult
from ._block_1111 import _FL_PATCH


def test_file_limiter_no_messages_no_new_files(tmp_path):
    """FileLimiter returns empty messages + no new files → no output, no writes."""
    f = tmp_path / "big.py"
    original = "".join(f"var_{i} = {i}\n" for i in range(10))
    f.write_text(original, encoding="utf-8")
    no_op_result = FileLimiterResult(
        original_source=original,
        new_files={},
        messages=[],
        abort=False,
    )
    with patch(_FL_PATCH, return_value=no_op_result):
        msgs = list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
            )
        )
    assert not any("FileLimiter" in m for m in msgs)
