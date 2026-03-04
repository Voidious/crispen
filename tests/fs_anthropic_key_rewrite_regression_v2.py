from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import FunctionSplitter


def _splitter_rewrite_is_none_with_anthropic_key(
    src: str,
    *,
    ranges: list[tuple[int, int]] | None = None,
    verbose: bool = False,
    max_lines: int = 10,
) -> None:
    if ranges is None:
        ranges = [(1, 1000)]

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            ranges, source=src, verbose=verbose, max_lines=max_lines
        )

    assert splitter.get_rewritten_source() is None
