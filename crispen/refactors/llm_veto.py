from __future__ import annotations
from typing import List, Optional, Tuple
from .. import llm_client as _llm_client
from .blocks import _MODEL, _VETO_TOOL
from .sequence_info import _SeqInfo


def _llm_veto(
    client,
    group: List[_SeqInfo],
    model: str = _MODEL,
    provider: str = "anthropic",
    tool_choice_override: Optional[str] = None,
) -> Tuple[bool, str, str]:
    blocks_text = "\n\n".join(
        f"Block {i + 1} (scope: {s.scope}, lines {s.start_line}-{s.end_line}):\n"
        f"```python\n{s.source.rstrip()}\n```"
        for i, s in enumerate(group)
    )
    prompt = (
        f"Here are {len(group)} structurally similar code blocks from the same "
        f"Python file:\n\n{blocks_text}\n\n"
        "Do these blocks represent the same semantic operation such that extracting "
        "a shared helper function would improve clarity? Or are they coincidentally "
        "similar but conceptually distinct?\n\n"
        "If you accept (is_valid_duplicate=True), also fill in extraction_notes "
        "with any potential pitfalls the extraction step should watch out for — "
        "e.g., tricky variable scoping, mutable arguments, subtle differences in "
        "variable names between blocks, or return-value handling edge cases."
    )
    result = _llm_client.call_with_tool(
        client,
        provider,
        model,
        384,
        _VETO_TOOL,
        "evaluate_duplicate",
        [{"role": "user", "content": prompt}],
        caller="DuplicateExtractor",
        tool_choice_override=tool_choice_override,
    )
    if result is not None:
        return (
            result["is_valid_duplicate"],
            result.get("reason", ""),
            result.get("extraction_notes", ""),
        )
    return False, "no tool response", ""  # pragma: no cover
