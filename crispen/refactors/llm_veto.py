from __future__ import annotations
from typing import List, Optional, Tuple
from .. import llm_client as _llm_client
from .blocks import _MODEL, _VETO_TOOL
from .models import _FunctionInfo, _SeqInfo
from .validation import _parse_validation_result


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
    return _parse_validation_result(result)  # pragma: no cover


def _llm_veto_func_match(
    client,
    seq: _SeqInfo,
    func: _FunctionInfo,
    full_source: str,
    model: str = _MODEL,
    provider: str = "anthropic",
    tool_choice_override: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """Ask the LLM whether *seq* performs the same operation as *func*'s body."""
    snippet = full_source[:4000] if len(full_source) > 4000 else full_source
    prompt = (
        "A code block in a Python file may be replaceable by a call to an existing "
        "function.\n\n"
        f"Code block (scope: {seq.scope}, lines {seq.start_line}-{seq.end_line}):\n"
        f"```python\n{seq.source.rstrip()}\n```\n\n"
        f"Existing function '{func.name}':\n"
        f"```python\n{func.source.rstrip()}\n```\n\n"
        f"File source:\n```python\n{snippet}\n```\n\n"
        "Does this code block perform the same semantic operation as the function "
        "body, such that it could be replaced by a call to the function? "
        "Use the evaluate_duplicate tool to answer."
    )
    result = _llm_client.call_with_tool(
        client,
        provider,
        model,
        256,
        _VETO_TOOL,
        "evaluate_duplicate",
        [{"role": "user", "content": prompt}],
        caller="DuplicateExtractor",
        tool_choice_override=tool_choice_override,
    )
    return _parse_validation_result(result)  # pragma: no cover
