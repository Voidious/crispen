from __future__ import annotations
from typing import Optional
from .. import llm_client as _llm_client
from .code_blocks import _CALL_GEN_TOOL, _MODEL
from .collector import _FunctionInfo, _SeqInfo


def _generate_no_arg_call(seq: _SeqInfo, func: _FunctionInfo) -> str:
    """Algorithmically generate a no-argument call to *func*, preserving indentation."""
    first_line = seq.source.splitlines()[0]
    indent = first_line[: len(first_line) - len(first_line.lstrip())]
    return indent + func.name + "()\n"


def _llm_generate_call(
    client,
    seq: _SeqInfo,
    func: _FunctionInfo,
    full_source: str,
    model: str = _MODEL,
    provider: str = "anthropic",
    tool_choice_override: Optional[str] = None,
) -> Optional[str]:
    """Ask the LLM to generate a call expression replacing *seq* with *func*."""
    snippet = full_source[:4000] if len(full_source) > 4000 else full_source
    prompt = (
        f"Replace this code block with a call to the existing function"
        f" '{func.name}'.\n\n"
        f"Code block (scope: {seq.scope}, lines {seq.start_line}-{seq.end_line}):\n"
        f"```python\n{seq.source.rstrip()}\n```\n\n"
        f"Function '{func.name}':\n"
        f"```python\n{func.source.rstrip()}\n```\n\n"
        f"File source:\n```python\n{snippet}\n```\n\n"
        "Generate a replacement that preserves the original indentation and ends "
        "with a newline. Pass the replacement to the generate_call tool."
    )
    result = _llm_client.call_with_tool(
        client,
        provider,
        model,
        256,
        _CALL_GEN_TOOL,
        "generate_call",
        [{"role": "user", "content": prompt}],
        caller="DuplicateExtractor",
        tool_choice_override=tool_choice_override,
    )
    if result is not None:
        return result["replacement"]
    return None  # pragma: no cover
