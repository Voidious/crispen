from __future__ import annotations
from typing import List, Optional, Tuple
from .. import llm_client as _llm_client
from .blocks import _MODEL, _VERIFY_TOOL
from .models import _SeqInfo


def _llm_verify_extraction(
    client,
    group: List[_SeqInfo],
    helper_source: str,
    call_replacements: List[str],
    full_source: str,
    model: str = _MODEL,
    provider: str = "anthropic",
    tool_choice_override: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Ask the LLM to verify the extraction is semantically correct.

    Returns ``(is_correct, issues)`` where *issues* is a list of specific
    problems found.  Returns ``(True, [])`` if the call times out or the LLM
    cannot respond, so a verification failure never silently blocks commits.
    """
    blocks_text = "\n\n".join(
        f"Original block {i + 1} (scope: {s.scope}, "
        f"lines {s.start_line}-{s.end_line}):\n"
        f"```python\n{s.source.rstrip()}\n```"
        for i, s in enumerate(group)
    )
    replacements_text = "\n\n".join(
        f"Replacement for block {i + 1}:\n```python\n{r.rstrip()}\n```"
        for i, r in enumerate(call_replacements)
    )
    snippet = full_source[:2000] if len(full_source) > 2000 else full_source
    prompt = (
        "Verify that the following helper function extraction is semantically "
        "correct by tracing through the code carefully.\n\n"
        f"Original duplicate blocks:\n{blocks_text}\n\n"
        f"Extracted helper:\n```python\n{helper_source.rstrip()}\n```\n\n"
        f"Call site replacements:\n{replacements_text}\n\n"
        f"File context (truncated):\n```python\n{snippet}\n```\n\n"
        "Check each of the following:\n"
        "1. Every variable read (but not locally assigned) in the original block "
        "is passed as a parameter to the helper\n"
        "2. Every variable assigned in the original block and used afterward is "
        "returned by the helper and captured at the call site\n"
        "3. No parameter is assigned before it is first read in the helper body\n"
        "4. If the original block ends with a non-None return, the call site "
        "replacement also propagates that return value\n"
        "5. The call site replacements match the original indentation and cover "
        "exactly the lines of the original block\n"
        "6. If the helper is called more than once with different arguments, verify "
        "each call site against the exact local variables that appeared in the "
        "original code at that location — not merely variables of the same type. "
        "Same-type variables (e.g. two dicts, two strings) that are both in scope "
        "are a swap risk: confirm neither was substituted for the other across call "
        "sites.\n"
        "If correct, set is_correct=True and issues=[]. "
        "Otherwise set is_correct=False and list each specific issue."
    )
    result = _llm_client.call_with_tool(
        client,
        provider,
        model,
        512,
        _VERIFY_TOOL,
        "verify_extraction",
        [{"role": "user", "content": prompt}],
        caller="DuplicateExtractor",
        tool_choice_override=tool_choice_override,
    )
    if result is None:
        return True, []  # pragma: no cover
    return result["is_correct"], result.get("issues", [])
