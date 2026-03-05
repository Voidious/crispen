from __future__ import annotations
from typing import List, Optional, Tuple
from .. import llm_client as _llm_client
from .block_helpers import (
    _CALL_GEN_TOOL,
    _EXTRACT_TOOL,
    _MODEL,
    _VERIFY_TOOL,
    _VETO_TOOL,
)
from .info_objects import _FunctionInfo, _SeqInfo


def _process_llm_result(result):
    if result is not None:
        return (
            result["is_valid_duplicate"],
            result.get("reason", ""),
            result.get("extraction_notes", ""),
        )
    return False, "no tool response", ""  # pragma: no cover


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
    return _process_llm_result(result)


def _llm_extract(
    client,
    group: List[_SeqInfo],
    full_source: str,
    escaping_vars: frozenset = frozenset(),
    used_names: frozenset = frozenset(),
    model: str = _MODEL,
    helper_docstrings: bool = True,
    provider: str = "anthropic",
    veto_notes: str = "",
    prev_failures: List[str] = [],
    prev_output: Optional[dict] = None,
    tool_choice_override: Optional[str] = None,
) -> Optional[dict]:
    src_lines = full_source.splitlines(keepends=True)
    block_entries = []
    for i, s in enumerate(group):
        entry = (
            f"Block {i + 1} (scope: {s.scope}, lines {s.start_line}-{s.end_line}):\n"
            f"```python\n{s.source.rstrip()}\n```"
        )
        next_idx = s.end_line  # 0-based index of the first line after the block
        if next_idx < len(src_lines):
            next_line = src_lines[next_idx].rstrip()
            if next_line.strip():
                entry += (
                    f"\nLine immediately after this block"
                    f" (must NOT appear in the replacement): `{next_line}`"
                )
        block_entries.append(entry)
    blocks_text = "\n\n".join(block_entries)
    snippet = full_source[:4000] if len(full_source) > 4000 else full_source
    escaping_note = ""
    if escaping_vars:
        vars_str = ", ".join(sorted(escaping_vars))
        escaping_note = (
            f"\n\nThe following variables are assigned within the duplicate block "
            f"and referenced by code that immediately follows the block at one or "
            f"more call sites: {vars_str}. The helper function must return these "
            f"variables. At call sites where the return value is needed, capture it; "
            f"at call sites where it is not needed, discard the return value."
        )
    used_names_note = ""
    if used_names:
        names_str = ", ".join(sorted(used_names))
        used_names_note = (
            f"\n\nThe following function names are already defined in this file "
            f"or reserved by a previous extraction: {names_str}. "
            f"Do not use any of these names for the helper function."
        )
    docstring_note = (
        ""
        if helper_docstrings
        else "\n\nDo not include a docstring in the helper function."
    )
    veto_notes_note = ""
    if veto_notes:
        veto_notes_note = (
            f"\n\nNotes from code review (watch out for these pitfalls): "
            f"{veto_notes[:500]}"
        )
    failures_note = ""
    if prev_failures:
        failures_str = "\n".join(f"- {f}" for f in prev_failures)
        if prev_output is not None:
            prior_helper = prev_output.get("helper_source", "")
            prior_repls = prev_output.get("call_site_replacements", [])
            repls_text = "\n".join(
                f"  [{i + 1}] {r!r}" for i, r in enumerate(prior_repls)
            )
            failures_note = (
                f"\n\nThe previous extraction attempt produced:\n\n"
                f"helper_source:\n```python\n{prior_helper}```\n\n"
                f"call_site_replacements:\n{repls_text}\n\n"
                f"But failed verification with these issues:\n{failures_str}\n\n"
                f"Please correct these issues in your new attempt."
            )
        else:
            failures_note = (
                f"\n\nThe previous extraction attempt failed. Please correct these "
                f"issues:\n{failures_str}"
            )
    class_scopes = {s.class_scope for s in group}
    all_same_class = len(class_scopes) == 1 and None not in class_scopes
    if all_same_class:
        staticmethod_instruction = (
            "If all call sites are inside the same class, use a @staticmethod. "
        )
    else:
        staticmethod_instruction = (
            "Use module_level placement — call sites span different classes or scopes. "
        )
    prompt = (
        "Extract the following duplicate code blocks from this Python file into a "
        f"helper function.\n\nFile source:\n```python\n{snippet}\n```\n\n"
        f"Duplicate blocks:\n{blocks_text}\n\n"
        "Place the helper immediately before the enclosing function of its first use. "
        f"{staticmethod_instruction}"
        "Return complete, valid Python for the helper and each call site replacement. "
        "Each call site replacement must start with the same leading indentation as "
        "the block it replaces, end with a trailing newline, and cover only the exact "
        "lines of the duplicate block — stopping before the 'Line immediately after "
        "this block' marker shown above. Do not include any code from before or after "
        "the block. "
        "Double-check that only required parameters are passed to the helper — do not "
        "include an unused parameter, or one that is overwritten before being read. "
        "Be mindful of the code being removed from the call site: if variable "
        "assignments are moved into the helper, those variables may no longer be "
        "defined in the calling scope at that point. "
        "If the helper uses a sentinel return value to signal an error path (such as "
        "returning an empty collection), check for it at the call site with `==`, not "
        "`is` — `is` only gives correct results for singletons like `None`, `True`, "
        "and `False`, not for constructed objects like `set()`."
        f"{escaping_note}"
        f"{used_names_note}"
        f"{docstring_note}"
        f"{veto_notes_note}"
        f"{failures_note}"
    )
    return _llm_client.call_with_tool(
        client,
        provider,
        model,
        1024,
        _EXTRACT_TOOL,
        "extract_helper",
        [{"role": "user", "content": prompt}],
        caller="DuplicateExtractor",
        tool_choice_override=tool_choice_override,
    )


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
    return _process_llm_result(result)


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
