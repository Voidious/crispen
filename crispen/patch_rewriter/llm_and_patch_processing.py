from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, TYPE_CHECKING, Tuple, Union
import ast
import difflib
import sys
from libcst.metadata import MetadataWrapper, PositionProvider
import libcst as cst
from ..llm_client import call_with_tool, get_api_key, make_client
from ..patch_updater import apply_patch_strings
from .models_and_consts import (
    RewriteAccumulator,
    _ConstRef,
    _EXCLUDED_DIR_NAMES,
    _FLContext,
    _TestFunctionInfo,
    _build_attr_const_map,
    _build_const_map,
    _is_patch_call,
    _matches_any,
)


if TYPE_CHECKING:
    from ..config import CrispenConfig


_PATCH_RULES = (
    "\n## Rules for updating patch() strings:\n"
    "**Core principle:** `@patch('A.B.Name')` intercepts the `Name` attribute in "
    "module `A.B`'s namespace. Python resolves `Name` in the **defining** module "
    "of the function that uses it — NOT in any module that merely re-exports it.\n\n"
    "**Step-by-step algorithm:**\n"
    "1. Identify the production function **F** being tested "
    "(look at what the test calls or constructs).\n"
    "2. Look up **F** in the entity migration map:\n"
    "   - If F was **not migrated**: it still runs in the original module — "
    "leave the patch unchanged.\n"
    "   - If F was **migrated to new module M**: go to step 3.\n"
    "3. Find the **patched name** (last component of the patch string before the "
    "closing quote, e.g. `Name` in `old_module.Name`).\n"
    "4. Check new module M's source: does it import `Name`? "
    "(e.g. `from libcst.metadata import MetadataWrapper`)\n"
    "   - If yes: update the patch string to `M.Name`.\n"
    "   - If no: leave the patch unchanged "
    "(Name is not used by F in its new location).\n"
    "5. **Re-exports are irrelevant.** If `old_module/__init__.py` still imports "
    "`Name` independently (e.g. `from libcst.metadata import Name`) or re-exports "
    "F (e.g. `from .M import F`), those are SEPARATE bindings in a SEPARATE "
    "namespace. Patching `old_module.Name` does NOT intercept F's lookup of `Name` "
    "in M.\n\n"
    "**Example:**\n"
    "```python\n"
    "# Before split: _apply_foo() defined in crispen.engine, imports MetadataWrapper\n"
    "@patch('crispen.engine.MetadataWrapper')  # correct before split\n\n"
    "# After split: _apply_foo() moved to crispen.engine.helpers\n"
    "# crispen/engine/helpers.py: from libcst.metadata import MetadataWrapper\n"
    "# crispen/engine/__init__.py may ALSO import MetadataWrapper, but\n"
    "# that is a separate binding — patching it does NOT affect "
    "helpers.MetadataWrapper\n"
    "@patch('crispen.engine.helpers.MetadataWrapper')  # correct after split\n"
    "```\n"
)

_PATCH_SINGLE_REWRITE_TOOL: dict = {
    "name": "update_patch_string",
    "description": (
        "Decide whether a single patch() string needs updating after a source "
        "file was split into sub-modules. Return the new string, or the original "
        "if no change is needed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "new_patch_string": {
                "type": "string",
                "description": (
                    "The updated patch string. Return the ORIGINAL string unchanged "
                    "if the existing patch is still correct."
                ),
            }
        },
        "required": ["new_patch_string"],
    },
}

_PATCH_SINGLE_VERIFY_TOOL: dict = {
    "name": "verify_patch_update",
    "description": (
        "Verify whether a proposed patch() string update is correct after a "
        "source file was split into sub-modules."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "correct": {
                "type": "boolean",
                "description": "True if the proposed update is correct.",
            },
            "issue": {
                "type": "string",
                "description": (
                    "What is wrong with the proposed update. "
                    "Empty string when correct."
                ),
            },
        },
        "required": ["correct", "issue"],
    },
}


class _ConstSubstitutor(cst.CSTTransformer):
    """Replace ``@patch(NAME)`` with ``@patch("value")`` for known constants."""

    def __init__(self, substitutions: Dict[str, str]) -> None:
        self._subs = substitutions  # {const_name: string_value}

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        if not _is_patch_call(updated_node):
            return updated_node
        if not updated_node.args:
            return updated_node
        arg0 = updated_node.args[0].value
        if isinstance(arg0, cst.Name):
            key = arg0.value
        elif isinstance(arg0, cst.Attribute) and isinstance(arg0.value, cst.Name):
            key = f"{arg0.value.value}.{arg0.attr.value}"
        else:
            return updated_node
        if key not in self._subs:
            return updated_node
        new_string = cst.SimpleString(f'"{self._subs[key]}"')
        new_arg = updated_node.args[0].with_changes(value=new_string)
        return updated_node.with_changes(args=(new_arg,) + updated_node.args[1:])


def _substitute_consts_in_func_text(
    full_text: str, substitutions: Dict[str, str]
) -> str:
    """Return *full_text* with ``@patch(NAME)`` replaced by ``@patch("value")``."""
    if not substitutions:
        return full_text
    try:
        tree = cst.parse_module(full_text)
    except cst.ParserSyntaxError:
        return full_text
    return tree.visit(_ConstSubstitutor(substitutions)).code


def _find_with_patch_paths_in_body(
    func_text: str,
    old_paths: Set[str],
    const_map: Dict[str, Tuple[str, str]],
    attr_const_map: Dict[str, Dict[str, Tuple[str, str]]],
) -> List[str]:
    """Return patch() string args from ``with patch(...)`` statements in *func_text*.

    Walks the function body but does not recurse into nested function definitions.
    Handles plain string literals, module-level named constants, and
    ``module.CONSTANT`` attribute forms for both ``patch(...)`` and ``*.patch(...)``.
    Also covers ``async with patch(...)`` context managers.
    """
    try:
        tree = ast.parse(func_text)
    except SyntaxError:
        return []
    func_node: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = None
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_node = stmt
            break
    if func_node is None:
        return []

    # Collect IDs of all descendants of nested FunctionDef/AsyncFunctionDef nodes so
    # that ``with`` statements inside closures are excluded from results.
    excluded: Set[int] = set()
    for node in ast.walk(func_node):
        if node is func_node:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                excluded.add(id(child))

    results: List[str] = []
    for node in ast.walk(func_node):
        if id(node) in excluded:
            continue
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            is_patch = (isinstance(func, ast.Name) and func.id == "patch") or (
                isinstance(func, ast.Attribute) and func.attr == "patch"
            )
            if not is_patch or not call.args:
                continue
            arg0 = call.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                if _matches_any(arg0.value, old_paths):
                    results.append(arg0.value)
            elif isinstance(arg0, ast.Name) and arg0.id in const_map:
                val, _ = const_map[arg0.id]
                if _matches_any(val, old_paths):
                    results.append(val)
            elif isinstance(arg0, ast.Attribute) and isinstance(arg0.value, ast.Name):
                alias = arg0.value.id
                attr_name = arg0.attr
                if alias in attr_const_map and attr_name in attr_const_map[alias]:
                    val, _ = attr_const_map[alias][attr_name]
                    if _matches_any(val, old_paths):
                        results.append(val)
    return results


class _PatchFunctionCollector(cst.CSTVisitor):
    """Collect FunctionDef nodes whose @patch decorators match *old_paths*."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        source_lines: List[str],
        old_paths: Set[str],
        const_map: Dict[str, Tuple[str, str]],
        attr_const_map: Dict[str, Dict[str, Tuple[str, str]]],
    ) -> None:
        self._lines = source_lines
        self._old_paths = old_paths
        self._const_map = const_map  # {name: (string_value, abs_def_file)}
        self._attr_const_map = attr_const_map  # {alias: {attr: (value, abs_def_file)}}
        self.functions: List[_TestFunctionInfo] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        old_patch_paths: List[str] = []
        const_refs: List[_ConstRef] = []
        patch_dec_idx = 0  # counts only @patch / *.patch decorators

        for dec in node.decorators:
            if not isinstance(dec.decorator, cst.Call):
                continue
            if not _is_patch_call(dec.decorator):
                continue
            if not dec.decorator.args:
                patch_dec_idx += 1
                continue
            arg0 = dec.decorator.args[0].value
            if isinstance(arg0, cst.SimpleString):
                raw = arg0.value
                if raw and raw[0] in ('"', "'") and not raw.startswith(('"""', "'''")):
                    inner = raw[1:-1]
                    if _matches_any(inner, self._old_paths):
                        old_patch_paths.append(inner)
            elif isinstance(arg0, cst.Name) and arg0.value in self._const_map:
                const_val, const_file = self._const_map[arg0.value]
                if _matches_any(const_val, self._old_paths):
                    old_patch_paths.append(const_val)
                    const_refs.append(
                        _ConstRef(
                            const_name=arg0.value,
                            source_file=const_file,
                            resolved_value=const_val,
                            patch_dec_idx=patch_dec_idx,
                        )
                    )
            elif isinstance(arg0, cst.Attribute) and isinstance(arg0.value, cst.Name):
                module_alias = arg0.value.value
                attr_name = arg0.attr.value
                if module_alias in self._attr_const_map:
                    attr_map = self._attr_const_map[module_alias]
                    if attr_name in attr_map:
                        const_val, const_file = attr_map[attr_name]
                        if _matches_any(const_val, self._old_paths):
                            old_patch_paths.append(const_val)
                            const_refs.append(
                                _ConstRef(
                                    const_name=f"{module_alias}.{attr_name}",
                                    source_file=const_file,
                                    resolved_value=const_val,
                                    patch_dec_idx=patch_dec_idx,
                                )
                            )
            patch_dec_idx += 1

        # Compute line range and extract full text before the early-return check so
        # the body scan (below) can run even when there are no matching decorators.
        func_pos = self.get_metadata(PositionProvider, node)
        if node.decorators:
            # FunctionDef position starts at "def", not the first decorator.
            dec_pos = self.get_metadata(PositionProvider, node.decorators[0])
            start_line = dec_pos.start.line
        else:
            start_line = func_pos.start.line
        end_line = func_pos.end.line

        original_full_text = "\n".join(self._lines[start_line - 1 : end_line])

        # Also scan the function body for ``with patch(...)`` context managers.
        body_paths = _find_with_patch_paths_in_body(
            original_full_text, self._old_paths, self._const_map, self._attr_const_map
        )
        old_patch_paths.extend(body_paths)

        if not old_patch_paths:
            return

        # Build the full_text sent to the LLM: substitute constant names with
        # their string values so the LLM always sees plain string literals.
        if const_refs:
            subs = {ref.const_name: ref.resolved_value for ref in const_refs}
            llm_full_text = _substitute_consts_in_func_text(original_full_text, subs)
        else:
            llm_full_text = original_full_text

        self.functions.append(
            _TestFunctionInfo(
                function_name=node.name.value,
                full_text=llm_full_text,
                old_patch_paths=old_patch_paths,
                start_line=start_line,
                end_line=end_line,
                const_refs=const_refs,
            )
        )


def _find_test_functions_to_update(
    source: str,
    old_paths: Set[str],
    scan_file: str = "",
    repo_root: Optional[str] = None,
) -> List[_TestFunctionInfo]:
    """Return functions in *source* with @patch decorators matching *old_paths*.

    When *scan_file* is provided, also resolves named constants used as @patch
    arguments (module-level assignments and resolvable from-imports).
    """
    if not old_paths:
        return []
    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError:
        return []
    lines = source.splitlines()
    const_map: Dict[str, Tuple[str, str]] = {}
    attr_const_map: Dict[str, Dict[str, Tuple[str, str]]] = {}
    if scan_file:
        const_map = _build_const_map(source, scan_file, repo_root)
        attr_const_map = _build_attr_const_map(source, scan_file, repo_root)
    wrapper = MetadataWrapper(tree)
    collector = _PatchFunctionCollector(lines, old_paths, const_map, attr_const_map)
    wrapper.visit(collector)
    return collector.functions


def _build_context_message(fl_contexts: List[_FLContext]) -> str:
    """Build the shared LLM prompt context describing all split files."""
    parts: List[str] = [
        "A Python source file was split into multiple sub-modules by an automated "
        "refactoring tool.  Update the patch() call strings in the provided "
        "test functions so they reference the correct new module paths.\n"
    ]

    for ctx in fl_contexts:
        parts.append(f"\n## Split module: `{ctx.old_module}` ({ctx.filepath})\n")

        orig_lines = ctx.original_source.splitlines(keepends=True)
        mod_lines = ctx.modified_source.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                orig_lines,
                mod_lines,
                fromfile=f"{ctx.old_module} (before)",
                tofile=f"{ctx.old_module} (after)",
            )
        )
        if diff_lines:
            parts.append("### Changes to original file (diff):\n```diff\n")
            parts.extend(diff_lines)
            parts.append("```\n")

        parts.append(
            f"### Modified original file `{ctx.old_module}` (current state):\n"
            "```python\n"
        )
        parts.append(ctx.modified_source)
        parts.append("```\n")

        for rel_path, content in ctx.new_files.items():
            new_mod = ctx.new_module_paths.get(rel_path, rel_path)
            parts.append(
                f"### New file `{rel_path}` (module: `{new_mod}`):\n```python\n"
            )
            parts.append(content)
            parts.append("```\n")

        parts.append("### Entity migration:\n")
        for entity_name in sorted(ctx.entity_to_target):
            target_rel = ctx.entity_to_target[entity_name]
            new_mod = ctx.new_module_paths.get(target_rel, target_rel)
            parts.append(f"- `{entity_name}` → `{target_rel}` (module: `{new_mod}`)\n")

    return "".join(parts)


def _build_single_patch_prompt(
    context_msg: str,
    function_text: str,
    old_patch_string: str,
    prev_issue: Optional[str] = None,
    prev_proposed: Optional[str] = None,
) -> str:
    """Build the user prompt for a single-patch rewrite LLM call."""
    parts = [context_msg, _PATCH_RULES]
    if prev_issue:
        parts.append(
            f"\n## Previous attempt was rejected:\n"
            f"- Proposed: `{prev_proposed}`\n"
            f"- Issue: {prev_issue}\n"
        )
    parts.append(
        f"\n## Test function:\n```python\n{function_text}\n```\n\n"
        f"## Patch string to evaluate:\n`{old_patch_string}`\n\n"
        f"Should this patch string change after the split? "
        f"Return the new string, or the original `{old_patch_string}` "
        f"if no change is needed.\n"
    )
    return "".join(parts)


def _build_single_verify_prompt(
    context_msg: str,
    function_text: str,
    old_patch_string: str,
    new_patch_string: str,
) -> str:
    """Build the user prompt for a single-patch verify LLM call."""
    parts = [
        context_msg,
        _PATCH_RULES,
        f"\n## Test function:\n```python\n{function_text}\n```\n\n"
        f"## Proposed patch() string update:\n"
        f"- Original: `{old_patch_string}`\n"
        f"- Proposed: `{new_patch_string}`\n\n"
        f"Is this update correct? Set `correct` to True only if the proposed "
        f"patch string points to where the name is looked up after the split.\n",
    ]
    return "".join(parts)


def _process_file_source(
    source: str,
    all_forking_paths: Set[str],
    context_msg: str,
    client: Any,
    config: "CrispenConfig",
    max_attempts: int,
    scan_file: str = "",
    repo_root: Optional[str] = None,
    verbose: bool = False,
    _acc: Optional[RewriteAccumulator] = None,
) -> Tuple[str, bool, Dict[str, Dict[str, str]]]:
    """Scan *source* for @patch functions matching *all_forking_paths* and update.

    Processes one patch string at a time.  Returns
    ``(updated_source, was_changed, cross_file_patch_maps)`` where
    *cross_file_patch_maps* maps absolute file path → {old_string: new_string}
    for constant definitions in other files.
    """
    functions = _find_test_functions_to_update(
        source, all_forking_paths, scan_file, repo_root
    )
    if not functions:
        return source, False, {}

    # Build {old_path: representative_function} for each unique patch string.
    file_desc = f"'{scan_file}'" if scan_file else "file"
    unique_patches: Dict[str, _TestFunctionInfo] = {}
    for func in functions:
        for old_path in func.old_patch_paths:
            if old_path not in unique_patches:
                unique_patches[old_path] = func

    patch_map: Dict[str, str] = {}  # old_path → new_path

    for old_path, rep_func in unique_patches.items():
        prev_issue: Optional[str] = None
        prev_proposed: Optional[str] = None
        attempts_left = max_attempts

        while attempts_left > 0:
            attempts_left -= 1

            prompt = _build_single_patch_prompt(
                context_msg, rep_func.full_text, old_path, prev_issue, prev_proposed
            )
            retry_label = " (retry)" if prev_issue is not None else ""
            if verbose:
                print(
                    f"crispen: patch_rewriter: evaluating '{old_path}'"
                    f" in {file_desc}{retry_label}",
                    file=sys.stderr,
                    flush=True,
                )
            r = call_with_tool(
                client,
                config.provider,
                config.model,
                256,
                _PATCH_SINGLE_REWRITE_TOOL,
                "update_patch_string",
                [{"role": "user", "content": prompt}],
                caller="patch_rewriter",
                tool_choice_override=config.tool_choice,
            )
            if _acc is not None:
                _acc.calls += 1
                _acc.elapsed += r.elapsed
                _acc.input_tokens += r.input_tokens
                _acc.output_tokens += r.output_tokens
            if verbose and config.timing == "detailed":
                print(
                    f"crispen: patch_rewriter:   → done [{r.elapsed:.2f}s,"
                    f" {r.input_tokens:,} in / {r.output_tokens:,} out]",
                    file=sys.stderr,
                    flush=True,
                )
            if r.tool_input is None:
                break

            new_path = r.tool_input.get("new_patch_string", old_path)
            if not isinstance(new_path, str) or not new_path:
                break
            if new_path == old_path:
                break  # no change needed

            # Verify.
            verify_prompt = _build_single_verify_prompt(
                context_msg, rep_func.full_text, old_path, new_path
            )
            if verbose:
                print(
                    f"crispen: patch_rewriter: verifying '{old_path}'"
                    f" → '{new_path}'",
                    file=sys.stderr,
                    flush=True,
                )
            v = call_with_tool(
                client,
                config.provider,
                config.model,
                256,
                _PATCH_SINGLE_VERIFY_TOOL,
                "verify_patch_update",
                [{"role": "user", "content": verify_prompt}],
                caller="patch_rewriter",
                tool_choice_override=config.tool_choice,
            )
            if _acc is not None:
                _acc.calls += 1
                _acc.elapsed += v.elapsed
                _acc.input_tokens += v.input_tokens
                _acc.output_tokens += v.output_tokens
            if verbose and config.timing == "detailed":
                print(
                    f"crispen: patch_rewriter:   → done [{v.elapsed:.2f}s,"
                    f" {v.input_tokens:,} in / {v.output_tokens:,} out]",
                    file=sys.stderr,
                    flush=True,
                )
            if v.tool_input is None:
                # Verify call failed; accept proposed update.
                patch_map[old_path] = new_path
                break

            verify_correct = v.tool_input.get("correct", False)
            issue = v.tool_input.get("issue", "")
            if verbose:
                v_status = "ACCEPTED" if verify_correct else "REJECTED"
                print(
                    f"crispen: patch_rewriter: verify {v_status}",
                    file=sys.stderr,
                    flush=True,
                )
                if not verify_correct and issue:
                    print(
                        f"crispen: patch_rewriter:   issue: {issue}",
                        file=sys.stderr,
                        flush=True,
                    )
            if verify_correct:
                patch_map[old_path] = new_path
                break
            else:
                if attempts_left > 0:
                    prev_issue = issue
                    prev_proposed = new_path
                # else: retries exhausted — skip this patch string

    # Collect cross-file constant definition updates.
    # Same-file constants are automatically updated by apply_patch_strings below,
    # since it replaces all string literals matching the old path.
    cross_file_patch_maps: Dict[str, Dict[str, str]] = {}
    if patch_map and scan_file:
        scan_file_abs = str(Path(scan_file).resolve())
        for func in functions:
            for ref in func.const_refs:
                if ref.source_file == scan_file_abs:
                    continue  # handled by apply_patch_strings
                new_val = patch_map.get(ref.resolved_value)
                if new_val is None or new_val == ref.resolved_value:
                    continue
                cross_file_patch_maps.setdefault(ref.source_file, {})[
                    ref.resolved_value
                ] = new_val

    if not patch_map:
        return source, False, cross_file_patch_maps

    updated = apply_patch_strings(source, patch_map)
    return updated, updated != source, cross_file_patch_maps


def _apply_cross_file_const_updates(
    cross_file_proposals: Dict[str, Dict[str, Set[str]]],
    per_file: Dict[str, Any],
    _acc: Optional[RewriteAccumulator] = None,
) -> Iterator[str]:
    """Apply agreed-upon cross-file constant definition updates.

    *cross_file_proposals* maps absolute file path → {old_val → {new_val, …}}.
    For each constant source file, we apply the update only when every proposal
    agrees on a single new value.  Conflicting proposals (different scan files
    suggesting different new values for the same constant) are skipped — the
    affected functions have already been inlined with their individual new values.
    """
    for abs_file, old_to_new_sets in cross_file_proposals.items():
        resolved = {
            old: next(iter(new_set))
            for old, new_set in old_to_new_sets.items()
            if len(new_set) == 1
        }
        if not resolved:
            continue

        # Check whether this file is tracked in per_file.
        per_file_entry = next(
            (
                state
                for f, state in per_file.items()
                if str(Path(f).resolve()) == abs_file
            ),
            None,
        )
        if per_file_entry is not None:
            new_src = apply_patch_strings(per_file_entry["source"], resolved)
            if new_src != per_file_entry["source"]:
                per_file_entry["source"] = new_src
                per_file_entry["msgs"].append(
                    "patch_update: updated @patch constant definition (rewrite)"
                )
                if _acc is not None:
                    _acc.files_updated += 1
            continue

        # Disk file.
        try:
            old_src = Path(abs_file).read_text(encoding="utf-8")
        except OSError:
            continue
        new_src = apply_patch_strings(old_src, resolved)
        if new_src != old_src:
            Path(abs_file).write_text(new_src, encoding="utf-8")
            if _acc is not None:
                _acc.files_updated += 1
            yield (
                f"{abs_file}: patch_update: "
                "updated @patch constant definition (rewrite)"
            )


def apply_patch_rewrite(
    fl_contexts: List[_FLContext],
    per_file: Dict[str, Any],
    repo_root: Optional[str],
    config: "CrispenConfig",
    verbose: bool = False,
    _acc: Optional[RewriteAccumulator] = None,
) -> Iterator[str]:
    """Update @patch strings for forking entities using LLM.

    Called after "basic" patch updates have already been applied.  Handles
    entities that basic mode skipped because they appeared in multiple callers.
    Also resolves named constants used as @patch arguments and updates their
    definitions when all usages agree on the same new value.
    """
    if not fl_contexts:
        return

    all_forking_paths: Set[str] = set()
    for ctx in fl_contexts:
        all_forking_paths |= ctx.forking_old_paths

    if not all_forking_paths:
        return

    api_key = get_api_key(config.provider, "patch_rewriter")
    client = make_client(
        config.provider, api_key, timeout=config.api_timeout, base_url=config.base_url
    )

    context_msg = _build_context_message(fl_contexts)
    max_attempts = 1 + config.patch_update_retries

    per_file_abs = {str(Path(f).resolve()) for f in per_file}

    # Aggregate cross-file constant proposals:
    # abs_file → {old_val → {proposed_new_val, …}}
    cross_file_proposals: Dict[str, Dict[str, Set[str]]] = {}

    # Update per_file sources (in memory, not yet written to disk).
    for filepath, state in per_file.items():
        new_src, changed, cross = _process_file_source(
            state["source"],
            all_forking_paths,
            context_msg,
            client,
            config,
            max_attempts,
            scan_file=filepath,
            repo_root=repo_root,
            verbose=verbose,
            _acc=_acc,
        )
        if changed:
            state["source"] = new_src
            state["msgs"].append(
                f"{filepath}: patch_update: updated @patch strings (rewrite)"
            )
            if _acc is not None:
                _acc.files_updated += 1
        for abs_file, patch_map in cross.items():
            for old_val, new_val in patch_map.items():
                cross_file_proposals.setdefault(abs_file, {}).setdefault(
                    old_val, set()
                ).add(new_val)

    if repo_root is None:
        yield from _apply_cross_file_const_updates(
            cross_file_proposals, per_file, _acc=_acc
        )
        return

    # Scan every other .py file in the repo.
    repo_root_path = Path(repo_root)
    for py_file in sorted(repo_root_path.rglob("*.py")):
        if str(py_file.resolve()) in per_file_abs:
            continue
        if any(
            part in _EXCLUDED_DIR_NAMES
            for part in py_file.relative_to(repo_root_path).parts[:-1]
        ):
            continue
        try:
            src = py_file.read_text(encoding="utf-8")
        except OSError:
            continue
        new_src, changed, cross = _process_file_source(
            src,
            all_forking_paths,
            context_msg,
            client,
            config,
            max_attempts,
            scan_file=str(py_file),
            repo_root=repo_root,
            verbose=verbose,
            _acc=_acc,
        )
        if changed:
            py_file.write_text(new_src, encoding="utf-8")
            if _acc is not None:
                _acc.files_updated += 1
            yield f"{py_file}: patch_update: updated @patch strings (rewrite)"
        for abs_file, patch_map in cross.items():
            for old_val, new_val in patch_map.items():
                cross_file_proposals.setdefault(abs_file, {}).setdefault(
                    old_val, set()
                ).add(new_val)

    yield from _apply_cross_file_const_updates(
        cross_file_proposals, per_file, _acc=_acc
    )
