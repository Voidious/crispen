"""LLM-powered @patch string rewriter for FileLimiter 'rewrite' mode."""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TYPE_CHECKING

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from .llm_client import call_with_tool, get_api_key, make_client
from .patch_updater import apply_patch_strings

if TYPE_CHECKING:
    from .config import CrispenConfig  # pragma: no cover

# Test functions per LLM call.
_CHUNK_SIZE = 10

# Directory names excluded from the repo-wide file scan (mirrors engine.py).
_EXCLUDED_DIR_NAMES = frozenset(
    {".venv", "venv", "env", ".tox", "__pycache__", "node_modules"}
)


@dataclass
class _FLContext:
    """Context from one FileLimiter result, consumed by the patch rewriter."""

    filepath: str
    old_module: str
    original_source: str  # source BEFORE splitting
    modified_source: str  # source AFTER splitting (fl_result.original_source)
    new_files: Dict[str, str]  # rel_path → content
    new_module_paths: Dict[str, str]  # rel_path → dotted module path
    entity_to_target: Dict[str, str]  # entity_name → rel_path
    forking_old_paths: Set[str] = field(default_factory=set)


@dataclass
class _ConstRef:
    """Records that a @patch decorator arg was substituted from a named constant.

    When a test function uses ``@patch(SOME_NAME)`` instead of a string literal,
    we resolve the constant's value, substitute it for the LLM, and record the
    substitution here so we can later update the constant's definition.
    """

    const_name: str
    source_file: str  # absolute path of the file where the constant is defined
    resolved_value: str  # the old string value (e.g. "myapp.service.MyClass")
    patch_dec_idx: int  # 0-based index among @patch decorators on this function


@dataclass
class _TestFunctionInfo:
    """A test function containing @patch decorators referencing old paths."""

    function_name: str
    full_text: str  # text sent to LLM (constants substituted with their values)
    old_patch_paths: List[str]
    start_line: int  # 1-indexed, inclusive (includes decorators)
    end_line: int  # 1-indexed, inclusive
    const_refs: List[_ConstRef] = field(default_factory=list)


_PATCH_REWRITE_TOOL: dict = {
    "name": "update_patch_strings",
    "description": (
        "Update @patch decorator strings in the provided test functions so they "
        "point to the correct new module paths after a source file was split."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "description": "Exact name of the test function.",
                        },
                        "updated_code": {
                            "type": "string",
                            "description": (
                                "Complete updated function code including all "
                                "decorators and body, with corrected @patch strings."
                            ),
                        },
                    },
                    "required": ["function_name", "updated_code"],
                },
                "description": "One entry per test function that needs updating.",
            }
        },
        "required": ["updates"],
    },
}

_PATCH_VERIFY_TOOL: dict = {
    "name": "verify_patch_updates",
    "description": (
        "Verify that proposed @patch string updates for test functions are "
        "correct after a source file was split into multiple sub-modules."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "correct": {
                "type": "boolean",
                "description": "True if ALL proposed updates are correct.",
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "function_name": {"type": "string"},
                        "issue": {
                            "type": "string",
                            "description": "What is wrong with the proposed update.",
                        },
                    },
                    "required": ["function_name", "issue"],
                },
                "description": "Empty list when correct is True.",
            },
        },
        "required": ["correct", "issues"],
    },
}


def _is_patch_call(call: cst.Call) -> bool:
    """Return True if *call* is a patch(...) or *.patch(...) call."""
    func = call.func
    if isinstance(func, cst.Name) and func.value == "patch":
        return True
    if isinstance(func, cst.Attribute) and func.attr.value == "patch":
        return True
    return False


def _matches_any(inner: str, old_paths: Set[str]) -> bool:
    """Return True if *inner* matches any old_path exactly or as a dotted prefix."""
    for old in old_paths:
        if inner == old or inner.startswith(old + "."):
            return True
    return False


def _compiles(code: str) -> bool:
    """Return True if *code* compiles as valid Python."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Constant resolution helpers
# ---------------------------------------------------------------------------


def _build_local_const_map(source: str) -> Dict[str, str]:
    """Return {name: string_value} for module-level ``NAME = "string"`` assignments."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    result: Dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            result[node.targets[0].id] = node.value.value
    return result


def _resolve_import_to_file(
    module: Optional[str],
    level: int,
    scan_file: str,
    repo_root: Optional[str],
) -> Optional[str]:
    """Resolve a from-import to an absolute file path, or None if unresolvable.

    *level* is the number of leading dots (0 = absolute, 1 = same package,
    2 = parent package, …).
    """
    scan_dir = Path(scan_file).parent
    if level > 0:
        # Relative import: walk up (level - 1) directories from the scan file's
        # package directory.
        base = scan_dir
        for _ in range(level - 1):
            base = base.parent
        if module:
            parts = module.split(".")
            candidate = base.joinpath(*parts)
            for path in [candidate / "__init__.py", candidate.with_suffix(".py")]:
                if path.exists():
                    return str(path)
        else:
            # "from . import NAME" — the source is the package __init__.py.
            init = base / "__init__.py"
            if init.exists():
                return str(init)
    else:
        # Absolute import — requires repo_root to locate the file.
        if repo_root is None or not module:
            return None
        parts = module.split(".")
        candidate = Path(repo_root).joinpath(*parts)
        for path in [candidate / "__init__.py", candidate.with_suffix(".py")]:
            if path.exists():
                return str(path)
    return None


def _build_const_map(
    source: str,
    scan_file: str,
    repo_root: Optional[str],
) -> Dict[str, Tuple[str, str]]:
    """Return {name: (string_value, abs_def_file)} for resolving @patch constants.

    Covers:
    - Module-level ``NAME = "string"`` assignments in the same file.
    - Names brought in by ``from X import NAME`` where the source file is
      reachable.  Local definitions take priority over imported ones.
    """
    scan_abs = str(Path(scan_file).resolve())
    result: Dict[str, Tuple[str, str]] = {}

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return result

    # Same-file constants first (highest priority).
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            result[node.targets[0].id] = (node.value.value, scan_abs)

    # Cross-file: from-imports that bring in named constants.
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if any(alias.name == "*" for alias in node.names):
            continue
        imp_file = _resolve_import_to_file(
            node.module, node.level or 0, scan_file, repo_root
        )
        if imp_file is None:
            continue
        try:
            imp_src = Path(imp_file).read_text(encoding="utf-8")
        except OSError:
            continue
        imp_consts = _build_local_const_map(imp_src)
        imp_abs = str(Path(imp_file).resolve())
        for alias in node.names:
            local_name = alias.asname if alias.asname else alias.name
            if local_name in result:
                continue  # local definition takes priority
            if alias.name in imp_consts:
                result[local_name] = (imp_consts[alias.name], imp_abs)

    return result


def _build_attr_const_map(
    source: str,
    scan_file: str,
    repo_root: Optional[str],
) -> Dict[str, Dict[str, Tuple[str, str]]]:
    """Return {module_alias: {attr_name: (value, abs_def_file)}} for ``import X`` stmts.

    Enables resolving ``@patch(module.CONSTANT)`` attribute-access style references.
    For ``import constants``, the alias is ``"constants"``.
    For ``import myapp.constants as C``, the alias is ``"C"``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    result: Dict[str, Dict[str, Tuple[str, str]]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            local_name = alias.asname if alias.asname else alias.name
            mod_file = _resolve_import_to_file(alias.name, 0, scan_file, repo_root)
            if mod_file is None:
                continue
            try:
                mod_src = Path(mod_file).read_text(encoding="utf-8")
            except OSError:
                continue
            mod_consts = _build_local_const_map(mod_src)
            mod_abs = str(Path(mod_file).resolve())
            result[local_name] = {
                name: (val, mod_abs) for name, val in mod_consts.items()
            }
    return result


# ---------------------------------------------------------------------------
# Constant substitution helpers (prepare full_text for LLM)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Extraction helper (read new values back from LLM output)
# ---------------------------------------------------------------------------


def _extract_patch_args_from_code(code: str) -> List[Optional[str]]:
    """Return first-arg string values for ``@patch`` decorators in *code*, in order.

    Only ``patch(...)`` and ``*.patch(...)`` calls are counted.  A non-string-literal
    first arg (or a decorator with no args) yields ``None`` at that position.
    """
    try:
        tree = cst.parse_module(code)
    except cst.ParserSyntaxError:
        return []
    results: List[Optional[str]] = []
    for stmt in tree.body:
        if not isinstance(stmt, cst.FunctionDef):
            continue
        for dec in stmt.decorators:
            if not isinstance(dec.decorator, cst.Call):
                continue
            if not _is_patch_call(dec.decorator):
                continue
            if not dec.decorator.args:
                results.append(None)
                continue
            arg0 = dec.decorator.args[0].value
            if isinstance(arg0, cst.SimpleString):
                raw = arg0.value
                if raw and raw[0] in ('"', "'") and not raw.startswith(('"""', "'''")):
                    results.append(raw[1:-1])
                else:
                    results.append(None)
            else:
                results.append(None)
    return results


# ---------------------------------------------------------------------------
# CST visitor: collect functions whose @patch decorators reference old paths
# ---------------------------------------------------------------------------


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

        if not old_patch_paths:
            return

        func_pos = self.get_metadata(PositionProvider, node)
        if node.decorators:
            # FunctionDef position starts at "def", not the first decorator.
            dec_pos = self.get_metadata(PositionProvider, node.decorators[0])
            start_line = dec_pos.start.line
        else:  # pragma: no cover
            start_line = func_pos.start.line
        end_line = func_pos.end.line

        original_full_text = "\n".join(self._lines[start_line - 1 : end_line])

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


# ---------------------------------------------------------------------------
# LLM prompt builders
# ---------------------------------------------------------------------------


def _build_context_message(fl_contexts: List[_FLContext]) -> str:
    """Build the shared LLM prompt context describing all split files."""
    parts: List[str] = [
        "A Python source file was split into multiple sub-modules by an automated "
        "refactoring tool.  Update the @patch decorator strings in the provided "
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

    parts.append(
        "\n## Rules for updating @patch strings:\n"
        "1. If the modified original file still re-exports an entity "
        '(e.g. `from .newfile import Entity`), then `@patch("old_module.Entity")` '
        "is **still valid** — leave it unchanged.\n"
        "2. If the entity is no longer accessible from the original module, update "
        "the @patch path to point to the new sub-module.\n"
        "3. For entities split across multiple new files, use what the test function "
        "calls or imports to determine which new module is the correct patch target.\n"
        "4. Only modify @patch string literals — do not change any other code.\n"
        "5. Return every function in the input, even those left unchanged.\n"
    )
    return "".join(parts)


def _build_rewrite_prompt(
    context_msg: str,
    functions: List[_TestFunctionInfo],
    prev_issues: Optional[List[dict]] = None,
) -> str:
    """Build the user message for the rewrite LLM call."""
    parts = [context_msg]
    if prev_issues:
        parts.append("\n## Previous attempt had these issues — please fix them:\n")
        for issue in prev_issues:
            parts.append(f"- **{issue['function_name']}**: {issue['issue']}\n")
    parts.append("\n## Test functions to update:\n")
    for func in functions:
        parts.append(f"```python\n{func.full_text}\n```\n")
    return "".join(parts)


def _build_verify_prompt(
    context_msg: str,
    original_functions: List[_TestFunctionInfo],
    proposed_updates: Dict[str, str],
) -> str:
    """Build the user message for the verify LLM call."""
    parts = [context_msg, "\n## Proposed @patch string updates:\n"]
    for func in original_functions:
        parts.append(f"\n### `{func.function_name}`\n**Original:**\n```python\n")
        parts.append(func.full_text)
        parts.append("```\n")
        if func.function_name in proposed_updates:
            parts.append("**Proposed update:**\n```python\n")
            parts.append(proposed_updates[func.function_name])
            parts.append("```\n")
        else:
            parts.append("*(no update proposed)*\n")
    parts.append(
        "\nVerify each proposed update:\n"
        "- Are the new @patch paths correct for where each entity now resides?\n"
        "- Were any patches incorrectly changed that should remain unchanged?\n"
        "- Were any patches left unchanged that should have been updated?\n"
        "Set `correct` to True only if ALL updates are correct.\n"
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Source mutation helpers
# ---------------------------------------------------------------------------


def _apply_function_updates(
    source: str,
    functions: List[_TestFunctionInfo],
    updates: Dict[str, str],
) -> str:
    """Replace function text in *source* with LLM-provided updated versions.

    Applies replacements in reverse line order so earlier positions stay valid.
    """
    if not updates:
        return source
    lines = source.splitlines(keepends=True)
    to_replace = sorted(
        [f for f in functions if f.function_name in updates],
        key=lambda f: f.start_line,
        reverse=True,
    )
    for func in to_replace:
        new_code = updates[func.function_name]
        if not new_code.endswith("\n"):
            new_code += "\n"
        new_lines = new_code.splitlines(keepends=True)
        lines[func.start_line - 1 : func.end_line] = new_lines
    return "".join(lines)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def _process_file_source(
    source: str,
    all_forking_paths: Set[str],
    context_msg: str,
    client: Any,
    config: "CrispenConfig",
    max_attempts: int,
    scan_file: str = "",
    repo_root: Optional[str] = None,
) -> Tuple[str, bool, Dict[str, Dict[str, str]]]:
    """Scan *source* for @patch functions matching *all_forking_paths* and update.

    Returns ``(updated_source, was_changed, cross_file_patch_maps)`` where
    *cross_file_patch_maps* maps absolute file path → {old_string: new_string}
    for constant definitions in other files that all scan-file functions agree
    should be updated.
    """
    functions = _find_test_functions_to_update(
        source, all_forking_paths, scan_file, repo_root
    )
    if not functions:
        return source, False, {}

    all_updates: Dict[str, str] = {}

    for i in range(0, len(functions), _CHUNK_SIZE):
        chunk = functions[i : i + _CHUNK_SIZE]
        prev_issues: Optional[List[dict]] = None
        attempts_left = max_attempts

        while attempts_left > 0:
            attempts_left -= 1

            rewrite_prompt = _build_rewrite_prompt(context_msg, chunk, prev_issues)
            r = call_with_tool(
                client,
                config.provider,
                config.model,
                4096,
                _PATCH_REWRITE_TOOL,
                "update_patch_strings",
                [{"role": "user", "content": rewrite_prompt}],
                caller="patch_rewriter",
                tool_choice_override=config.tool_choice,
            )
            if r.tool_input is None:
                break  # LLM did not invoke the tool; skip chunk

            raw_updates = r.tool_input.get("updates", [])
            proposed: Dict[str, str] = {
                u["function_name"]: u["updated_code"]
                for u in raw_updates
                if isinstance(u, dict) and "function_name" in u and "updated_code" in u
            }

            # Syntax-check each proposed update.
            invalid = [name for name, code in proposed.items() if not _compiles(code)]
            if invalid:
                prev_issues = [
                    {"function_name": n, "issue": "syntax error in updated code"}
                    for n in invalid
                ]
                continue  # retry

            # LLM verify step.
            v = call_with_tool(
                client,
                config.provider,
                config.model,
                1024,
                _PATCH_VERIFY_TOOL,
                "verify_patch_updates",
                [
                    {
                        "role": "user",
                        "content": _build_verify_prompt(context_msg, chunk, proposed),
                    }
                ],
                caller="patch_rewriter",
                tool_choice_override=config.tool_choice,
            )
            if v.tool_input is None:
                # Verify call failed; accept proposed updates as-is.
                all_updates.update(proposed)
                break

            if v.tool_input.get("correct", False):
                all_updates.update(proposed)
                break
            else:
                issues = v.tool_input.get("issues", [])
                if attempts_left > 0:
                    prev_issues = issues
                # else: retries exhausted — skip this chunk

    # Post-process: find constant definitions to update in addition to inlining.
    # Functions with const_refs are always inlined with the LLM's new string values.
    # When every function using a given constant agrees on the same new value, we
    # also update the constant's definition (same-file or cross-file) as a bonus.
    same_file_patch_map: Dict[str, str] = {}
    cross_file_patch_maps: Dict[str, Dict[str, str]] = {}

    if all_updates and scan_file:
        scan_file_abs = str(Path(scan_file).resolve())
        per_const_proposals: Dict[Tuple[str, str], Set[str]] = {}
        per_const_old_val: Dict[Tuple[str, str], str] = {}

        for func in functions:
            if not func.const_refs:
                continue
            updated_code = all_updates.get(func.function_name)
            if updated_code is None:
                continue
            patch_args = _extract_patch_args_from_code(updated_code)
            for ref in func.const_refs:
                if ref.patch_dec_idx < len(patch_args):
                    new_val = patch_args[ref.patch_dec_idx]
                    if new_val is not None:
                        key = (ref.const_name, ref.source_file)
                        per_const_proposals.setdefault(key, set()).add(new_val)
                        per_const_old_val[key] = ref.resolved_value

        for (const_name, const_file), new_vals in per_const_proposals.items():
            if len(new_vals) != 1:
                continue  # diverging proposals — inline only, no const update
            new_val = next(iter(new_vals))
            old_val = per_const_old_val[(const_name, const_file)]
            if new_val == old_val:
                continue  # no actual change needed
            if const_file == scan_file_abs:
                same_file_patch_map[old_val] = new_val
            else:
                cross_file_patch_maps.setdefault(const_file, {})[old_val] = new_val

    if not all_updates:
        return source, False, cross_file_patch_maps

    updated = _apply_function_updates(source, functions, all_updates)
    if same_file_patch_map:
        updated = apply_patch_strings(updated, same_file_patch_map)
    return updated, updated != source, cross_file_patch_maps


# ---------------------------------------------------------------------------
# Cross-file constant definition update
# ---------------------------------------------------------------------------


def _apply_cross_file_const_updates(
    cross_file_proposals: Dict[str, Dict[str, Set[str]]],
    per_file: Dict[str, Any],
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
            continue

        # Disk file.
        try:
            old_src = Path(abs_file).read_text(encoding="utf-8")
        except OSError:
            continue
        new_src = apply_patch_strings(old_src, resolved)
        if new_src != old_src:
            Path(abs_file).write_text(new_src, encoding="utf-8")
            yield (
                f"{abs_file}: patch_update: "
                "updated @patch constant definition (rewrite)"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def apply_patch_rewrite(
    fl_contexts: List[_FLContext],
    per_file: Dict[str, Any],
    repo_root: Optional[str],
    config: "CrispenConfig",
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
        )
        if changed:
            state["source"] = new_src
            state["msgs"].append(
                f"{filepath}: patch_update: updated @patch strings (rewrite)"
            )
        for abs_file, patch_map in cross.items():
            for old_val, new_val in patch_map.items():
                cross_file_proposals.setdefault(abs_file, {}).setdefault(
                    old_val, set()
                ).add(new_val)

    if repo_root is None:
        yield from _apply_cross_file_const_updates(cross_file_proposals, per_file)
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
        )
        if changed:
            py_file.write_text(new_src, encoding="utf-8")
            yield f"{py_file}: patch_update: updated @patch strings (rewrite)"
        for abs_file, patch_map in cross.items():
            for old_val, new_val in patch_map.items():
                cross_file_proposals.setdefault(abs_file, {}).setdefault(
                    old_val, set()
                ).add(new_val)

    yield from _apply_cross_file_const_updates(cross_file_proposals, per_file)
