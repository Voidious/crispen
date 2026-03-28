"""LLM-powered @patch string rewriter for FileLimiter 'rewrite' mode."""

from __future__ import annotations

import ast
import difflib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TYPE_CHECKING, Union

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from .llm_client import call_with_tool, get_api_key, make_client
from .patch_updater import apply_patch_strings

if TYPE_CHECKING:
    from .config import CrispenConfig  # pragma: no cover

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


@dataclass
class RewriteAccumulator:
    """Mutable accumulator for timing and edit counts during patch rewrite."""

    calls: int = 0
    elapsed: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    files_updated: int = 0


_PATCH_RULES = (
    "\n## Rules for updating patch() strings:\n"
    "**Core principle:** `@patch('A.B.Name')` replaces the `Name` attribute in "
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
    "namespace. Patching `old_module.Name` does NOT affect F's lookup of `Name` "
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

_PATCH_CLASSIFY_TOOL: dict = {
    "name": "classify_patch_updates",
    "description": (
        "For each @patch string listed, provide its correct new value after the "
        "source file was split. Also flag if the test function needs structural "
        "changes beyond path renames (new @patch decorators, new mock parameters, "
        "or body changes)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "patch_renames": {
                "type": "object",
                "description": (
                    "Map of old patch string → new patch string. Provide an entry "
                    "for every patch path that was evaluated. Use the SAME string "
                    "as the value if no rename is needed for that path."
                ),
                "additionalProperties": {"type": "string"},
            },
            "needs_rewrite": {
                "type": "boolean",
                "description": (
                    "True if the function additionally requires structural changes "
                    "— new @patch decorators, new mock parameters, or body edits. "
                    "False if only path renames (or no change) are needed."
                ),
            },
        },
        "required": ["patch_renames", "needs_rewrite"],
    },
}

_PATCH_REWRITE_FUNC_TOOL: dict = {
    "name": "rewrite_test_function",
    "description": (
        "Produce a complete rewritten test function after a source file was split "
        "into sub-modules. The rewrite may add new @patch decorators, new mock "
        "parameters, and setup code as needed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "rewritten_function": {
                "type": "string",
                "description": (
                    "The complete rewritten test function including all decorators, "
                    "the def signature, and the full body. Must be valid Python and "
                    "preserve all original test logic."
                ),
            }
        },
        "required": ["rewritten_function"],
    },
}

_PATCH_REWRITE_VERIFY_TOOL: dict = {
    "name": "verify_rewrite",
    "description": (
        "Verify whether a rewritten test function is correct after a source "
        "file was split into sub-modules."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "correct": {
                "type": "boolean",
                "description": (
                    "True if the rewritten function correctly updates all @patch "
                    "strings, the mock parameters match their decorators, and all "
                    "original test logic is preserved without hallucinated code."
                ),
            },
            "issue": {
                "type": "string",
                "description": (
                    "What is wrong with the rewritten function. "
                    "Empty string when correct."
                ),
            },
        },
        "required": ["correct", "issue"],
    },
}

_PATCH_SINGLE_VERIFY_TOOL: dict = {
    "name": "verify_patch_update",
    "description": (
        "Verify whether proposed patch() string updates are correct after a "
        "source file was split into sub-modules."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "correct": {
                "type": "boolean",
                "description": "True if all proposed updates are correct.",
            },
            "corrections": {
                "type": "object",
                "description": (
                    "When correct=false because a module path changed, map each "
                    "current patch string to its corrected value — for example, "
                    '{"pkg.mod.Name": "pkg.mod.sub.Name"}. '
                    "Set to empty dict only when the fix requires adding a new "
                    "@patch decorator or mock parameter (not a path rename)."
                ),
                "additionalProperties": {"type": "string"},
            },
            "issue": {
                "type": "string",
                "description": (
                    "What is wrong with the proposed updates. "
                    "Empty string when correct."
                ),
            },
        },
        "required": ["correct", "corrections", "issue"],
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
# Body scan: collect patch() paths from ``with patch(...)`` context managers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LLM prompt builders
# ---------------------------------------------------------------------------


def _extract_migration_reminder(context_msg: str) -> str:
    """Pull entity migration bullets from context_msg for re-inclusion near instruction.

    The entity migration table appears near the top of the context message, which
    may be tens of thousands of tokens away from where the model generates its
    response.  Repeating a compact version at the end of the prompt keeps the
    relevant lookup table in the model's immediate attention window.
    """
    lines: List[str] = []
    capturing = False
    for line in context_msg.splitlines():
        stripped = line.rstrip()
        if stripped == "### Entity migration:":
            capturing = True
            continue
        if capturing:
            if stripped.startswith("#"):
                capturing = False
            elif stripped.startswith("- "):
                lines.append(stripped)
    if not lines:
        return ""
    return "## Entity migration (quick reference):\n" + "\n".join(lines) + "\n\n"


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


def _build_classify_prompt(
    context_msg: str,
    function_text: str,
    old_patch_paths: List[str],
    prev_issue: Optional[str] = None,
    prev_proposed: Optional[str] = None,
) -> str:
    """Build the user prompt for the per-function classify LLM call."""
    paths_list = "\n".join(f"- `{p}`" for p in old_patch_paths)
    migration_reminder = _extract_migration_reminder(context_msg)
    parts = [context_msg, _PATCH_RULES]
    parts.append(
        f"\n## Test function:\n```python\n{function_text}\n```\n\n"
        f"## Patch strings to evaluate:\n{paths_list}\n\n"
    )
    if migration_reminder:
        parts.append(migration_reminder)
    parts.append(
        "For **each** patch string, work through these steps:\n"
        "1. Identify the production function F this test exercises "
        "(look at what the test calls or patches).\n"
        "2. Look up F in the Entity migration quick reference above.\n"
        "   - If F was **not migrated**: the patch string is unchanged — "
        "include it in `patch_renames` with the **same** value.\n"
        "   - If F was **migrated to new module M**: go to step 3.\n"
        "3. Find the patched name (last component, e.g. `MetadataWrapper` in "
        "`crispen.engine.MetadataWrapper`). Search module M's source in the "
        "context above for an import of that name "
        "(e.g. `from libcst.metadata import MetadataWrapper`).\n"
        "   - If M imports the name: update the patch string to `M.Name`.\n"
        "   - If M does **not** import it: the patch string is unchanged.\n"
        "Include **every** evaluated path in `patch_renames`. "
        "Set `needs_rewrite` to True only for structural changes "
        "(new decorators, new mock parameters, or body edits).\n"
    )
    if prev_issue:
        parts.append(
            f"\n## Previous attempt was rejected:\n"
            f"- You previously proposed: {prev_proposed}\n"
            f"- Why it was wrong: {prev_issue}\n"
            f"- Output a corrected classification that fixes this issue.\n"
        )
    return "".join(parts)


def _build_func_verify_prompt(
    context_msg: str,
    function_text: str,
    patch_renames: Dict[str, str],
) -> str:
    """Build the user prompt for a per-function rename verify LLM call."""
    rename_lines = "\n".join(
        f"- `{old}` → `{new}`" for old, new in patch_renames.items()
    )
    parts = [
        context_msg,
        _PATCH_RULES,
        f"\n## Test function:\n```python\n{function_text}\n```\n\n"
        f"## Proposed patch() string updates:\n{rename_lines}\n\n"
        "Are all these updates correct? Set `correct` to True only if every "
        "proposed patch string points to where the name is looked up after "
        "the split.\n",
    ]
    return "".join(parts)


def _build_no_change_verify_prompt(
    context_msg: str,
    function_text: str,
) -> str:
    """Build the user prompt for a no-change verify LLM call."""
    parts = [
        context_msg,
        _PATCH_RULES,
        f"\n## Test function:\n```python\n{function_text}\n```\n\n"
        "The proposed update is: **no patch strings need changing**.\n\n"
        "Is this correct? Set `correct` to True only if all @patch strings in "
        "this function still point to the correct location after the split.\n"
        "If not correct, set `corrections` to map each @patch string that needs "
        "updating to its corrected path (e.g., "
        '{"crispen.engine.X": "crispen.engine.sub.X"}). '
        "Only leave `corrections` empty when the fix requires adding a new "
        "@patch decorator or mock parameter, not just updating an existing path.\n",
    ]
    return "".join(parts)


def _build_rewrite_func_prompt(
    context_msg: str,
    function_text: str,
    old_patch_paths: List[str],
    prev_error: Optional[str] = None,
) -> str:
    """Build the user prompt for the full function rewrite LLM call."""
    paths_list = "\n".join(f"- `{p}`" for p in old_patch_paths)
    parts = [context_msg, _PATCH_RULES]
    if prev_error:
        parts.append(f"\n## Previous rewrite was invalid:\n" f"- Error: {prev_error}\n")
    parts.append(
        f"\n## Test function to rewrite:\n```python\n{function_text}\n```\n\n"
        f"## Patch strings that need updating:\n{paths_list}\n\n"
        "Rewrite the complete function. You may add new @patch decorators with "
        "corresponding mock parameters and setup code. Preserve all original "
        "test logic. Return the complete function including all decorators and "
        "body.\n"
    )
    return "".join(parts)


def _build_rewrite_verify_prompt(
    context_msg: str,
    original_function_text: str,
    rewritten_function_text: str,
) -> str:
    """Build the user prompt for a full-rewrite verify LLM call."""
    parts = [
        context_msg,
        _PATCH_RULES,
        f"\n## Original test function:\n```python\n{original_function_text}\n```\n\n"
        f"## Rewritten test function:\n```python\n{rewritten_function_text}\n```\n\n"
        "Verify that the rewrite is correct:\n"
        "- All @patch strings point to where the name is looked up after the split.\n"
        "- All mock parameters correspond correctly to their @patch decorators "
        "(order, count, and names).\n"
        "- All original test logic is preserved — no hallucinated code, no "
        "missing assertions or setup.\n"
        "Set `correct` to True only if all of the above are satisfied.\n",
    ]
    return "".join(parts)


# ---------------------------------------------------------------------------
# Function splice helper
# ---------------------------------------------------------------------------


def _splice_function(
    source: str, start_line: int, end_line: int, new_func_text: str
) -> str:
    """Replace lines start_line..end_line (1-indexed, inclusive) with new_func_text."""
    lines = source.splitlines(True)
    if new_func_text and not new_func_text.endswith("\n"):
        new_func_text += "\n"
    new_lines = new_func_text.splitlines(True)
    lines[start_line - 1 : end_line] = new_lines
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
    verbose: bool = False,
    _acc: Optional[RewriteAccumulator] = None,
) -> Tuple[str, bool, Dict[str, Dict[str, str]]]:
    """Scan *source* for @patch functions matching *all_forking_paths* and update.

    Processes one function at a time using a two-stage approach: first classify
    whether the function needs a full rewrite or simple string renames, then act
    accordingly.  Returns ``(updated_source, was_changed, cross_file_patch_maps)``.
    """
    functions = _find_test_functions_to_update(
        source, all_forking_paths, scan_file, repo_root
    )
    if not functions:
        return source, False, {}

    file_desc = f"'{scan_file}'" if scan_file else "file"
    source_lines = source.splitlines()

    # (start_line, end_line, new_text) splices for both string-swap and full rewrites.
    func_splices: List[Tuple[int, int, str]] = []
    # Track string-swap funcs and their accepted renames for cross-file const handling.
    string_swap_results: List[Tuple[_TestFunctionInfo, Dict[str, str]]] = []
    # Same-file const definition updates (applied after all splices).
    same_file_const_map: Dict[str, str] = {}

    for func in functions:
        prev_issue: Optional[str] = None
        prev_proposed: Optional[str] = None
        attempts_left = max_attempts
        rename_verify_retries_left = config.llm_verify_retries
        # When the no-change verify path exhausts retries we escalate: skip
        # the next classify call and go directly to the full rewrite path,
        # seeding it with the verifier's explanation as the initial prev_error.
        _rewrite_escalation_error: Optional[str] = None
        r = None  # may be skipped when escalating

        while attempts_left > 0:
            attempts_left -= 1

            if _rewrite_escalation_error is not None:
                # Bypass classify; the no-change verify already identified the
                # problem — hand the explanation straight to the rewrite path.
                needs_rewrite = True
                if verbose:
                    print(
                        f"crispen: patch_rewriter: escalating to rewrite for"
                        f" '{func.function_name}' in {file_desc}"
                        f" (no-change verify retries exhausted)",
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                # Stage 1: Classify (and get renames if string-swap).
                classify_prompt = _build_classify_prompt(
                    context_msg,
                    func.full_text,
                    func.old_patch_paths,
                    prev_issue,
                    prev_proposed,
                )
                retry_label = " (retry)" if prev_issue is not None else ""
                if verbose:
                    print(
                        f"crispen: patch_rewriter: classifying '{func.function_name}'"
                        f" in {file_desc}{retry_label}",
                        file=sys.stderr,
                        flush=True,
                    )
                r = call_with_tool(
                    client,
                    config.provider,
                    config.model,
                    512,
                    _PATCH_CLASSIFY_TOOL,
                    "classify_patch_updates",
                    [{"role": "user", "content": classify_prompt}],
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

                needs_rewrite = r.tool_input.get("needs_rewrite", False)

            if needs_rewrite:
                # Stage 2: Full function rewrite.
                rewrite_attempts = max_attempts
                rewrite_verify_retries_left = config.llm_verify_retries
                prev_error: Optional[str] = _rewrite_escalation_error
                _rewrite_escalation_error = None
                while rewrite_attempts > 0:
                    rewrite_attempts -= 1
                    rewrite_prompt = _build_rewrite_func_prompt(
                        context_msg,
                        func.full_text,
                        func.old_patch_paths,
                        prev_error,
                    )
                    if verbose:
                        retry_rw = " (retry)" if prev_error is not None else ""
                        print(
                            f"crispen: patch_rewriter: rewriting"
                            f" '{func.function_name}' in {file_desc}{retry_rw}",
                            file=sys.stderr,
                            flush=True,
                        )
                    rw = call_with_tool(
                        client,
                        config.provider,
                        config.model,
                        2048,
                        _PATCH_REWRITE_FUNC_TOOL,
                        "rewrite_test_function",
                        [{"role": "user", "content": rewrite_prompt}],
                        caller="patch_rewriter",
                        tool_choice_override=config.tool_choice,
                    )
                    if _acc is not None:
                        _acc.calls += 1
                        _acc.elapsed += rw.elapsed
                        _acc.input_tokens += rw.input_tokens
                        _acc.output_tokens += rw.output_tokens
                    if verbose and config.timing == "detailed":
                        print(
                            f"crispen: patch_rewriter:   → done [{rw.elapsed:.2f}s,"
                            f" {rw.input_tokens:,} in / {rw.output_tokens:,} out]",
                            file=sys.stderr,
                            flush=True,
                        )
                    if rw.tool_input is None:
                        break
                    new_func_text = rw.tool_input.get("rewritten_function", "")
                    if not isinstance(new_func_text, str) or not new_func_text.strip():
                        break
                    if not _compiles(new_func_text):
                        prev_error = "Rewritten function is not valid Python."
                        continue
                    # LLM verify step.
                    rewrite_verify_prompt = _build_rewrite_verify_prompt(
                        context_msg, func.full_text, new_func_text
                    )
                    if verbose:
                        print(
                            f"crispen: patch_rewriter: verifying rewrite for"
                            f" '{func.function_name}' in {file_desc}",
                            file=sys.stderr,
                            flush=True,
                        )
                    rv = call_with_tool(
                        client,
                        config.provider,
                        config.model,
                        256,
                        _PATCH_REWRITE_VERIFY_TOOL,
                        "verify_rewrite",
                        [{"role": "user", "content": rewrite_verify_prompt}],
                        caller="patch_rewriter",
                        tool_choice_override=config.tool_choice,
                    )
                    if _acc is not None:
                        _acc.calls += 1
                        _acc.elapsed += rv.elapsed
                        _acc.input_tokens += rv.input_tokens
                        _acc.output_tokens += rv.output_tokens
                    if verbose and config.timing == "detailed":
                        print(
                            f"crispen: patch_rewriter:   → done [{rv.elapsed:.2f}s,"
                            f" {rv.input_tokens:,} in / {rv.output_tokens:,} out]",
                            file=sys.stderr,
                            flush=True,
                        )
                    rv_correct = rv.tool_input is None or rv.tool_input.get(
                        "correct", False
                    )
                    if verbose and rv.tool_input is not None:
                        rv_status = "ACCEPTED" if rv_correct else "REJECTED"
                        print(
                            f"crispen: patch_rewriter: rewrite verify {rv_status}",
                            file=sys.stderr,
                            flush=True,
                        )
                        if not rv_correct and rv.tool_input.get("issue"):
                            print(
                                f"crispen: patch_rewriter:   issue:"
                                f" {rv.tool_input.get('issue')}",
                                file=sys.stderr,
                                flush=True,
                            )
                    if rv_correct:
                        func_splices.append(
                            (func.start_line, func.end_line, new_func_text)
                        )
                        if verbose:
                            print(
                                f"crispen: patch_rewriter: rewrote"
                                f" '{func.function_name}'",
                                file=sys.stderr,
                                flush=True,
                            )
                        break
                    rv_issue = rv.tool_input.get("issue", "") if rv.tool_input else ""
                    if rewrite_verify_retries_left > 0:
                        rewrite_verify_retries_left -= 1
                        prev_error = rv_issue or "LLM verify rejected the rewrite."
                        rewrite_attempts += 1  # don't burn compile retry budget
                        continue
                    # verify retries exhausted — skip this function
                break  # done with this function (rewrite handled above)

            # String-swap: validate and filter renames.
            raw_renames = r.tool_input.get("patch_renames") or {}
            if not isinstance(raw_renames, dict):
                raw_renames = {}
            patch_renames: Dict[str, str] = {
                old: new
                for old, new in raw_renames.items()
                if (
                    isinstance(old, str)
                    and isinstance(new, str)
                    and old != new
                    and old in func.old_patch_paths
                )
            }

            if not patch_renames:
                # Verify the "no change needed" conclusion.
                no_change_verify_prompt = _build_no_change_verify_prompt(
                    context_msg, func.full_text
                )
                if verbose:
                    print(
                        f"crispen: patch_rewriter: verifying no-change for"
                        f" '{func.function_name}' in {file_desc}",
                        file=sys.stderr,
                        flush=True,
                    )
                v = call_with_tool(
                    client,
                    config.provider,
                    config.model,
                    512,
                    _PATCH_SINGLE_VERIFY_TOOL,
                    "verify_patch_update",
                    [{"role": "user", "content": no_change_verify_prompt}],
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
                    break  # verify call failed; accept no-change
                verify_correct = v.tool_input.get("correct", False)
                issue = v.tool_input.get("issue", "")
                if verbose:
                    v_status = "ACCEPTED" if verify_correct else "REJECTED"
                    print(
                        f"crispen: patch_rewriter: no-change verify {v_status}",
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
                    break  # confirmed: no change needed
                # Check if verifier provided corrections for a direct apply.
                corrections = v.tool_input.get("corrections") or {}
                corrections_renames: Dict[str, str] = {
                    old: new
                    for old, new in corrections.items()
                    if (
                        isinstance(old, str)
                        and isinstance(new, str)
                        and old != new
                        and old in func.old_patch_paths
                    )
                }
                if corrections_renames:
                    # Verifier provided corrections — verify before applying.
                    vc_prompt = _build_func_verify_prompt(
                        context_msg, func.full_text, corrections_renames
                    )
                    if verbose:
                        print(
                            f"crispen: patch_rewriter: verifying corrections for"
                            f" '{func.function_name}' in {file_desc}",
                            file=sys.stderr,
                            flush=True,
                        )
                    vc = call_with_tool(
                        client,
                        config.provider,
                        config.model,
                        256,
                        _PATCH_SINGLE_VERIFY_TOOL,
                        "verify_patch_update",
                        [{"role": "user", "content": vc_prompt}],
                        caller="patch_rewriter",
                        tool_choice_override=config.tool_choice,
                    )
                    if _acc is not None:
                        _acc.calls += 1
                        _acc.elapsed += vc.elapsed
                        _acc.input_tokens += vc.input_tokens
                        _acc.output_tokens += vc.output_tokens
                    if verbose and config.timing == "detailed":
                        print(
                            f"crispen: patch_rewriter:   → done [{vc.elapsed:.2f}s,"
                            f" {vc.input_tokens:,} in / {vc.output_tokens:,} out]",
                            file=sys.stderr,
                            flush=True,
                        )
                    vc_correct = vc.tool_input is None or vc.tool_input.get(
                        "correct", False
                    )
                    if verbose and vc.tool_input is not None:
                        vc_status = "ACCEPTED" if vc_correct else "REJECTED"
                        print(
                            f"crispen: patch_rewriter: corrections verify {vc_status}",
                            file=sys.stderr,
                            flush=True,
                        )
                        if not vc_correct and vc.tool_input.get("issue"):
                            print(
                                f"crispen: patch_rewriter:   issue:"
                                f" {vc.tool_input.get('issue')}",
                                file=sys.stderr,
                                flush=True,
                            )
                    if vc_correct:
                        orig_text = "\n".join(
                            source_lines[func.start_line - 1 : func.end_line]
                        )
                        new_text = apply_patch_strings(orig_text, corrections_renames)
                        if new_text != orig_text:
                            func_splices.append(
                                (func.start_line, func.end_line, new_text)
                            )
                        string_swap_results.append((func, corrections_renames))
                        break
                    # Corrections verify rejected; update issue for retry/escalation.
                    issue = vc.tool_input.get("issue", "") or issue
                if rename_verify_retries_left > 0:
                    rename_verify_retries_left -= 1
                    prev_issue = issue
                    no_change_paths = ", ".join(f"`{p}`" for p in func.old_patch_paths)
                    prev_proposed = (
                        str(corrections_renames)
                        if corrections_renames
                        else f"no change (kept {no_change_paths} unchanged)"
                    )
                    attempts_left += 1  # don't burn classify retry budget
                    continue
                # Retries exhausted.  When verify_retries were enabled the
                # verify step consistently identified a required change that
                # the classify path could not produce — escalate to the full
                # rewrite path, seeding it with the verifier's explanation.
                if config.llm_verify_retries > 0:
                    _rewrite_escalation_error = issue
                    attempts_left += 1  # allow one more outer iteration
                    continue
                break  # llm_verify_retries=0 → accept no-change

            # Verify the renames.
            verify_prompt = _build_func_verify_prompt(
                context_msg, func.full_text, patch_renames
            )
            if verbose:
                print(
                    f"crispen: patch_rewriter: verifying renames for"
                    f" '{func.function_name}' in {file_desc}",
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
                # Verify failed; accept renames.
                orig_text = "\n".join(source_lines[func.start_line - 1 : func.end_line])
                new_text = apply_patch_strings(orig_text, patch_renames)
                if new_text != orig_text:
                    func_splices.append((func.start_line, func.end_line, new_text))
                string_swap_results.append((func, patch_renames))
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
                orig_text = "\n".join(source_lines[func.start_line - 1 : func.end_line])
                new_text = apply_patch_strings(orig_text, patch_renames)
                if new_text != orig_text:
                    func_splices.append((func.start_line, func.end_line, new_text))
                string_swap_results.append((func, patch_renames))
                break
            else:
                if rename_verify_retries_left > 0:
                    rename_verify_retries_left -= 1
                    prev_issue = issue
                    prev_proposed = str(patch_renames)
                    attempts_left += 1  # don't burn classify retry budget
                elif config.llm_verify_retries > 0:
                    _rewrite_escalation_error = issue
                    attempts_left += 1  # allow one more outer iteration
                # else (llm_verify_retries=0): retries exhausted — skip

    # Collect cross-file and same-file constant definition updates.
    cross_file_patch_maps: Dict[str, Dict[str, str]] = {}
    if string_swap_results and scan_file:
        scan_file_abs = str(Path(scan_file).resolve())
        for func, accepted in string_swap_results:
            for ref in func.const_refs:
                new_val = accepted.get(ref.resolved_value)
                if new_val is None or new_val == ref.resolved_value:
                    continue
                if ref.source_file == scan_file_abs:
                    same_file_const_map[ref.resolved_value] = new_val
                else:
                    cross_file_patch_maps.setdefault(ref.source_file, {})[
                        ref.resolved_value
                    ] = new_val

    if not func_splices and not same_file_const_map:
        return source, False, cross_file_patch_maps

    # Apply splices (bottom to top to preserve line indices).
    result_source = source
    for start_line, end_line, new_text in sorted(func_splices, key=lambda x: -x[0]):
        result_source = _splice_function(result_source, start_line, end_line, new_text)

    # Apply same-file constant definition updates.
    if same_file_const_map:
        result_source = apply_patch_strings(result_source, same_file_const_map)

    return result_source, result_source != source, cross_file_patch_maps


# ---------------------------------------------------------------------------
# Cross-file constant definition update
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
