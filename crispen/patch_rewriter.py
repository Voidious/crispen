"""LLM-powered @patch string rewriter for FileLimiter 'rewrite' mode."""

from __future__ import annotations

import ast
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
    "1. Find the patched name **N** (last component of the patch string, "
    "e.g. `call_with_tool` in `pkg.module.call_with_tool`).\n"
    "2. Inspect the **Modified original file** source in the context. "
    "Scan its import lines for N (e.g. `import N` or `from ... import N` or "
    "`from ... import N as ...`).\n"
    "   - If N is **NOT imported** in the modified original file: "
    "it was removed from that module during the split. "
    "Scan the **Imports** section under each new file in the context for N. "
    "The new file whose imports include N is the new resolution point — "
    "update the patch string to `new_module_path.N`. Done.\n"
    "   - If N **IS imported** in the modified original file: go to step 3.\n"
    "3. Identify the production function **F** being tested "
    "(look at what the test calls or constructs).\n"
    "4. Look up **F** in the entity migration table:\n"
    "   - If F was **not migrated**: N is still resolved in the original module "
    "when F runs. Leave the patch unchanged — even if the lookup section "
    "shows N is 'also imported in' a new submodule.\n"
    "   - If F was **migrated to new module M**: before updating, "
    "check M's **Imports** section in the context and confirm N appears "
    "there. If N is NOT listed in M's imports, do NOT update the patch "
    "to `M.N` — leave the patch pointing to the module that still "
    "externally imports N (i.e. the original module). "
    "If N IS listed in M's imports, update the patch to `M.N`.\n"
    "5. **Re-exports do NOT count as imports for step 2.** "
    "If `old_module/__init__.py` has `from .submodule import N` "
    "that is a re-export, not a true import into the original module's own "
    "resolution. Look for N in lines that import it from an external source "
    "(e.g. `from ...llm_client import N`, `from third_party import N`). "
    "If the only mention of N in the original file is via a re-export from a "
    "new submodule, treat N as NOT imported in the original file and apply "
    "the step 2 'not imported' branch.\n"
    "6. **from-import pitfall:** When M does "
    "`from libcst.metadata import MetadataWrapper`,"
    " it creates an INDEPENDENT local binding in M's namespace. Patching "
    "`libcst.metadata.MetadataWrapper` does NOT affect that local binding — only "
    "`M.MetadataWrapper` intercepts calls made inside M. "
    "The source of the import is irrelevant to the patch target.\n\n"
    "**Package constraint:** A file split only moves entities within the same project "
    "package. The top-level package (first path component) NEVER changes in a rename. "
    "e.g. `crispen.engine.X` always renames to another `crispen.*` path — "
    "a rename to `libcst.metadata.X` is ALWAYS wrong.\n\n"
    "**Example A — N no longer imported in original file (most common):**\n"
    "```python\n"
    "# Before split: crispen/engine.py imports call_with_tool and _apply_foo()\n"
    "# (both defined/imported in same module)\n"
    "@patch('crispen.engine.call_with_tool')  # correct before split\n\n"
    "# After split: crispen/engine/__init__.py (modified original) — "
    "does NOT import call_with_tool\n"
    "#              crispen/engine/helpers.py (new file) — "
    "imports call_with_tool: from ...llm_client import call_with_tool\n"
    "# Step 2: call_with_tool not in __init__.py imports → moved to helpers\n"
    "@patch('crispen.engine.helpers.call_with_tool')  # correct after split\n"
    "# WRONG: @patch('crispen.engine.call_with_tool')  "
    "-- not imported in __init__ anymore\n"
    "```\n\n"
    "**Example B — N still in original file, F migrated:**\n"
    "```python\n"
    "# Before split: _apply_foo() defined in crispen.engine, imports MetadataWrapper\n"
    "@patch('crispen.engine.MetadataWrapper')  # correct before split\n\n"
    "# After split: MetadataWrapper IS imported in crispen/engine/__init__.py\n"
    "# _apply_foo() moved to crispen.engine.helpers\n"
    "# Step 2: MetadataWrapper IS in __init__.py → step 3\n"
    "# Step 3/4: F=_apply_foo migrated to crispen.engine.helpers\n"
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
        "source file was split into sub-modules. When rejecting, populate "
        "corrections with {current_path: corrected_path} for each rename needed."
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
                    "When correct=false and the fix is a module path rename: "
                    "required mapping of {current_path: corrected_path} for "
                    "each string that needs renaming. For example: "
                    '{"crispen.engine.MetadataWrapper": '
                    '"crispen.engine.helpers.MetadataWrapper"}. '
                    "Leave empty only if the fix requires adding an entirely "
                    "new @patch decorator (not renaming an existing one)."
                ),
                "additionalProperties": {"type": "string"},
            },
            "issue": {
                "type": "string",
                "description": (
                    "Explain why the patch string is wrong (module that moved, "
                    "etc.). Do not include the corrected path here — "
                    "put that in corrections."
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
        # Collect ALL resolvable patch paths from decorators, tracking which match.
        # A function is only processed when at least one path matches old_paths, but
        # when triggered we send ALL patch paths to the LLM so it can evaluate every
        # @patch decorator — not just the ones that triggered collection.
        all_patch_paths: List[str] = []  # every resolvable decorator patch string
        all_const_refs: List[_ConstRef] = []  # const refs for every const @patch
        has_match = False  # True when at least one decorator path matches old_paths
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
                    all_patch_paths.append(inner)
                    if _matches_any(inner, self._old_paths):
                        has_match = True
            elif isinstance(arg0, cst.Name) and arg0.value in self._const_map:
                const_val, const_file = self._const_map[arg0.value]
                all_patch_paths.append(const_val)
                all_const_refs.append(
                    _ConstRef(
                        const_name=arg0.value,
                        source_file=const_file,
                        resolved_value=const_val,
                        patch_dec_idx=patch_dec_idx,
                    )
                )
                if _matches_any(const_val, self._old_paths):
                    has_match = True
            elif isinstance(arg0, cst.Attribute) and isinstance(arg0.value, cst.Name):
                module_alias = arg0.value.value
                attr_name = arg0.attr.value
                if module_alias in self._attr_const_map:
                    attr_map = self._attr_const_map[module_alias]
                    if attr_name in attr_map:
                        const_val, const_file = attr_map[attr_name]
                        all_patch_paths.append(const_val)
                        all_const_refs.append(
                            _ConstRef(
                                const_name=f"{module_alias}.{attr_name}",
                                source_file=const_file,
                                resolved_value=const_val,
                                patch_dec_idx=patch_dec_idx,
                            )
                        )
                        if _matches_any(const_val, self._old_paths):
                            has_match = True
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
        if body_paths:
            has_match = True

        if not has_match:
            return

        # Include ALL decorator patch paths (not just matching ones) so the LLM
        # evaluates every @patch in this function.  A single test may patch
        # get_api_key, make_client, and call_with_tool for the same migrated
        # function — each of those may need updating to a different sub-module.
        old_patch_paths = all_patch_paths + body_paths

        # Build the full_text sent to the LLM: substitute constant names with
        # their string values so the LLM always sees plain string literals.
        if all_const_refs:
            subs = {ref.const_name: ref.resolved_value for ref in all_const_refs}
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
                const_refs=all_const_refs,
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


def _extract_patch_lookup(context_msg: str) -> str:
    """Extract the pre-computed patch target lookup from context_msg.

    The lookup section lists which names moved to new sub-modules and which
    remain in the original file.  Repeating it near the classify instructions
    keeps the concrete name→module mapping in the model's immediate attention
    window so it does not need to scan the full context.
    """
    lines: List[str] = []
    capturing = False
    for line in context_msg.splitlines():
        stripped = line.rstrip()
        if stripped == "### Patch target lookup (pre-computed):":
            capturing = True
            continue
        if capturing:
            if stripped.startswith("##"):
                capturing = False
            else:
                lines.append(stripped)
    if not lines:
        return ""
    return "### Patch target lookup (pre-computed):\n" + "\n".join(lines) + "\n\n"


def _extract_still_imported_names(context_msg: str) -> Set[str]:
    """Extract names listed as 'still externally imported' in context_msg.

    These names remain directly imported by the modified original file, so any
    no-change verifier correction that tries to relocate them is a hallucination
    and should be filtered out before applying.
    """
    names: Set[str] = set()
    in_section = False
    for line in context_msg.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("Names still externally imported"):
            in_section = True
            continue
        if in_section:
            if not stripped.startswith("- `"):
                in_section = False
                continue
            end = stripped.find("`", 3)
            if end > 3:
                names.add(stripped[3:end])
    return names


def _extract_moved_out_names(context_msg: str) -> Set[str]:
    """Extract names listed as removed from the modified original in context_msg.
    These names are no longer in the original module after the split.  Any rename
    that tries to *shallow* a patch path for one of these names (reducing module
    depth) would produce a path that doesn't exist — it must be blocked.
    """
    names: Set[str] = set()
    in_section = False
    for line in context_msg.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("Names REMOVED from the modified original"):
            in_section = True
            continue
        if in_section:
            if not stripped.startswith("- `"):
                in_section = False
                continue
            end = stripped.find("`", 3)
            if end > 3:
                names.add(stripped[3:end])
    return names


def _extract_still_in_orig_users(context_msg: str) -> Dict[str, List[str]]:
    """Extract the 'used in original module by' annotations from still-imported names.

    Returns ``{name: [func1, func2, ...]}`` for each still-imported name that
    has an 'used in original module by:' annotation in context_msg.  These users
    are functions in the original (pre-split) module that call the name directly,
    so any @patch path targeting them must remain at the original module level.
    """
    result: Dict[str, List[str]] = {}
    in_section = False
    for line in context_msg.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("Names still externally imported"):
            in_section = True
            continue
        if in_section:
            if not stripped.startswith("- `"):
                in_section = False
                continue
            end_name = stripped.find("`", 3)
            if end_name <= 3:
                continue
            name = stripped[3:end_name]
            orig_prefix = "used in original module by: "
            if orig_prefix not in stripped:
                continue
            idx = stripped.index(orig_prefix) + len(orig_prefix)
            rest = stripped[idx:]
            if ";" in rest:
                rest = rest[: rest.index(";")]
            parts = rest.split("`")
            users = [parts[i] for i in range(1, len(parts), 2) if parts[i]]
            if users:
                result[name] = users
    return result


def _is_bad_rename(
    old: str,
    new: str,
    moved_out_names: Set[str],
    still_imported: Set[str],
    orig_users_map: Dict[str, List[str]],
    test_text: str,
) -> bool:
    """Return True if the rename (old → new) is a known-incorrect hallucination.

    Two patterns are blocked:

    A) **Shallowing a moved-out name**: the name is no longer in the original
       module, so a rename that *reduces* module depth (e.g.
       ``advisor.placement.call_with_tool`` → ``advisor.call_with_tool``) points
       at a non-existent binding and will raise AttributeError.

    B) **Deepening a still-in name whose original-module users appear in the test
       body**: the name is still in the original module AND the test calls a
       function (e.g. ``advise_file_limiter``) that uses the name at that level.
       Deepening the patch path (e.g. ``advisor.make_client`` →
       ``advisor.placement.make_client``) would miss the binding that the tested
       function actually resolves — another AttributeError.
    """
    name = old.rsplit(".", 1)[-1]
    old_depth = len(old.rsplit(".", 1)[0].split("."))
    new_depth = len(new.rsplit(".", 1)[0].split("."))
    # Pattern A: shallowing a moved-out name
    if name in moved_out_names and new_depth < old_depth:
        return True
    # Pattern B: deepening a still-in name when original-module callers are in test
    if name in still_imported and new_depth > old_depth and name in orig_users_map:
        for user in orig_users_map[name]:
            if user in test_text:
                return True
    return False


def _import_header(source: str) -> str:
    """Return the imports/header portion of a Python source file.

    Stops before the first top-level ``def``/``class``/``async def`` line so
    that function and class bodies are omitted.  Only the import declarations
    (including ``if TYPE_CHECKING:`` blocks) are needed by the LLM to determine
    which names are available in a module for patch-path resolution.
    """
    lines: List[str] = []
    for line in source.splitlines(keepends=True):
        if line.lstrip().startswith(("def ", "class ", "async def ")):
            break
        lines.append(line)
    while lines and not lines[-1].strip():
        lines.pop()
    return "".join(lines)


def _name_reference_map(source: str) -> Dict[str, List[str]]:
    """Return ``{imported_alias: [defn_names_that_use_it]}`` for *source*.

    For each top-level import alias (e.g. ``cst`` from ``import libcst as cst``,
    or ``DuplicateExtractor`` from a ``from … import`` statement) this collects
    which top-level function and class definitions reference that name.  The
    result lets the LLM determine which sub-module's local binding of a name is
    the correct patch target — it's the sub-module whose callers live there.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    # Collect all top-level imported aliases (the local name, i.e. what code uses).
    imported: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    imported.add(alias.asname if alias.asname else alias.name)

    # For each top-level function / class, record which imported names it uses.
    refs: Dict[str, List[str]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        defn_name = node.name
        seen: Set[str] = set()
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Name)
                and child.id in imported
                and child.id not in seen
            ):
                seen.add(child.id)
                refs.setdefault(child.id, []).append(defn_name)

    return refs


def _get_external_import_names(source: str) -> Set[str]:
    """Return names externally imported in *source*, excluding level-1 re-exports.

    Level-1 relative imports (``from .submodule import X``) are sibling re-exports
    introduced by a file split and do not represent true external dependencies.
    All other imports — absolute (``level=0``) and multi-level relative
    (``level>=2``) — are included as genuine external bindings.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 1:
                continue  # skip level-1 sibling re-exports
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname if alias.asname else alias.name)
    return names


def _build_context_message(fl_contexts: List[_FLContext]) -> str:
    """Build the shared LLM prompt context describing all split files."""
    parts: List[str] = [
        "A Python source file was split into multiple sub-modules by an automated "
        "refactoring tool.  Update the patch() call strings in the provided "
        "test functions so they reference the correct new module paths.\n"
    ]

    for ctx in fl_contexts:
        parts.append(f"\n## Split module: `{ctx.old_module}` ({ctx.filepath})\n")

        parts.append(
            f"### Modified original file `{ctx.old_module}` (current state):\n"
            "```python\n"
        )
        parts.append(ctx.modified_source)
        parts.append("```\n")

        for rel_path, content in ctx.new_files.items():
            new_mod = ctx.new_module_paths.get(rel_path, rel_path)
            parts.append(f"### New file `{rel_path}` (module: `{new_mod}`):\n")
            header = _import_header(content)
            if header:
                parts.append("**Imports:**\n```python\n")
                parts.append(header)
                parts.append("```\n")
            ref_map = _name_reference_map(content)
            if ref_map:
                parts.append(
                    "**Name references**"
                    " (imported alias → top-level definitions that use it):\n"
                )
                for name in sorted(ref_map):
                    callers = ref_map[name]
                    parts.append(
                        f"- `{name}`: {', '.join(f'`{c}`' for c in callers)}\n"
                    )

        parts.append("### Entity migration:\n")
        for entity_name in sorted(ctx.entity_to_target):
            target_rel = ctx.entity_to_target[entity_name]
            new_mod = ctx.new_module_paths.get(target_rel, target_rel)
            parts.append(f"- `{entity_name}` → `{target_rel}` (module: `{new_mod}`)\n")

        # Pre-compute which externally-imported names moved out of the original
        # module versus which still live there.  This spares the LLM from having
        # to scan thousands of tokens of source to answer that question.
        orig_ext = _get_external_import_names(ctx.original_source)
        mod_ext = _get_external_import_names(ctx.modified_source)
        moved_out = orig_ext - mod_ext
        still_in = orig_ext & mod_ext
        # Pre-compute ref_maps (imported name → using entities) for each new file
        # so multi-home names can be annotated with which entities use them.
        ref_maps_by_file: Dict[str, Dict[str, List[str]]] = {
            rel_path: _name_reference_map(content)
            for rel_path, content in ctx.new_files.items()
        }
        # Also compute a ref_map for the modified original so still_in entries
        # can be annotated with which original-module entities use each name.
        orig_mod_ref_map = _name_reference_map(ctx.modified_source)
        if moved_out or still_in:
            parts.append("### Patch target lookup (pre-computed):\n")
            if moved_out:
                parts.append(
                    "Names REMOVED from the modified original during the split"
                    " — patching at `original_module.name` WILL raise"
                    " AttributeError at test time:\n"
                )
                for name in sorted(moved_out):
                    homes: List[str] = []
                    for rel_path, content in ctx.new_files.items():
                        if name in _get_external_import_names(content):
                            new_mod = ctx.new_module_paths.get(rel_path, rel_path)
                            users = ref_maps_by_file.get(rel_path, {}).get(name, [])
                            if users:
                                homes.append(
                                    f"`{new_mod}` (used by: "
                                    + ", ".join(f"`{u}`" for u in users)
                                    + ")"
                                )
                            else:
                                homes.append(f"`{new_mod}`")
                    if homes:
                        parts.append(f"- `{name}` → {', '.join(sorted(homes))}\n")
                    else:
                        parts.append(f"- `{name}` → (not found in new files)\n")
            if still_in:
                parts.append(
                    "Names still externally imported in the modified original "
                    "(check entity migration to determine the correct patch target):\n"
                )
                for name in sorted(still_in):
                    orig_users = orig_mod_ref_map.get(name, [])
                    home_annotations: List[str] = []
                    for rp, content in ctx.new_files.items():
                        if name in _get_external_import_names(content):
                            new_mod = ctx.new_module_paths.get(rp, rp)
                            users = ref_maps_by_file.get(rp, {}).get(name, [])
                            if users:
                                home_annotations.append(
                                    f"`{new_mod}` (used by: "
                                    + ", ".join(f"`{u}`" for u in users)
                                    + ")"
                                )
                            else:
                                home_annotations.append(f"`{new_mod}`")
                    orig_note = (
                        "used in original module by: "
                        + ", ".join(f"`{u}`" for u in orig_users)
                        + "; "
                        if orig_users
                        else ""
                    )
                    if home_annotations:
                        parts.append(
                            f"- `{name}` — {orig_note}also externally imported in: "
                            + ", ".join(sorted(home_annotations))
                            + "; patch at that submodule only if the test's"
                            " F is one of the listed 'used by' functions\n"
                        )
                    else:
                        parts.append(
                            f"- `{name}` — {orig_note}NOT imported in any new"
                            " submodule\n"
                            if orig_note
                            else f"- `{name}` — NOT imported in any new submodule\n"
                        )

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
    patch_lookup = _extract_patch_lookup(context_msg)
    parts = [context_msg, _PATCH_RULES]
    parts.append(
        f"\n## Test function:\n```python\n{function_text}\n```\n\n"
        "## All @patch strings in this function (evaluate every one):\n"
        f"{paths_list}\n\n"
    )
    if migration_reminder:
        parts.append(migration_reminder)
    if patch_lookup:
        parts.append(patch_lookup)
    parts.append(
        "Every patch string listed above must be evaluated independently — "
        "a single test function may need **multiple** patch strings updated, "
        "each potentially pointing to a **different** new sub-module.\n\n"
    )
    if patch_lookup:
        parts.append(
            "**For each patch string:**\n"
            "1. Find N (the last component, "
            "e.g. `call_with_tool` in `pkg.module.call_with_tool`).\n"
            "2. Identify F (the production function being tested). "
            "Look up F in the Entity migration quick reference:\n"
            "   - F **not migrated** (stays in original module): "
            "patch is **unchanged** — F still resolves N from the original "
            "module's namespace (via re-export or direct import).\n"
            "   - F **migrated to module M**: the patch must be updated. "
            "Now use the Patch target lookup to find N's new home:\n"
            '     - N listed under **"moved to X"**: '
            "new patch = `X.N` (X is the sub-module that now imports N).\n"
            '     - N listed under **"still imported"** or not in the lookup: '
            "new patch = `M.N` (M imports N locally).\n"
        )
    else:
        parts.append(
            "For **each** patch string, work through these steps:\n"
            "1. Find N (the patched name — last component of the patch string).\n"
            "2. Look at the **Modified original file** imports in the context above. "
            "Is N imported from an external source there "
            "(e.g. `from ...llm_client import N`, `from third_party import N`)? "
            "Do NOT count a re-export from a new submodule "
            "(e.g. `from .new_submodule import N`) as an import.\n"
            "   - If N is **not** imported from an external source in the original "
            "file: scan the **Imports** of each new file for N. "
            "The new file that imports N externally is the new home — "
            "update the patch to `new_module.N` and move to the next string.\n"
            "   - If N **is** imported from an external source in the original file: "
            "go to step 3.\n"
            "3. Identify F (production function this test exercises).\n"
            "4. Look up F in the Entity migration quick reference.\n"
            "   - If F was **not migrated**: N is still resolved in the original "
            "module. Patch unchanged.\n"
            "   - If F was **migrated to M**: update patch to `M.N`.\n"
        )
    parts.append(
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
    migration_reminder = _extract_migration_reminder(context_msg)
    patch_lookup = _extract_patch_lookup(context_msg)
    parts = [
        context_msg,
        _PATCH_RULES,
        f"\n## Test function:\n```python\n{function_text}\n```\n\n"
        f"## Proposed patch() string updates:\n{rename_lines}\n\n",
    ]
    if migration_reminder:
        parts.append(migration_reminder)
    if patch_lookup:
        parts.append(patch_lookup)
    parts.append(
        "Are all these updates correct? Set `correct` to True only if every "
        "proposed patch string points to where the name is looked up after "
        "the split. Remember: renames must stay within the same top-level package "
        "(the first path component never changes).\n"
    )
    return "".join(parts)


def _build_no_change_verify_prompt(
    context_msg: str,
    function_text: str,
    old_patch_paths: List[str],
) -> str:
    """Build the user prompt for a no-change verify LLM call."""
    paths_list = "\n".join(f"- `{p}`" for p in old_patch_paths)
    migration_reminder = _extract_migration_reminder(context_msg)
    patch_lookup = _extract_patch_lookup(context_msg)
    parts = [
        context_msg,
        _PATCH_RULES,
        f"\n## Test function:\n```python\n{function_text}\n```\n\n"
        f"## Patch strings in this test (no update proposed):\n{paths_list}\n\n",
    ]
    if migration_reminder:
        parts.append(migration_reminder)
    if patch_lookup:
        parts.append(patch_lookup)
    parts.append(
        "The proposed update is: **no patch strings need changing**.\n\n"
        "Is this correct? Set `correct` to True only if all @patch strings "
        "in this function still point to the correct location after the split.\n"
        "When you reject (correct=false): populate `corrections` with "
        "{current_path: corrected_path} for each string listed above that "
        "needs renaming — use the exact strings from the list as keys. "
        "The `issue` field is for explaining why the current path is wrong, "
        "not for describing the fix. "
        "Only leave `corrections` empty if the fix requires an entirely new "
        "@patch decorator.\n"
    )
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
    # Same-file const definition proposals.
    # Maps old_val → set of proposed new_vals.  Also tracks "passthrough" usages:
    # tests that use the constant without renaming it.  When all renaming tests
    # agree AND no test passes through unchanged, the constant definition is updated.
    # When there is any conflict (passthrough + rename, or multiple different renames),
    # each affected function gets its decorator inlined to the correct value instead.
    same_file_proposals: Dict[str, Set[str]] = {}
    same_file_passthrough: Set[str] = (
        set()
    )  # old_vals kept unchanged by at least one test
    same_file_const_map: Dict[str, str] = {}  # populated after conflict resolution

    # Pre-compute name sets once for all per-function rename guards.
    _moved_out_names = _extract_moved_out_names(context_msg)
    _orig_users_map = _extract_still_in_orig_users(context_msg)
    _still_imported = _extract_still_imported_names(context_msg)

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
                    rate_limit_retries=config.rate_limit_retries,
                    rate_limit_backoff=config.rate_limit_backoff,
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
                        rate_limit_retries=config.rate_limit_retries,
                        rate_limit_backoff=config.rate_limit_backoff,
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
                        rate_limit_retries=config.rate_limit_retries,
                        rate_limit_backoff=config.rate_limit_backoff,
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
                    # A file split never moves entities across top-level packages.
                    # Renames like crispen.engine.X → libcst.metadata.X are always
                    # wrong (confusing import source with patch target).
                    and new.split(".")[0] == old.split(".")[0]
                    # A rename must not change the patched name itself (last
                    # component).  A split only relocates the defining module;
                    # the entity name is unchanged.
                    and new.rsplit(".", 1)[-1] == old.rsplit(".", 1)[-1]
                )
            }

            if not patch_renames:
                # Verify the "no change needed" conclusion.
                no_change_verify_prompt = _build_no_change_verify_prompt(
                    context_msg, func.full_text, func.old_patch_paths
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
                    rate_limit_retries=config.rate_limit_retries,
                    rate_limit_backoff=config.rate_limit_backoff,
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
                        and new.split(".")[0] == old.split(".")[0]
                        and new.rsplit(".", 1)[-1] == old.rsplit(".", 1)[-1]
                    )
                }
                # Guard: drop corrections that are known-incorrect hallucinations.
                # _is_bad_rename blocks two patterns:
                #   A) shallowing a moved-out name (would raise AttributeError)
                #   B) deepening a still-in name when the test exercises the
                #      original-module caller (wrong binding intercepted)
                # The second filter retains the pre-existing guard: drop any
                # still-in name that isn't being deepened into a true sub-module.
                if _still_imported or _moved_out_names:
                    corrections_renames = {
                        old: new
                        for old, new in corrections_renames.items()
                        if not _is_bad_rename(
                            old,
                            new,
                            _moved_out_names,
                            _still_imported,
                            _orig_users_map,
                            func.full_text,
                        )
                        and (
                            old.rsplit(".", 1)[-1] not in _still_imported
                            or new.rsplit(".", 1)[0].startswith(
                                old.rsplit(".", 1)[0] + "."
                            )
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
                        rate_limit_retries=config.rate_limit_retries,
                        rate_limit_backoff=config.rate_limit_backoff,
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
                rate_limit_retries=config.rate_limit_retries,
                rate_limit_backoff=config.rate_limit_backoff,
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
                # Verify failed; accept renames — but still filter bad ones.
                patch_renames_safe = {
                    old: new
                    for old, new in patch_renames.items()
                    if not _is_bad_rename(
                        old,
                        new,
                        _moved_out_names,
                        _still_imported,
                        _orig_users_map,
                        func.full_text,
                    )
                }
                orig_text = "\n".join(source_lines[func.start_line - 1 : func.end_line])
                new_text = apply_patch_strings(orig_text, patch_renames_safe)
                if new_text != orig_text:
                    func_splices.append((func.start_line, func.end_line, new_text))
                string_swap_results.append((func, patch_renames_safe))
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
                patch_renames_safe = {
                    old: new
                    for old, new in patch_renames.items()
                    if not _is_bad_rename(
                        old,
                        new,
                        _moved_out_names,
                        _still_imported,
                        _orig_users_map,
                        func.full_text,
                    )
                }
                orig_text = "\n".join(source_lines[func.start_line - 1 : func.end_line])
                new_text = apply_patch_strings(orig_text, patch_renames_safe)
                if new_text != orig_text:
                    func_splices.append((func.start_line, func.end_line, new_text))
                string_swap_results.append((func, patch_renames_safe))
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

    # Collect cross-file and same-file constant proposals.
    cross_file_patch_maps: Dict[str, Dict[str, str]] = {}
    if string_swap_results and scan_file:
        scan_file_abs = str(Path(scan_file).resolve())
        for func, accepted in string_swap_results:
            for ref in func.const_refs:
                new_val = accepted.get(ref.resolved_value)
                if new_val is None or new_val == ref.resolved_value:
                    # Test uses the constant without renaming it → passthrough.
                    if ref.source_file == scan_file_abs:
                        same_file_passthrough.add(ref.resolved_value)
                    continue
                if ref.source_file == scan_file_abs:
                    same_file_proposals.setdefault(ref.resolved_value, set()).add(
                        new_val
                    )
                else:
                    cross_file_patch_maps.setdefault(ref.source_file, {})[
                        ref.resolved_value
                    ] = new_val

        # Resolve same-file proposals into updates and per-function inlines.
        # All renaming tests agree AND no passthrough → update the constant definition.
        # Any conflict (passthrough + rename, OR multiple different renames) → inline
        # the correct string directly into each affected function's decorator so the
        # shared constant stays stable and each test sees the right patch target.
        same_file_const_map = {
            old: next(iter(new_set))
            for old, new_set in same_file_proposals.items()
            if len(new_set) == 1 and old not in same_file_passthrough
        }
        conflicting_old_vals = {
            old
            for old, new_set in same_file_proposals.items()
            if len(new_set) > 1 or old in same_file_passthrough
        }
        if conflicting_old_vals:
            for func, accepted in string_swap_results:
                inline_subs: Dict[str, str] = {}
                for ref in func.const_refs:
                    if (
                        ref.source_file == scan_file_abs
                        and ref.resolved_value in conflicting_old_vals
                    ):
                        new_val = accepted.get(ref.resolved_value)
                        if new_val and new_val != ref.resolved_value:
                            inline_subs[ref.const_name] = new_val
                if not inline_subs:
                    continue
                orig_text = "\n".join(source_lines[func.start_line - 1 : func.end_line])
                # Start from any already-pending splice for this function.
                base_text = orig_text
                existing_idx: Optional[int] = None
                for idx_i, (sl, el, txt) in enumerate(func_splices):
                    if sl == func.start_line and el == func.end_line:
                        base_text = txt
                        existing_idx = idx_i
                        break
                inlined = _substitute_consts_in_func_text(base_text, inline_subs)
                if inlined == base_text:
                    continue  # pragma: no cover
                if existing_idx is not None:
                    func_splices[existing_idx] = (
                        func.start_line,
                        func.end_line,
                        inlined,
                    )
                else:
                    func_splices.append((func.start_line, func.end_line, inlined))

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

    max_attempts = 1 + config.patch_update_retries

    per_file_abs = {str(Path(f).resolve()) for f in per_file}

    # Aggregate cross-file constant proposals:
    # abs_file → {old_val → {proposed_new_val, …}}
    cross_file_proposals: Dict[str, Dict[str, Set[str]]] = {}

    # Update per_file sources (in memory, not yet written to disk).
    for filepath, state in per_file.items():
        file_src = state["source"]
        relevant_contexts = [
            ctx
            for ctx in fl_contexts
            if any(path in file_src for path in ctx.forking_old_paths)
        ]
        context_msg = _build_context_message(relevant_contexts)
        new_src, changed, cross = _process_file_source(
            file_src,
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
        relevant_contexts = [
            ctx
            for ctx in fl_contexts
            if any(path in src for path in ctx.forking_old_paths)
        ]
        context_msg = _build_context_message(relevant_contexts)
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
