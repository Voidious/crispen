"""LLM-powered @patch string rewriter for FileLimiter 'rewrite' mode."""

from __future__ import annotations

import ast
import re
import sys
from collections import deque
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
class _CgIndex:
    """Pre-built call-graph index for BFS-based forking path resolution.

    Built once per ``apply_patch_callgraph`` call by scanning the repo and
    adding new sub-module sources from each :class:`_FLContext`.  Import maps
    are resolved lazily and cached on first access.
    """

    module_to_source: Dict[str, str]  # dotted module path → source
    module_to_package: Dict[str, str]  # dotted module path → package path
    module_to_defs: Dict[str, Set[str]]  # dotted module path → top-level names
    file_to_module: Dict[str, str]  # abs file path → dotted module path
    _import_cache: Dict[str, Dict[str, Tuple[str, str]]] = field(default_factory=dict)

    def get_imports(self, module: str) -> Dict[str, Tuple[str, str]]:
        """Return ``{local_name: (module_path, orig_name)}`` for *module*, cached."""
        if module not in self._import_cache:
            src = self.module_to_source.get(module, "")
            pkg = self.module_to_package.get(module, "")
            self._import_cache[module] = _cg_parse_imports(src, pkg)
        return self._import_cache[module]


@dataclass
class _TestFunctionInfo:
    """A test function containing @patch decorators referencing old paths."""

    function_name: str
    full_text: str  # text sent to LLM (constants substituted with their values)
    old_patch_paths: List[str]  # forking paths that need LLM attention
    stable_patch_paths: List[str]  # already-correct paths (set by basic mode)
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
    cg_resolved: int = 0  # forking paths resolved by callgraph
    no_change: int = 0  # functions LLM confirmed need no change
    rename: int = 0  # functions LLM renamed via string-swap
    rewrite: int = 0  # functions LLM fully rewrote
    edit_failures: int = (
        0  # functions where retries exhausted without a verified update
    )


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
    "check two things:\n"
    "     (a) Is N listed in M's **Imports** section? "
    "If not, leave the patch unchanged.\n"
    "     (b) Is F (or a non-parameter-receiving callee of F within M) "
    "listed under N in the **Name references** section for M? "
    "The Name references map records top-level **definitions** (standalone "
    "functions and classes) — if F is a method inside a class, check "
    "whether F's **containing class** is listed. "
    "If neither F nor F's containing class appears in N's Name references "
    "for M, F does not call N — it receives the resource as a direct "
    "argument from its caller. The original-module patch is a harmless "
    "no-op that still resolves: leave it unchanged.\n"
    "     Only update the patch to `M.N` when BOTH (a) and (b) are true.\n"
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
    "The source of the import is irrelevant to the patch target.\n"
    "7. **Parameter-passing helpers:** If F calls N to produce a resource and passes "
    "it to migrated helpers — e.g. `conn = build_conn(fetch_key(...))` then "
    "`_process(conn, data)` where `_process` was migrated — those helpers do NOT "
    "call N; F does. Apply steps 3–4 using F (not the helpers) as the caller of N.\n\n"
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
    """Return {name: string_value} for module-level ``NAME = "string"`` assignments.

    Handles both plain assignments (``NAME = "value"``) and annotated assignments
    (``NAME: str = "value"``).
    """
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
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            result[node.target.id] = node.value.value
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


class _ConstReverter(cst.CSTTransformer):
    """Reverse ``@patch("value")`` → ``@patch(NAME)`` for unchanged constants."""

    def __init__(self, reverse_map: Dict[str, str]) -> None:
        self._rev = reverse_map  # {resolved_value: const_name}

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        if not _is_patch_call(updated_node):
            return updated_node
        if not updated_node.args:
            return updated_node
        arg0 = updated_node.args[0].value
        if not isinstance(arg0, cst.SimpleString):
            return updated_node
        inner = ast.literal_eval(arg0.value)
        if not isinstance(inner, str) or inner not in self._rev:
            return updated_node
        const_name = self._rev[inner]
        if "." in const_name:
            mod, attr = const_name.split(".", 1)
            new_arg_val: cst.BaseExpression = cst.Attribute(
                value=cst.Name(mod),
                attr=cst.Name(attr),
                dot=cst.Dot(),
            )
        else:
            new_arg_val = cst.Name(const_name)
        new_arg = updated_node.args[0].with_changes(value=new_arg_val)
        return updated_node.with_changes(args=(new_arg,) + updated_node.args[1:])


def _restore_const_refs(func_text: str, const_refs: List["_ConstRef"]) -> str:
    """Restore @patch constant references the LLM left as their substituted values.

    Before calling the LLM, constant names are expanded to their string values
    (``@patch(NAME)`` → ``@patch("value")``).  Any decorator the LLM left with
    the original value should be reverted to the named-constant form so the
    output file matches the source style.
    """
    if not const_refs:
        return func_text
    reverse_map = {ref.resolved_value: ref.const_name for ref in const_refs}
    try:
        tree = cst.parse_module(func_text)
    except cst.ParserSyntaxError:
        return func_text
    return tree.visit(_ConstReverter(reverse_map)).code


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
        # Collect resolvable patch paths from decorators, split into two buckets:
        #   forking_dec_paths  — paths matching old_paths (need LLM attention)
        #   stable_dec_paths   — paths not matching old_paths (already correct)
        # A function is only collected when at least one decorator path matches
        # old_paths.  The LLM receives only forking paths to evaluate; stable
        # paths are shown separately with a "do not modify" instruction.
        forking_dec_paths: List[str] = []  # decorator paths matching old_paths
        stable_dec_paths: List[str] = []  # decorator paths already correct
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
                    if _matches_any(inner, self._old_paths):
                        forking_dec_paths.append(inner)
                        has_match = True
                    else:
                        stable_dec_paths.append(inner)
            elif isinstance(arg0, cst.Name) and arg0.value in self._const_map:
                const_val, const_file = self._const_map[arg0.value]
                all_const_refs.append(
                    _ConstRef(
                        const_name=arg0.value,
                        source_file=const_file,
                        resolved_value=const_val,
                        patch_dec_idx=patch_dec_idx,
                    )
                )
                if _matches_any(const_val, self._old_paths):
                    forking_dec_paths.append(const_val)
                    has_match = True
                else:
                    stable_dec_paths.append(const_val)
            elif isinstance(arg0, cst.Attribute) and isinstance(arg0.value, cst.Name):
                module_alias = arg0.value.value
                attr_name = arg0.attr.value
                if module_alias in self._attr_const_map:
                    attr_map = self._attr_const_map[module_alias]
                    if attr_name in attr_map:
                        const_val, const_file = attr_map[attr_name]
                        all_const_refs.append(
                            _ConstRef(
                                const_name=f"{module_alias}.{attr_name}",
                                source_file=const_file,
                                resolved_value=const_val,
                                patch_dec_idx=patch_dec_idx,
                            )
                        )
                        if _matches_any(const_val, self._old_paths):
                            forking_dec_paths.append(const_val)
                            has_match = True
                        else:
                            stable_dec_paths.append(const_val)
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

        # old_patch_paths: only forking paths (need LLM attention).
        # body_paths already contains only forking paths (filtered in
        # _find_with_patch_paths_in_body). stable_dec_paths holds the
        # already-correct paths that should not be re-evaluated.
        old_patch_paths = forking_dec_paths + body_paths

        # Build the full_text sent to the LLM: substitute constant names with
        # their string values so the LLM always sees plain string literals.
        # Substitute ALL const refs (both forking and stable) so the full
        # function text is coherent and the LLM can follow the mock params.
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
                stable_patch_paths=stable_dec_paths,
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


def _is_bad_rename(
    old: str,
    new: str,
    moved_out_names: Set[str],
    still_imported: Set[str],
    orig_users_map: Dict[str, List[str]],
    test_text: str,
    new_module_imports: Optional[Dict[str, Set[str]]] = None,
) -> bool:
    """Return True if the rename (old → new) is a known-incorrect hallucination.

    Three patterns are blocked:

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

    C) **Target module doesn't import the name**: the proposed target sub-module
       is a known new file (from the split) but its source does not import the
       name being patched.  Patching at that path would always raise
       AttributeError regardless of which function is being tested.
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
    # Pattern C: target module is a known new file that doesn't import the name.
    # Only applies to names tracked as external imports (moved_out or still_imported)
    # so that locally-defined symbols (classes, functions) are never blocked.
    if new_module_imports is not None and (
        name in moved_out_names or name in still_imported
    ):
        new_mod_path = new.rsplit(".", 1)[0]
        if new_mod_path in new_module_imports:
            if name not in new_module_imports[new_mod_path]:
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


# ---------------------------------------------------------------------------
# Call-graph tracing helpers (algorithmic forking resolution)
# ---------------------------------------------------------------------------

_CG_MAX_DEPTH = 12  # maximum BFS hops from test function to a sub-module target
_CG_MAX_MODULES = 50  # maximum distinct modules visited per resolution attempt
_CG_CANDIDATES_LLM_THRESHOLD = 10  # max candidates to include in LLM prompts


def _cg_collect_called_names(source: str) -> Set[str]:
    """Return all function/attribute names called anywhere in *source*.

    For attribute-access calls like ``m.func()`` where the receiver is a plain
    name, both the bare attribute (``"func"``) and the ``"alias.attr"`` form
    (``"m.func"``) are emitted.  This lets the BFS distinguish between a
    directly imported name and a name accessed through a module alias.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
                if isinstance(node.func.value, ast.Name):
                    names.add(f"{node.func.value.id}.{node.func.attr}")
    return names


def _cg_collect_func_body_calls(source: str, func_name: str) -> Set[str]:
    """Return names called within *func_name*'s body in *source*.

    Searches for the first function definition named *func_name* in the
    module-level body and collects all Call targets within it.  Returns an
    empty set when the function is not found or *source* fails to parse.

    Like :func:`_cg_collect_called_names`, attribute-access calls of the form
    ``alias.method()`` emit both ``"method"`` and ``"alias.method"``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != func_name:
            continue
        calls: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(child.func.attr)
                    if isinstance(child.func.value, ast.Name):
                        calls.add(f"{child.func.value.id}.{child.func.attr}")
        return calls
    return set()


def _cg_resolve_call_to_import(
    called_name: str,
    imports: Dict[str, Tuple[str, str]],
) -> Optional[Tuple[str, str]]:
    """Resolve a called name to ``(module, func_name)`` via *imports*.

    Handles two forms:

    * Plain name (``"func"``): looked up directly in *imports*.
    * Alias-access (``"alias.func"``): the alias is looked up in *imports* to
      retrieve the target module; ``func`` becomes the function name within that
      module.  This covers calls like ``import mymod as m; m.func()`` where
      ``_cg_collect_called_names`` emits ``"m.func"``.

    Returns ``None`` when the name cannot be resolved.
    """
    if "." in called_name:
        alias, attr = called_name.split(".", 1)
        if alias in imports:
            mod, _ = imports[alias]
            return (mod, attr)
        return None
    if called_name in imports:
        return imports[called_name]
    return None


def _cg_collect_defined_names(source: str) -> Set[str]:
    """Return top-level function and class names defined in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _cg_file_to_module_and_package(file_path: Path, repo_root: Path) -> Tuple[str, str]:
    """Return ``(module_dotted_path, package_dotted_path)`` for a ``.py`` file.

    ``pkg/utils/__init__.py`` → ``("pkg.utils", "pkg.utils")``.
    ``pkg/utils/helpers.py``  → ``("pkg.utils.helpers", "pkg.utils")``.
    """
    rel = file_path.relative_to(repo_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
        module = ".".join(parts)
        package = module
    else:
        parts[-1] = parts[-1][:-3]
        module = ".".join(parts)
        package = ".".join(parts[:-1])
    return module, package


def _cg_parse_imports(source: str, package: str) -> Dict[str, Tuple[str, str]]:
    """Parse import statements in *source*.

    Returns ``local_name → (module, orig_name)``.

    *package* is the dotted path of the package that contains this module
    (used to resolve relative imports).  For example, for a file at
    ``pkg/utils/helpers.py`` the package is ``"pkg.utils"``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    result: Dict[str, Tuple[str, str]] = {}
    pkg_parts = package.split(".") if package else []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname if alias.asname else alias.name.split(".")[0]
                result[local] = (alias.name, alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                # Relative import: level=1 means same package (0 levels up),
                # level=2 means one level up, etc.
                go_up = node.level - 1
                if go_up > len(pkg_parts):
                    continue  # invalid — can't go above root
                base_parts = pkg_parts[: len(pkg_parts) - go_up] if go_up else pkg_parts
                base = ".".join(base_parts)
                if node.module:
                    mod_path = f"{base}.{node.module}" if base else node.module
                else:
                    mod_path = base
            else:
                mod_path = node.module or ""  # pragma: no branch
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname if alias.asname else alias.name
                result[local] = (mod_path, alias.name)
    return result


def _cg_build_index(
    repo_root: Optional[str],
    per_file_sources: Dict[str, str],
    fl_contexts: List[_FLContext],
) -> _CgIndex:
    """Build a repo-wide :class:`_CgIndex` for BFS call-graph resolution.

    Scans all ``.py`` files under *repo_root* (using *per_file_sources* as
    in-memory overrides for modified files), then adds new sub-module sources
    from each :class:`_FLContext` that are not yet on disk.
    """
    module_to_source: Dict[str, str] = {}
    module_to_package: Dict[str, str] = {}
    module_to_defs: Dict[str, Set[str]] = {}
    file_to_module: Dict[str, str] = {}

    if repo_root is not None:
        repo_root_path = Path(repo_root).resolve()
        for py_file in repo_root_path.rglob("*.py"):
            rel_parts = py_file.relative_to(repo_root_path).parts
            if any(p in _EXCLUDED_DIR_NAMES for p in rel_parts[:-1]):
                continue
            try:
                mod, pkg = _cg_file_to_module_and_package(py_file, repo_root_path)
            except ValueError:  # pragma: no cover
                continue
            abs_path = str(py_file.resolve())
            src = per_file_sources.get(abs_path)
            if src is None:
                try:
                    src = py_file.read_text(encoding="utf-8")
                except OSError:
                    continue
            module_to_source[mod] = src
            module_to_package[mod] = pkg
            module_to_defs[mod] = _cg_collect_defined_names(src)
            file_to_module[abs_path] = mod

    # Add new sub-module sources (in-memory; not yet written to disk).
    for ctx in fl_contexts:
        for rel_path, src in ctx.new_files.items():
            new_mod = ctx.new_module_paths.get(rel_path)
            if not new_mod or not src or new_mod in module_to_source:
                continue
            pkg = (
                new_mod
                if Path(rel_path).name == "__init__.py"
                else (new_mod.rsplit(".", 1)[0] if "." in new_mod else "")
            )
            module_to_source[new_mod] = src
            module_to_package[new_mod] = pkg
            module_to_defs[new_mod] = _cg_collect_defined_names(src)

    return _CgIndex(
        module_to_source=module_to_source,
        module_to_package=module_to_package,
        module_to_defs=module_to_defs,
        file_to_module=file_to_module,
    )


def _expand_module_terminals(
    src: str,
    module: str,
    forking_name: str,
    terminal: Dict[Tuple[str, str], str],
) -> None:
    """Expand *terminal* in-place with transitive callers within *module*.

    After direct references to *forking_name* are seeded into *terminal*, this
    finds all locally-defined functions that call (directly or transitively)
    any of those seed functions and adds them as terminals for the same path.

    Example: if ``_helper`` directly calls ``use_fn`` and ``public_func`` calls
    ``_helper``, both will be in the terminal after expansion — so the BFS can
    match a test that calls ``public_func`` without needing to traverse inside
    the sub-module.
    """
    direct = {f_name for (mod, f_name) in terminal if mod == module}
    if not direct:
        return
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    # Collect top-level function names defined in this module.
    local_defs: Set[str] = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # For each local function, record which other local functions it calls.
    local_calls: Dict[str, Set[str]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id in local_defs:
                    calls.add(child.func.id)
        local_calls[node.name] = calls
    # Fixed-point expansion: add callers of already-reachable functions.
    reachable = set(direct)
    changed = True
    while changed:
        changed = False
        for f_name, called_locals in local_calls.items():
            if f_name not in reachable and called_locals & reachable:
                reachable.add(f_name)
                changed = True
    new_path = f"{module}.{forking_name}"
    for f_name in reachable - direct:
        terminal[(module, f_name)] = new_path


def _resolve_forking_path_candidates(
    forking_name: str,
    func_text: str,
    ctx: _FLContext,
    index: _CgIndex,
    calling_module: str,
    max_depth: int = _CG_MAX_DEPTH,
    max_modules: int = _CG_MAX_MODULES,
) -> Tuple[Optional[str], List[str], bool, List[str]]:
    """BFS call-graph resolution returning the full candidate set.

    Returns ``(resolved_path, sorted_candidates, truncated, static_candidates)``
    where:

    - ``resolved_path`` — the single new dotted path when exactly one candidate
      is found; ``None`` when zero or multiple candidates exist or when the
      pre-check fails.
    - ``sorted_candidates`` — BFS-reachable candidate paths (may be empty).
      Only meaningful when ``truncated`` is ``False``; when limits were hit the
      list may be incomplete so callers must not rely on it for validation.
    - ``truncated`` — ``True`` when *max_depth* or *max_modules* was reached
      during traversal, indicating the
      candidate list may be incomplete.
    - ``static_candidates`` — all possible new paths derived from the terminal
      set (independent of BFS reachability).  When ``sorted_candidates`` is
      empty and ``truncated`` is ``False``, callers may fall back to this set
      as the constraint for LLM prompts and verification.
    """
    if not calling_module:
        return None, [], False, []
    if forking_name not in _get_external_import_names(ctx.original_source):
        return None, [], False, []

    # Terminal set: (module_path, func_name) → new dotted path.
    # Includes new sub-modules AND the original module: if some functions that
    # call forking_name remained in the original file after the split, patching
    # the original path is still correct for tests exercising those functions.
    terminal: Dict[Tuple[str, str], str] = {}
    for rel_path, src in ctx.new_files.items():
        if not src:
            continue
        new_mod = ctx.new_module_paths.get(rel_path)
        if new_mod is None:
            continue
        ref_map = _name_reference_map(src)
        if forking_name not in ref_map:
            continue
        for f_name in ref_map[forking_name]:
            terminal[(new_mod, f_name)] = f"{new_mod}.{forking_name}"
        _expand_module_terminals(src, new_mod, forking_name, terminal)

    # Original module: use the post-split source so only functions that actually
    # remained there (and still reference forking_name) contribute terminals.
    if ctx.modified_source:
        orig_ref_map = _name_reference_map(ctx.modified_source)
        for f_name in orig_ref_map.get(forking_name, []):
            terminal[(ctx.old_module, f_name)] = f"{ctx.old_module}.{forking_name}"
        _expand_module_terminals(
            ctx.modified_source, ctx.old_module, forking_name, terminal
        )

    if not terminal:
        return None, [], False, []

    # Static candidates: all possible new paths from the terminal set, before
    # BFS filtering.  Used as fallback when BFS finds no reachable candidates.
    static_cands = sorted(set(terminal.values()))

    # Exclude __init__.py entries: the package __init__ acts as a transparent
    # re-export shim after a split and should be traversed like any other module
    # so BFS can follow calls through it to the implementing sub-module.
    new_module_set = {
        mod
        for rel, mod in ctx.new_module_paths.items()
        if not rel.endswith("__init__.py")
    }

    visited: Set[Tuple[str, str]] = set()
    modules_seen: Set[str] = set()
    candidates: Set[str] = set()
    truncated = False
    queue = deque()  # (module_path, func_name, depth)

    init_imports = index.get_imports(calling_module)
    for called_name in _cg_collect_called_names(func_text):
        resolved = _cg_resolve_call_to_import(called_name, init_imports)
        if resolved is not None:
            queue.append((resolved[0], resolved[1], 0))

    while queue:
        module, func_name, depth = queue.popleft()
        key = (module, func_name)

        if key in visited:
            continue
        visited.add(key)

        if key in terminal:
            candidates.add(terminal[key])
            # After recording this candidate, scan the terminal function's body
            # for cross-module calls to other new submodules.  An orchestrator
            # in one new submodule (e.g. main.py) may delegate work to
            # functions in a sibling submodule (e.g. llm_steps.py) that also
            # use the forking name — each sibling is an independent candidate
            # the test must patch separately.
            if module in new_module_set:
                src = index.module_to_source.get(module)
                if src and func_name in index.module_to_defs.get(module, set()):
                    body_calls = _cg_collect_func_body_calls(src, func_name)
                    mod_imports = index.get_imports(module)
                    for called_name in body_calls:
                        resolved = _cg_resolve_call_to_import(called_name, mod_imports)
                        if resolved is not None:
                            next_mod, next_name = resolved
                            if (
                                next_mod in new_module_set
                                and (next_mod, next_name) not in visited
                            ):
                                queue.append((next_mod, next_name, depth + 1))
            continue

        if module in new_module_set:
            continue  # non-terminal; _expand_module_terminals handles local chains

        if depth >= max_depth:
            truncated = True
            continue

        if module not in modules_seen:
            if len(modules_seen) >= max_modules:
                truncated = True
                continue
            modules_seen.add(module)

        src = index.module_to_source.get(module)
        if src is None:
            continue

        if func_name in index.module_to_defs.get(module, set()):
            # Function is defined here — follow its body calls.
            body_calls = _cg_collect_func_body_calls(src, func_name)
            mod_imports = index.get_imports(module)
            for called_name in body_calls:
                resolved = _cg_resolve_call_to_import(called_name, mod_imports)
                if resolved is not None:
                    next_mod, next_name = resolved
                    if (next_mod, next_name) not in visited:
                        queue.append((next_mod, next_name, depth + 1))
                elif called_name in index.module_to_defs.get(module, set()):
                    # Locally defined in the same module — follow at same depth.
                    if (module, called_name) not in visited:
                        queue.append((module, called_name, depth))
        else:
            # Not defined here — follow a module-level re-export if present.
            mod_imports = index.get_imports(module)
            if func_name in mod_imports:
                rexport_mod, rexport_name = mod_imports[func_name]
                if (rexport_mod, rexport_name) not in visited:
                    queue.append((rexport_mod, rexport_name, depth))

    sorted_cands = sorted(candidates)
    if len(candidates) == 1:
        return next(iter(candidates)), sorted_cands, truncated, static_cands
    return None, sorted_cands, truncated, static_cands


def _resolve_forking_path_via_callgraph(
    forking_name: str,
    func_text: str,
    ctx: _FLContext,
    index: _CgIndex,
    calling_module: str,
    max_depth: int = _CG_MAX_DEPTH,
    max_modules: int = _CG_MAX_MODULES,
) -> Optional[str]:
    """Resolve a forking @patch path by BFS across the repo call graph.

    Starting from the test function's direct calls (resolved through
    *calling_module*'s import statements), performs a BFS across the repo
    index to find which new sub-module's function is transitively reachable.
    Returns the new dotted path when exactly one sub-module qualifies;
    returns ``None`` when zero or multiple qualify or when the pre-check fails.

    Module-level re-exports (``from .sub import F`` where ``F`` is not defined
    locally) are followed without consuming depth budget, so that the split
    original file does not artificially cap traversal depth.

    Limits: *max_depth* hops and *max_modules* distinct modules per resolution
    attempt.
    """
    path, _, _, _ = _resolve_forking_path_candidates(
        forking_name,
        func_text,
        ctx,
        index,
        calling_module,
        max_depth=max_depth,
        max_modules=max_modules,
    )
    return path


def _build_rename_guard_sets(
    fl_contexts: List[_FLContext],
) -> Tuple[
    Set[str],
    Set[str],
    Dict[str, List[str]],
    Dict[str, Set[str]],
]:
    """Derive rename-guard data directly from the split contexts.

    Returns ``(moved_out_names, still_imported, orig_users_map,
    new_module_imports)`` where:
    - ``moved_out_names`` — names removed from the modified original during the
      split (patching the original path would raise AttributeError).
    - ``still_imported`` — names still directly imported by the modified original.
    - ``orig_users_map`` — maps each still-imported name to the top-level
      functions in the modified original that reference it.
    - ``new_module_imports`` — maps each new sub-module's dotted path to the set
      of external names it imports (for Pattern C guard).
    """
    moved_out_names: Set[str] = set()
    still_imported: Set[str] = set()
    orig_users_map: Dict[str, List[str]] = {}
    new_module_imports: Dict[str, Set[str]] = {}
    for ctx in fl_contexts:
        orig_ext = _get_external_import_names(ctx.original_source)
        mod_ext = _get_external_import_names(ctx.modified_source)
        moved_out_names |= orig_ext - mod_ext
        still_in = orig_ext & mod_ext
        still_imported |= still_in
        ref_map = _name_reference_map(ctx.modified_source)
        for name in still_in:
            users = ref_map.get(name, [])
            if users:
                if name in orig_users_map:
                    # Merge without duplicates when name appears in multiple contexts.
                    orig_users_map[name] = list(
                        dict.fromkeys(orig_users_map[name] + users)
                    )
                else:
                    orig_users_map[name] = users
        for rel_path, content in ctx.new_files.items():
            mod_path = ctx.new_module_paths.get(rel_path, rel_path)
            new_module_imports[mod_path] = _get_external_import_names(content)
    return moved_out_names, still_imported, orig_users_map, new_module_imports


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
                            + "; if F was **migrated to that submodule** (check the"
                            " entity migration table) AND F is listed under N in the"
                            " **Name references** for that submodule, update the"
                            " patch to `submodule.N`. If F is NOT listed in N's"
                            " Name references for that submodule, F does not call N"
                            " — leave the patch unchanged (harmless no-op)\n"
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
    stable_patch_paths: Optional[List[str]] = None,
    candidates_per_path: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Build the user prompt for the per-function classify LLM call."""
    paths_list = "\n".join(f"- `{p}`" for p in old_patch_paths)
    migration_reminder = _extract_migration_reminder(context_msg)
    patch_lookup = _extract_patch_lookup(context_msg)
    parts = [context_msg, _PATCH_RULES]
    parts.append(
        f"\n## Test function:\n```python\n{function_text}\n```\n\n"
        "## Patch strings that need updating:\n"
        f"{paths_list}\n\n"
    )
    if stable_patch_paths:
        stable_list = "\n".join(f"- `{p}`" for p in stable_patch_paths)
        parts.append(
            "## Patch strings already correct — do not modify:\n" f"{stable_list}\n\n"
        )
    if migration_reminder:
        parts.append(migration_reminder)
    if patch_lookup:
        parts.append(patch_lookup)
    parts.append(
        "Every patch string in the **'Patch strings that need updating'** list "
        "must be evaluated independently — a single test function may need "
        "**multiple** strings updated, each potentially pointing to a **different** "
        "new sub-module. Do **not** modify any string listed under "
        "'Patch strings already correct'.\n\n"
    )
    if patch_lookup:
        parts.append(
            "**For each patch string that needs updating:**\n"
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
            "new patch = `M.N` (M imports N locally — F may call N directly"
            " or via a helper in M; either way M's local binding must be patched).\n"
        )
    else:
        parts.append(
            "For **each** patch string that needs updating, work through these steps:\n"
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
        "Include every path from the 'Patch strings that need updating' list in "
        "`patch_renames` (mapping old → new, or old → old if unchanged). "
        "Do **not** include paths from 'Patch strings already correct' in "
        "`patch_renames`. "
        "Set `needs_rewrite` to True only for structural changes "
        "(new decorators, new mock parameters, or body edits). "
        "A rewrite is also needed when the split routes mock calls through more "
        "than one sub-module that each import the patched name independently — "
        "a single @patch path can only intercept calls in its own module, so a "
        "new @patch decorator must be added for each sub-module involved.\n"
    )
    if candidates_per_path:
        small_cands = {
            old: cands
            for old, cands in candidates_per_path.items()
            if cands and len(cands) <= _CG_CANDIDATES_LLM_THRESHOLD
        }
        if small_cands:
            parts.append(
                "\n## Call-graph candidate paths (from static analysis):\n"
                "Static call-graph analysis found which new sub-module(s) each "
                "patched name is reachable in from this test:\n"
            )
            for old in sorted(small_cands):
                cands = small_cands[old]
                cands_str = ", ".join(f"`{c}`" for c in cands)
                if len(cands) > 1:
                    parts.append(
                        f"- `{old}` → reachable in **multiple** sub-modules: "
                        f"{cands_str}.\n"
                        f"  **IMPORTANT: the old path `{old}` no longer exists "
                        f"at that location — patching it will raise `AttributeError` "
                        f"at test runtime. Returning `patch_renames: {{}}` (no "
                        f"change) is ALWAYS WRONG when this path appears in the "
                        f"candidates list.**\n"
                        f"  You MUST either rename it to one of the candidates or "
                        f"set `needs_rewrite=True`.\n"
                        f"  **Multi-submodule signal**: count the mock's "
                        f"`side_effect` entries. If the total exceeds what any "
                        f"single sub-module can account for, set "
                        f"`needs_rewrite=True` — each sub-module needs its own "
                        f"@patch. Only pick a single sub-module if the test is "
                        f"structured so that the others are never reached "
                        f"(e.g. an early-return guard, or a call_count==1 "
                        f"assertion).\n"
                    )
                else:
                    parts.append(
                        f"- `{old}` → must be: {cands_str}.\n"
                        f"  **IMPORTANT: the old path `{old}` no longer exists "
                        f"— patching it raises `AttributeError`. Returning "
                        f"`patch_renames: {{}}` (no change) is WRONG.**\n"
                    )
            parts.append("\n")
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
    candidates_per_path: Optional[Dict[str, List[str]]] = None,
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
    if candidates_per_path:
        small_cands = {
            old: cands
            for old, cands in candidates_per_path.items()
            if cands and len(cands) <= _CG_CANDIDATES_LLM_THRESHOLD
        }
        if small_cands:
            parts.append(
                "\n## Call-graph candidate paths (from static analysis):\n"
                "Valid new locations for each old path (from call-graph analysis):\n"
            )
            for old in sorted(small_cands):
                cands_str = ", ".join(f"`{c}`" for c in small_cands[old])
                parts.append(f"- `{old}` → valid options: {cands_str}\n")
            parts.append("\n")
    parts.append(
        "Are all these updates correct? Set `correct` to True only if every "
        "proposed patch string points to where the name is looked up after "
        "the split. Remember: renames must stay within the same top-level package "
        "(the first path component never changes).\n"
        "Trace the full execution path for each mock call in the test: if the "
        "mock has multiple side_effect entries or is asserted called multiple "
        "times, each call may route through a different sub-module. Verify the "
        "proposed path intercepts ALL of them, not just the first hop. Ground "
        "your analysis in the test's actual construction — if a branch is not "
        "triggered by the test's inputs (e.g. an empty list skips an `if` "
        "block), do not assume that branch executes.\n"
        "**Multi-submodule completeness check**: require an additional @patch "
        "for a second sub-module only when the test **directly exercises** "
        "that sub-module's code path — for example, the function under test "
        "calls a helper that is defined in the other sub-module. Do NOT reject "
        "a single @patch solely because the mock's side_effect count or "
        "call_count is ≥2: multiple calls may all route through the same "
        "sub-module (e.g., two sequential calls to `handlers.fetch_remote` and "
        "`handlers.store_result` both import `send_request` from `handlers`, so "
        "patching `handlers.send_request` alone covers both). Set "
        "`correct=False` with a multi-submodule issue only when the test's "
        "setup or inputs provably trigger the other sub-module's code path.\n"
    )
    return "".join(parts)


def _build_no_change_verify_prompt(
    context_msg: str,
    function_text: str,
    old_patch_paths: List[str],
    stable_patch_paths: Optional[List[str]] = None,
    candidates_per_path: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Build the user prompt for a no-change verify LLM call."""
    paths_list = "\n".join(f"- `{p}`" for p in old_patch_paths)
    migration_reminder = _extract_migration_reminder(context_msg)
    patch_lookup = _extract_patch_lookup(context_msg)
    parts = [
        context_msg,
        _PATCH_RULES,
        f"\n## Test function:\n```python\n{function_text}\n```\n\n"
        f"## Patch strings under review (no update proposed):\n{paths_list}\n\n",
    ]
    if stable_patch_paths:
        stable_list = "\n".join(f"- `{p}`" for p in stable_patch_paths)
        parts.append(
            "## Patch strings already correct — do not include in corrections:\n"
            f"{stable_list}\n\n"
        )
    if migration_reminder:
        parts.append(migration_reminder)
    if patch_lookup:
        parts.append(patch_lookup)
    if candidates_per_path:
        small_cands = {
            old: cands
            for old, cands in candidates_per_path.items()
            if cands and len(cands) <= _CG_CANDIDATES_LLM_THRESHOLD
        }
        if small_cands:
            parts.append(
                "\n## Call-graph candidate paths (from static analysis):\n"
                "These paths require updates — call-graph found the following "
                "valid new locations:\n"
            )
            for old in sorted(small_cands):
                cands_str = ", ".join(f"`{c}`" for c in small_cands[old])
                parts.append(f"- `{old}` → must be one of: {cands_str}\n")
            parts.append("\n")
    parts.append(
        "The proposed update is: **no changes needed to the reviewed patch "
        "strings**.\n\n"
        "Is this correct? Set `correct` to True only if all reviewed @patch strings "
        "still point to the correct location after the split.\n"
        "Trace the full execution path for each mock call in the test: if the "
        "mock is called multiple times, each call may route through a different "
        "sub-module — a single patch path only intercepts calls in its own "
        "module. Ground your analysis in the test's actual construction — if a "
        "branch is not triggered by the test's inputs (e.g. an empty list skips "
        "an `if` block), do not assume that branch executes.\n"
        "If the patch target lookup shows N is 'used in original module by F', "
        "and the test calls F directly, the original-module patch is correct "
        "regardless of whether F internally delegates to a migrated helper — "
        "parameter-passing does not change where N is called.\n"
        "**Mandatory check for still-imported names:** Before concluding that a "
        "still-imported name N needs relocating to submodule M: look up N in M's "
        "**Name references** section. If F (the function under test) is NOT listed "
        "as a caller of N in M, then F does not call N — it either receives the "
        "resource as a parameter or doesn't use N at all. The mere fact that M "
        "imports N and F lives in M is irrelevant. Patching at the original module "
        "is correct. Proceed to flag it only if F IS listed.\n"
        "**No-op patch rule:** If N appears in 'Names still externally imported "
        "in the modified original' (meaning the name is still importable at the "
        "original path — no AttributeError at test time), AND neither F nor "
        "F's containing class (if F is a method) is listed in the **Name "
        "references** for N in any new submodule (the map records top-level "
        "class and function definitions, not individual methods — a class is "
        "listed when any of its methods call N), then F does not call N and "
        "the patch is a harmless no-op. Set `correct=True` — do NOT reject.\n"
        "When you reject (correct=false): populate `corrections` with "
        "{current_path: corrected_path} for each reviewed string that "
        "needs renaming — use the exact strings from the 'under review' list as keys. "
        "Do **not** include strings from 'already correct' in corrections. "
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
    stable_patch_paths: Optional[List[str]] = None,
    candidates_per_path: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Build the user prompt for the full function rewrite LLM call."""
    paths_list = "\n".join(f"- `{p}`" for p in old_patch_paths)
    parts = [context_msg, _PATCH_RULES]
    if prev_error:
        parts.append(f"\n## Previous rewrite was invalid:\n" f"- Error: {prev_error}\n")
    parts.append(
        f"\n## Test function to rewrite:\n```python\n{function_text}\n```\n\n"
        f"## Patch strings that need updating:\n{paths_list}\n\n"
    )
    if stable_patch_paths:
        stable_list = "\n".join(f"- `{p}`" for p in stable_patch_paths)
        parts.append(
            "## Patch strings already correct — do not modify:\n" f"{stable_list}\n\n"
        )
    if candidates_per_path:
        small_cands = {
            old: cands
            for old, cands in candidates_per_path.items()
            if cands and len(cands) <= _CG_CANDIDATES_LLM_THRESHOLD
        }
        if small_cands:
            parts.append(
                "## Call-graph candidate paths (from static analysis):\n"
                "Static analysis narrowed the valid new locations for each path:\n"
            )
            for old in sorted(small_cands):
                cands_str = ", ".join(f"`{c}`" for c in small_cands[old])
                parts.append(f"- `{old}` → must be one of: {cands_str}\n")
            parts.append("\n")
    parts.append(
        "Rewrite the complete function, updating ONLY the patch strings listed "
        "under 'Patch strings that need updating'. Do **not** modify any @patch "
        "decorator listed under 'Patch strings already correct'. "
        "You may add or remove @patch decorators only when specifically required "
        "by the rules below. Preserve all original test logic. Return the "
        "complete function including all decorators and body.\n"
        "**When to add a new @patch decorator**: add one only when the test "
        "**directly exercises** code in multiple sub-modules that independently "
        "import the same name — for example, the function under test explicitly "
        "calls both `module_a.process()` and `module_b.validate()`, and both "
        "import `fetch_data`. Do NOT add a second @patch based solely on the "
        "mock's side_effect count or call_count: multiple calls may all route "
        "through the same sub-module. Each sub-module that is provably exercised "
        "needs its own @patch. Add a new parameter for each new decorator in the "
        "same decorator→parameter order (@patch decorators stack bottom-up).\n"
        "**When to remove a @patch decorator**: remove one when the patched "
        "name N is not imported in the module where the function under test is "
        "defined AND the function receives the corresponding dependency as a "
        "direct argument rather than calling a factory. For example: if "
        "`@patch('app.worker.build_session')` exists but `worker.py` does not "
        "import `build_session` and the function under test `execute_task` "
        "receives `session` as a direct parameter, the @patch is a no-op — "
        "remove it and remove its corresponding mock parameter. Do not keep "
        "no-op patches to 'preserve test logic'; removing them IS preserving "
        "the logic since they had no effect.\n"
    )
    return "".join(parts)


def _build_rewrite_verify_prompt(
    context_msg: str,
    original_function_text: str,
    rewritten_function_text: str,
    candidates_per_path: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Build the user prompt for a full-rewrite verify LLM call."""
    parts = [
        context_msg,
        _PATCH_RULES,
        f"\n## Original test function:\n```python\n{original_function_text}\n```\n\n"
        f"## Rewritten test function:\n```python\n{rewritten_function_text}\n```\n\n",
    ]
    if candidates_per_path:
        small_cands = {
            old: cands
            for old, cands in candidates_per_path.items()
            if cands and len(cands) <= _CG_CANDIDATES_LLM_THRESHOLD
        }
        if small_cands:
            parts.append(
                "## Call-graph candidate paths (from static analysis):\n"
                "Static analysis narrowed the valid new locations for each path:\n"
            )
            for old in sorted(small_cands):
                cands_str = ", ".join(f"`{c}`" for c in small_cands[old])
                parts.append(f"- `{old}` → must be one of: {cands_str}\n")
            parts.append("\n")
    parts.append(
        "Verify that the rewrite is correct:\n"
        "- All @patch strings point to where the name is looked up after the split.\n"
        "- **Multi-submodule completeness**: require an additional @patch for a "
        "second sub-module only when the test directly exercises that "
        "sub-module's code path. Do NOT reject a rewrite that patches a single "
        "sub-module solely because the mock's side_effect count is ≥2 or "
        "candidates span ≥2 sub-modules: multiple calls may all route through "
        "the same sub-module. Accept a single @patch when there is no direct "
        "evidence in the test (from the function under test's code or the "
        "test's setup/inputs) that the other sub-module's path is reached.\n"
        "- **Removed @patch decorators**: accept a rewrite that removes a "
        "@patch for name N when N is not imported in the module where the "
        "function under test is defined AND the function receives the "
        "dependency as a direct parameter. For example: removing "
        "`@patch('app.worker.build_session')` is correct when `worker.py` "
        "does not import `build_session` and the function under test receives "
        "`session` as a parameter — the patch was a no-op. Do not set "
        "`correct=False` for removals of genuinely no-op patches.\n"
        "- All mock parameters correspond correctly to their @patch decorators "
        "(order, count, and names).\n"
        "- All original test logic is preserved — no hallucinated code, no "
        "missing assertions or setup.\n"
        "Set `correct` to True only if all of the above are satisfied.\n",
    )
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
# Candidates constraint check
# ---------------------------------------------------------------------------


def _candidates_check(
    patch_renames: Dict[str, str],
    old_patch_paths: List[str],
    candidates: Dict[str, List[str]],
) -> Optional[str]:
    """Return a rejection reason if LLM output conflicts with call-graph candidates.

    *candidates* maps each old patch path to a list of valid new paths found by
    the call-graph BFS (populated only when multiple candidates were found and
    the BFS was not truncated by limits).

    Returns ``None`` when the output is consistent with candidates, or when no
    candidates are available for the paths in question.
    """
    for old in old_patch_paths:
        cands = candidates.get(old)
        if not cands:
            continue
        proposed_new = patch_renames.get(old)
        if proposed_new is None:
            # No rename proposed. If the original path is itself one of the
            # candidates (e.g. an import alias re-exported by the new __init__.py
            # so it remains accessible at the same dotted path), keeping the patch
            # unchanged is correct — don't treat it as an error.
            if old in cands:
                continue
            return (
                f"`{old}` requires an update — call-graph found it moved to: "
                + ", ".join(f"`{c}`" for c in cands)
            )
        if proposed_new not in cands:
            return (
                f"Proposed `{proposed_new}` for `{old}` is not among the "
                "call-graph candidates: " + ", ".join(f"`{c}`" for c in cands)
            )
    return None


# Matches the first string arg of any ``patch(...)`` or ``X.patch(...)`` call.
_PATCH_ARG_RE = re.compile(r'\bpatch\s*\(\s*["\']([^"\']+)["\']')


def _patch_strings_in_text(text: str) -> Set[str]:
    """Return all first-argument string values from ``patch(...)`` calls in *text*."""
    return set(_PATCH_ARG_RE.findall(text))


def _rewrite_candidates_check(
    old_patch_paths: List[str],
    new_func_text: str,
    candidates: Dict[str, List[str]],
) -> Optional[str]:
    """Verify rewritten function's patch strings against call-graph candidates.

    Extracts patch string values from *new_func_text* and checks each old path
    that has candidates:

    - If the old path is still present → the path was not updated, but it must be.
    - If the old path is absent and none of its candidates appear → the patch was
      removed entirely (possible dead-code removal); let the LLM verify step decide.

    Returns ``None`` when the rewrite is consistent with candidates, or an error
    string describing the first problem found.
    """
    new_strings = _patch_strings_in_text(new_func_text)
    for old in old_patch_paths:
        cands = candidates.get(old)
        if not cands:
            continue
        if old in new_strings:
            return (
                f"`{old}` was not updated — call-graph found it moved to: "
                + ", ".join(f"`{c}`" for c in cands)
            )
        # Old path was removed. If a known candidate now appears, it was correctly
        # relocated. If no candidate appears, the decorator may have been removed
        # entirely (e.g. the patch was dead code). Allow both through — the LLM
        # verify step assesses correctness.
    return None


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
    moved_out_names: Optional[Set[str]] = None,
    still_imported: Optional[Set[str]] = None,
    orig_users_map: Optional[Dict[str, List[str]]] = None,
    new_module_imports: Optional[Dict[str, Set[str]]] = None,
    cg_candidates: Optional[Dict[str, Dict[str, List[str]]]] = None,
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
    # Maps old_val → set of voted values.  Every test function that uses the
    # constant casts a vote: a renamed test votes for its new_val; a test whose
    # rename was blocked or absent votes for the old_val (keep-as-is).
    # Only when ALL votes agree on ONE new value (and that value differs from the
    # old value) is the constant definition updated — all tests then implicitly
    # pick up the change.  Whenever any test casts a "keep old" vote alongside a
    # rename vote, the proposals conflict (len > 1) → each affected function is
    # inlined individually instead.
    same_file_proposals: Dict[str, Set[str]] = {}
    same_file_const_map: Dict[str, str] = {}  # populated after conflict resolution

    # Use caller-supplied guard sets (derived from _FLContext objects directly).
    _moved_out_names: Set[str] = (
        moved_out_names if moved_out_names is not None else set()
    )
    _orig_users_map: Dict[str, List[str]] = (
        orig_users_map if orig_users_map is not None else {}
    )
    _still_imported: Set[str] = still_imported if still_imported is not None else set()
    _new_module_imports: Optional[Dict[str, Set[str]]] = new_module_imports

    for func in functions:
        # Per-function call-graph candidates: old_path → sorted list of new paths.
        # Only populated when BFS found multiple candidates without hitting limits.
        func_candidates: Dict[str, List[str]] = (cg_candidates or {}).get(
            func.function_name, {}
        )
        prev_issue: Optional[str] = None
        prev_proposed: Optional[str] = None
        attempts_left = max_attempts
        rename_verify_retries_left = config.llm_verify_retries
        # When the verify step identifies a required change the classify path
        # cannot produce, escalate: skip the next classify call and go directly
        # to the full rewrite path, seeding it with the verifier's explanation.
        _rewrite_escalation_error: Optional[str] = None
        _rewrite_escalation_reason: str = ""
        r = None  # may be skipped when escalating
        _outcome: Optional[str] = None  # "no_change", "rename", or "rewrite"

        while attempts_left > 0:
            attempts_left -= 1

            if _rewrite_escalation_error is not None:
                # Bypass classify; the verifier identified a required change
                # the classify path could not produce — hand the explanation
                # straight to the rewrite path.
                needs_rewrite = True
                if verbose:
                    print(
                        f"crispen: patch_rewriter: escalating to rewrite for"
                        f" '{func.function_name}' in {file_desc}"
                        f" ({_rewrite_escalation_reason})",
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
                    stable_patch_paths=func.stable_patch_paths or None,
                    candidates_per_path=func_candidates or None,
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
                _rw_ok = False
                while rewrite_attempts > 0:
                    rewrite_attempts -= 1
                    rewrite_prompt = _build_rewrite_func_prompt(
                        context_msg,
                        func.full_text,
                        func.old_patch_paths,
                        prev_error,
                        stable_patch_paths=func.stable_patch_paths or None,
                        candidates_per_path=func_candidates or None,
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
                    # Candidates pre-check: algorithmic validation before LLM verify.
                    # Use the raw string-literal form (before constant restoration)
                    # so the regex-based checker can match patch paths correctly.
                    if func_candidates:
                        rw_cand_issue = _rewrite_candidates_check(
                            func.old_patch_paths, new_func_text, func_candidates
                        )
                        if rw_cand_issue:
                            prev_error = rw_cand_issue
                            continue
                    # LLM verify step.  Pass the raw string-literal form for both
                    # original and rewrite so the verifier can compare them without
                    # needing to resolve named constants (e.g. _PATCH_CLIENT).
                    rewrite_verify_prompt = _build_rewrite_verify_prompt(
                        context_msg,
                        func.full_text,
                        new_func_text,
                        candidates_per_path=func_candidates or None,
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
                        # Restore @patch constant references only after the
                        # rewrite is accepted, so the verifier always compares
                        # string-literal forms in both original and rewrite.
                        if func.const_refs:
                            new_func_text = _restore_const_refs(
                                new_func_text, func.const_refs
                            )
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
                        _rw_ok = True
                        break
                    rv_issue = rv.tool_input.get("issue", "") if rv.tool_input else ""
                    if rewrite_verify_retries_left > 0:
                        rewrite_verify_retries_left -= 1
                        prev_error = rv_issue or "LLM verify rejected the rewrite."
                        rewrite_attempts += 1  # don't burn compile retry budget
                        continue
                    # verify retries exhausted — skip this function
                if _rw_ok:
                    _outcome = "rewrite"
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

            # Pre-filter: drop renames that are structurally impossible regardless
            # of which function the test exercises.  Running this before the verify
            # call prevents the verifier from being confused by impossible renames
            # in the proposed set, and — if everything is dropped — redirects
            # naturally into the no-change verify path so a retry can fire.
            if patch_renames and (
                _moved_out_names or _still_imported or _new_module_imports
            ):
                patch_renames = {
                    old: new
                    for old, new in patch_renames.items()
                    if not _is_bad_rename(
                        old,
                        new,
                        _moved_out_names,
                        _still_imported,
                        _orig_users_map,
                        func.full_text,
                        _new_module_imports,
                    )
                }

            # Candidates pre-check: if the call-graph found valid candidates for a
            # path (multiple new homes, BFS not truncated), verify the LLM's proposal
            # matches one of them.  If not, reject and retry without calling LLM verify.
            if func_candidates:
                cand_issue = _candidates_check(
                    patch_renames, func.old_patch_paths, func_candidates
                )
                if cand_issue:
                    if verbose:
                        print(
                            f"crispen: patch_rewriter: candidates check rejected:"
                            f" {cand_issue}",
                            file=sys.stderr,
                            flush=True,
                        )
                    if attempts_left == 0:
                        # All classify retries exhausted with persistent candidates
                        # check failure.  The old path is known-broken; escalate to
                        # full rewrite so the function body can be analysed to
                        # determine which sub-module(s) the test exercises.
                        _rewrite_escalation_error = cand_issue
                        _rewrite_escalation_reason = (
                            "candidates check retries exhausted"
                        )
                        attempts_left += 1  # allow one more outer iteration
                    else:
                        prev_issue = cand_issue
                        prev_proposed = (
                            str(patch_renames) if patch_renames else "no change"
                        )
                    continue

            if not patch_renames:
                # Verify the "no change needed" conclusion.
                no_change_verify_prompt = _build_no_change_verify_prompt(
                    context_msg,
                    func.full_text,
                    func.old_patch_paths,
                    stable_patch_paths=func.stable_patch_paths or None,
                    candidates_per_path=func_candidates or None,
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
                    _outcome = "no_change"
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
                    _outcome = "no_change"
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
                # _is_bad_rename blocks patterns A–C:
                #   A) shallowing a moved-out name (would raise AttributeError)
                #   B) deepening a still-in name when the test exercises the
                #      original-module caller (wrong binding intercepted)
                #   C) target module doesn't import the name
                # The second filter retains a guard: drop any still-in name
                # that isn't being deepened into a true sub-module.
                if _still_imported or _moved_out_names or _new_module_imports:
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
                            _new_module_imports,
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
                        context_msg,
                        func.full_text,
                        corrections_renames,
                        candidates_per_path=func_candidates or None,
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
                        _outcome = "rename"
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
                    _rewrite_escalation_reason = "no-change verify retries exhausted"
                    attempts_left += 1  # allow one more outer iteration
                    continue
                _outcome = "no_change"
                break  # llm_verify_retries=0 → accept no-change

            # Verify the renames.
            verify_prompt = _build_func_verify_prompt(
                context_msg,
                func.full_text,
                patch_renames,
                candidates_per_path=func_candidates or None,
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
                        _new_module_imports,
                    )
                }
                orig_text = "\n".join(source_lines[func.start_line - 1 : func.end_line])
                new_text = apply_patch_strings(orig_text, patch_renames_safe)
                if new_text != orig_text:
                    func_splices.append((func.start_line, func.end_line, new_text))
                string_swap_results.append((func, patch_renames_safe))
                _outcome = "rename"
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
                        _new_module_imports,
                    )
                }
                orig_text = "\n".join(source_lines[func.start_line - 1 : func.end_line])
                new_text = apply_patch_strings(orig_text, patch_renames_safe)
                if new_text != orig_text:
                    func_splices.append((func.start_line, func.end_line, new_text))
                string_swap_results.append((func, patch_renames_safe))
                _outcome = "rename"
                break
            else:
                if rename_verify_retries_left > 0:
                    rename_verify_retries_left -= 1
                    prev_issue = issue
                    prev_proposed = str(patch_renames)
                    attempts_left += 1  # don't burn classify retry budget
                elif config.llm_verify_retries > 0:
                    _rewrite_escalation_error = issue
                    _rewrite_escalation_reason = "rename verify retries exhausted"
                    attempts_left += 1  # allow one more outer iteration
                # else (llm_verify_retries=0): retries exhausted — skip

        if _acc is not None:
            if _outcome == "no_change":
                _acc.no_change += 1
            elif _outcome == "rename":
                _acc.rename += 1
            elif _outcome == "rewrite":
                _acc.rewrite += 1
            else:
                _acc.edit_failures += 1

    # Collect cross-file and same-file constant proposals.
    cross_file_patch_maps: Dict[str, Dict[str, str]] = {}
    if string_swap_results and scan_file:
        scan_file_abs = str(Path(scan_file).resolve())
        for func, accepted in string_swap_results:
            for ref in func.const_refs:
                new_val = accepted.get(ref.resolved_value)
                if ref.source_file == scan_file_abs:
                    # Every test casts a vote: rename → new_val; blocked/absent → old
                    # value (keep-as-is).  This prevents a minority rename proposal
                    # from silently winning when the majority of tests had their
                    # rename blocked and thus implicitly prefer the old value.
                    vote = (
                        new_val
                        if new_val is not None and new_val != ref.resolved_value
                        else ref.resolved_value
                    )
                    same_file_proposals.setdefault(ref.resolved_value, set()).add(vote)
                elif new_val is not None and new_val != ref.resolved_value:
                    cross_file_patch_maps.setdefault(ref.source_file, {})[
                        ref.resolved_value
                    ] = new_val

        # Resolve same-file proposals into updates and per-function inlines.
        # All votes agree on ONE new value AND it differs from the old value →
        # update the constant definition.  Any "keep old" vote alongside a rename
        # vote produces len(new_set) > 1 → inline each affected function instead.
        same_file_const_map = {
            old: next(iter(new_set))
            for old, new_set in same_file_proposals.items()
            if len(new_set) == 1 and next(iter(new_set)) != old
        }
        conflicting_old_vals = {
            old for old, new_set in same_file_proposals.items() if len(new_set) > 1
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
# Call-graph algorithmic resolution
# ---------------------------------------------------------------------------


def _callgraph_update_file(
    source: str,
    all_forking_paths: Set[str],
    fl_contexts: List[_FLContext],
    scan_file: str = "",
    repo_root: Optional[str] = None,
    index: Optional[_CgIndex] = None,
    verbose: bool = False,
    max_depth: int = _CG_MAX_DEPTH,
    max_modules: int = _CG_MAX_MODULES,
    _acc: Optional["RewriteAccumulator"] = None,
) -> Tuple[str, bool, Dict[str, Dict[str, List[str]]]]:
    """Apply call-graph-resolved @patch updates to *source*.

    For each test function with forking @patch targets, attempts to determine
    the correct new sub-module for each target via BFS call-graph tracing
    using *index*.  Only updates a target when exactly one new sub-module in
    the call graph uses it.

    Returns ``(updated_source, was_changed, unresolved_candidates)`` where
    *unresolved_candidates* maps ``{func_name: {old_path: sorted_candidates}}``
    for paths that were ambiguous (multiple candidates, BFS not truncated).
    """
    functions = _find_test_functions_to_update(
        source, all_forking_paths, scan_file, repo_root
    )
    if not functions:
        return source, False, {}

    source_lines = source.splitlines()
    func_splices: List[Tuple[int, int, str]] = []
    same_file_proposals: Dict[str, Set[str]] = {}
    resolved_results: List[Tuple[_TestFunctionInfo, Dict[str, str]]] = []
    # func_name → old_path → sorted candidates (multiple, non-truncated BFS results)
    unresolved_candidates: Dict[str, Dict[str, List[str]]] = {}
    scan_file_abs = str(Path(scan_file).resolve()) if scan_file else ""
    calling_module = ""
    if index is not None and scan_file_abs:
        calling_module = index.file_to_module.get(scan_file_abs, "")

    for func in functions:
        # Const-ref old_vals for forking paths (handled via proposals, not string swap).
        const_ref_vals: Set[str] = {
            ref.resolved_value
            for ref in func.const_refs
            if ref.resolved_value in all_forking_paths
        }

        # Attempt call-graph resolution for each forking old path.
        # old_patch_paths contains only forking paths by construction.
        resolved: Dict[str, str] = {}
        for old_path in func.old_patch_paths:
            name = old_path.rsplit(".", 1)[-1]
            for ctx in fl_contexts:
                if old_path not in ctx.forking_old_paths:
                    continue
                if index is None:
                    new_path, cands, truncated, static_cands = None, [], False, []
                else:
                    new_path, cands, truncated, static_cands = (
                        _resolve_forking_path_candidates(
                            name,
                            func.full_text,
                            ctx,
                            index,
                            calling_module,
                            max_depth=max_depth,
                            max_modules=max_modules,
                        )
                    )
                if new_path is not None:
                    resolved[old_path] = new_path
                    # Clear any previously saved candidates for this path.
                    func_cands = unresolved_candidates.get(func.function_name, {})
                    func_cands.pop(old_path, None)
                    if not func_cands and func.function_name in unresolved_candidates:
                        del unresolved_candidates[func.function_name]
                    break
                # Multiple BFS candidates without truncation → save for rewrite mode.
                if truncated:
                    print(
                        f"crispen: patch_callgraph: traversal limit reached for"
                        f" '{old_path}' in '{scan_file}'"
                        f" (max_depth={max_depth}, max_modules={max_modules})"
                        f" — resolution skipped",
                        file=sys.stderr,
                        flush=True,
                    )
                elif len(cands) > 1:
                    unresolved_candidates.setdefault(func.function_name, {})[
                        old_path
                    ] = cands
                elif len(cands) == 0 and static_cands:
                    # BFS found no reachable candidates; fall back to the full
                    # static terminal set (all new modules that use the entity).
                    if len(static_cands) == 1:
                        resolved[old_path] = static_cands[0]
                        func_cands = unresolved_candidates.get(func.function_name, {})
                        func_cands.pop(old_path, None)
                        if (
                            not func_cands
                            and func.function_name in unresolved_candidates
                        ):
                            del unresolved_candidates[func.function_name]
                        break
                    else:
                        unresolved_candidates.setdefault(func.function_name, {})[
                            old_path
                        ] = static_cands

        if not resolved:
            continue

        # Track const proposals.
        for old_val in const_ref_vals:
            if old_val in resolved:
                same_file_proposals.setdefault(old_val, set()).add(resolved[old_val])

        # Apply STRING LITERAL resolutions (not const refs) as a func splice.
        string_res = {
            old: new for old, new in resolved.items() if old not in const_ref_vals
        }
        orig_text = "\n".join(source_lines[func.start_line - 1 : func.end_line])
        new_text = (
            apply_patch_strings(orig_text, string_res) if string_res else orig_text
        )
        if new_text != orig_text:
            func_splices.append((func.start_line, func.end_line, new_text))

        if _acc is not None:
            _acc.cg_resolved += len(resolved)
        resolved_results.append((func, resolved))
        if verbose:
            print(
                f"crispen: patch_callgraph: resolved '{func.function_name}'"
                f" in '{scan_file}'",
                file=sys.stderr,
                flush=True,
            )

    # Resolve same-file const proposals into definition updates or per-function inlines.
    # When all proposals agree on ONE new value, update the constant definition even if
    # some functions passed through unchanged — they continue using the const and
    # implicitly get the new value.  Only inline per-function when proposals disagree.
    same_file_const_map: Dict[str, str] = {
        old: next(iter(new_set))
        for old, new_set in same_file_proposals.items()
        if len(new_set) == 1
    }
    conflicting = {
        old for old, new_set in same_file_proposals.items() if len(new_set) > 1
    }

    # Inline conflicting const resolutions per function.
    if conflicting:
        for func, accepted in resolved_results:
            inline_subs: Dict[str, str] = {}
            for ref in func.const_refs:
                if (
                    ref.source_file == scan_file_abs
                    and ref.resolved_value in conflicting
                    and ref.resolved_value in accepted
                ):
                    new_val = accepted[ref.resolved_value]
                    if new_val != ref.resolved_value:
                        inline_subs[ref.const_name] = new_val
            if not inline_subs:
                continue
            orig_text = "\n".join(source_lines[func.start_line - 1 : func.end_line])
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
                func_splices[existing_idx] = (func.start_line, func.end_line, inlined)
            else:
                func_splices.append((func.start_line, func.end_line, inlined))

    if not func_splices and not same_file_const_map:
        return source, False, unresolved_candidates

    result_source = source
    for start_line, end_line, new_text in sorted(func_splices, key=lambda x: -x[0]):
        result_source = _splice_function(result_source, start_line, end_line, new_text)
    if same_file_const_map:
        result_source = apply_patch_strings(result_source, same_file_const_map)
    return result_source, result_source != source, unresolved_candidates


def apply_patch_callgraph(
    fl_contexts: List[_FLContext],
    per_file: Dict[str, Any],
    repo_root: Optional[str],
    verbose: bool = False,
    candidates_out: Optional[Dict[str, Dict[str, Dict[str, List[str]]]]] = None,
    config: Optional["CrispenConfig"] = None,
    _acc: Optional["RewriteAccumulator"] = None,
) -> Iterator[str]:
    """Algorithmically resolve forking @patch paths via call-graph tracing.

    For each @patch decorator referencing a forking old path, finds which new
    sub-module in the call graph of the test function imports and uses the
    patched name.  When exactly one such sub-module exists the decorator is
    updated without any LLM involvement.

    When *candidates_out* is provided it is populated with unresolved-but-ambiguous
    results: ``{abs_filepath: {func_name: {old_path: sorted_candidates}}}``.
    Only paths where BFS found multiple candidates without hitting traversal limits
    are recorded.  This data can be passed to :func:`apply_patch_rewrite` so the
    LLM receives a constrained multiple-choice question instead of a free-form one.

    Should be called after "basic" ``apply_patch_strings`` so already-resolved
    paths are not re-processed.  For "rewrite" mode the updated file sources
    will not contain the resolved old paths any more, so ``apply_patch_rewrite``
    only processes what call-graph tracing left unresolved.
    """
    if not fl_contexts:
        return

    all_forking_paths: Set[str] = set()
    for ctx in fl_contexts:
        all_forking_paths |= ctx.forking_old_paths

    if not all_forking_paths:
        return

    max_depth = config.callgraph_max_depth if config is not None else _CG_MAX_DEPTH
    max_modules = (
        config.callgraph_max_modules if config is not None else _CG_MAX_MODULES
    )

    # Build repo-wide index once; per_file in-memory sources override disk.
    per_file_sources = {
        str(Path(fp).resolve()): state["source"] for fp, state in per_file.items()
    }
    index = _cg_build_index(repo_root, per_file_sources, fl_contexts)

    per_file_abs = {str(Path(f).resolve()) for f in per_file}

    for filepath, state in per_file.items():
        file_src = state["source"]
        relevant_contexts = [
            ctx
            for ctx in fl_contexts
            if any(path in file_src for path in ctx.forking_old_paths)
        ]
        if not relevant_contexts:
            continue
        new_src, changed, unresolved = _callgraph_update_file(
            file_src,
            all_forking_paths,
            relevant_contexts,
            scan_file=filepath,
            repo_root=repo_root,
            index=index,
            verbose=verbose,
            max_depth=max_depth,
            max_modules=max_modules,
            _acc=_acc,
        )
        if changed:
            state["source"] = new_src
            state["msgs"].append(
                f"{filepath}: patch_update: updated @patch strings (call-graph)"
            )
        if unresolved and candidates_out is not None:
            abs_fp = str(Path(filepath).resolve())
            candidates_out[abs_fp] = unresolved

    if repo_root is None:
        return

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
        if not relevant_contexts:
            continue
        new_src, changed, unresolved = _callgraph_update_file(
            src,
            all_forking_paths,
            relevant_contexts,
            scan_file=str(py_file),
            repo_root=repo_root,
            index=index,
            verbose=verbose,
            max_depth=max_depth,
            max_modules=max_modules,
            _acc=_acc,
        )
        if changed:
            py_file.write_text(new_src, encoding="utf-8")
            yield f"{py_file}: patch_update: updated @patch strings (call-graph)"
        if unresolved and candidates_out is not None:
            candidates_out[str(py_file.resolve())] = unresolved


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
    cg_candidates: Optional[Dict[str, Dict[str, Dict[str, List[str]]]]] = None,
) -> Iterator[str]:
    """Update @patch strings for forking entities using LLM.

    Called after "basic" patch updates have already been applied.  Handles
    entities that basic mode skipped because they appeared in multiple callers.
    Also resolves named constants used as @patch arguments and updates their
    definitions when all usages agree on the same new value.

    When *cg_candidates* is provided (populated by :func:`apply_patch_callgraph`),
    per-file call-graph candidate lists are passed to the LLM prompts and used to
    pre-validate LLM output: ``{abs_filepath: {func_name: {old_path: candidates}}}``.
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
        (
            _moved_out,
            _still_in,
            _orig_users,
            _new_mod_imports,
        ) = _build_rename_guard_sets(relevant_contexts)
        abs_fp = str(Path(filepath).resolve())
        file_cands = (cg_candidates or {}).get(abs_fp) or None
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
            moved_out_names=_moved_out,
            still_imported=_still_in,
            orig_users_map=_orig_users,
            new_module_imports=_new_mod_imports,
            cg_candidates=file_cands,
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
        (
            _moved_out,
            _still_in,
            _orig_users,
            _new_mod_imports,
        ) = _build_rename_guard_sets(relevant_contexts)
        file_cands = (cg_candidates or {}).get(str(py_file.resolve())) or None
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
            moved_out_names=_moved_out,
            still_imported=_still_in,
            orig_users_map=_orig_users,
            new_module_imports=_new_mod_imports,
            cg_candidates=file_cands,
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
