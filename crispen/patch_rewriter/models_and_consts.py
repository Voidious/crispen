from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import ast
import libcst as cst


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
