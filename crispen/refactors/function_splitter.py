"""Refactor: split functions exceeding the line-count limit."""

from __future__ import annotations

import ast
import builtins
import textwrap
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from .. import llm_client as _llm_client
from .base import Refactor

_MODEL = "claude-sonnet-4-6"
_API_HARD_TIMEOUT = 90  # seconds per LLM call
_MAX_SPLIT_ITERATIONS = 100  # prevent infinite recursion
_MAX_SPLIT_CANDIDATES = 5  # split points tried per function

_BUILTINS: frozenset = frozenset(dir(builtins))


def _module_global_names(source: str) -> set:
    """Return all names defined at module level in *source*.

    Covers function/class definitions, imports, and simple (annotated)
    assignments.  Module-level names are always accessible inside helpers
    without being passed as parameters, so they must be excluded from the
    helper's param list.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


# ---------------------------------------------------------------------------
# Hard-timeout helper
# ---------------------------------------------------------------------------


class _ApiTimeout(Exception):
    """Raised when an LLM API call exceeds the hard per-call timeout."""


def _run_with_timeout(func, timeout, *args, **kwargs):
    """Run *func* in a daemon thread; raise _ApiTimeout if it doesn't finish."""
    result: list = [None]
    exc: list = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except BaseException as e:
            exc[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise _ApiTimeout(f"API call exceeded {timeout}s hard limit")
    if exc[0] is not None:
        raise exc[0]
    return result[0]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class _FuncInfo:
    node: cst.FunctionDef
    start_line: int  # includes decorators
    end_line: int
    class_name: Optional[str]  # None = module-level
    indent: str  # leading whitespace of def line (e.g. "    ")
    original_params: List[str]  # e.g. ['self', 'data', 'config']


@dataclass
class _SplitTask:
    func_info: _FuncInfo
    split_idx: int  # index into body_stmts where tail begins
    params: List[str]  # free variables of tail (excluding 'self' for instance methods)
    tail_source: str = ""  # dedented tail source, set before LLM call
    helper_name: str = ""  # filled in by LLM
    is_instance_method: bool = False  # True when helper must be a regular method


# ---------------------------------------------------------------------------
# LLM tool definition
# ---------------------------------------------------------------------------


_NAME_TOOL: dict = {
    "name": "name_helper_functions",
    "description": (
        "Suggest concise private names for extracted Python helper functions"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "names": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {
                            "type": "string",
                            "description": "snake_case, no leading underscore",
                        },
                    },
                    "required": ["id", "name"],
                },
            }
        },
        "required": ["names"],
    },
}


# ---------------------------------------------------------------------------
# CST helpers
# ---------------------------------------------------------------------------


def _is_docstring_stmt(stmt: cst.BaseStatement) -> bool:
    """Return True if *stmt* is a docstring expression statement."""
    if not isinstance(stmt, cst.SimpleStatementLine):
        return False
    if len(stmt.body) != 1:
        return False
    expr = stmt.body[0]
    if not isinstance(expr, cst.Expr):
        return False
    return isinstance(expr.value, (cst.SimpleString, cst.ConcatenatedString))


def _has_yield(func_node: cst.FunctionDef) -> bool:
    """Return True if func_node body contains Yield (not in nested functions)."""

    def _walk(node: cst.CSTNode) -> bool:
        if isinstance(node, cst.Yield):
            return True
        if isinstance(node, cst.FunctionDef):
            return False  # don't recurse into nested functions
        for child in node.children:
            if _walk(child):
                return True
        return False

    return _walk(func_node.body)


def _has_nested_funcdef(node: cst.FunctionDef) -> bool:
    """Return True if node's body contains a top-level nested FunctionDef.

    Functions with nested def statements create closures.  The existing
    split-point guard in ``_find_valid_splits`` already restricts splits to
    before the first nested def, but that still allows a split where the
    *entire* nested-def block ends up in the extracted tail.  The resulting
    helper is itself oversized and closure-entangled, triggering a recursive
    cascade of further splits.  Skipping such functions entirely—consistent
    with the async and generator exclusions—avoids the cascade.
    """
    return any(isinstance(stmt, cst.FunctionDef) for stmt in node.body.body)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _count_body_lines(func_source: str) -> int:
    """Count non-docstring body lines of the first function in func_source."""
    try:
        tree = ast.parse(func_source)
    except SyntaxError:
        return 0

    func = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node
            break
    if func is None or not func.body:
        return 0

    body = func.body
    first_idx = 0
    # Skip docstring
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        first_idx = 1

    if first_idx >= len(body):
        return 0

    first_stmt = body[first_idx]
    last_stmt = body[-1]
    return last_stmt.end_lineno - first_stmt.lineno + 1


def _find_free_vars(tail_source: str) -> List[str]:
    """Return sorted free variables of tail_source (excluding builtins).

    Uses scope-aware analysis with sequential block tracking so that
    write-before-read patterns within any nested block do not produce
    false positives.  Specifically:

    * Statements are walked in order at every nesting level.  A variable
      assigned before it is read within the same block is not free.
    * ``for``-loop targets, ``with``-statement targets, and
      ``except``-handler names are locally scoped within their body.
    * Augmented assignments (``x += 1``) treat the target as a *load*
      because the current value of the variable must exist beforehand.
    * Assignments in a ``try`` body propagate to code after the try block
      (optimistic: if execution reaches past the try, the body completed).
    * Assignments in ``if``, ``for``, ``while``, and ``with`` bodies do
      NOT propagate to after those constructs (conditional execution).
    * Does not recurse into nested ``def``/``class`` bodies.
    """
    try:
        tree = ast.parse(tail_source)
    except SyntaxError:
        return []

    loads: set = set()

    def _target_names(target) -> set:
        """Collect all plain Name identifiers bound by an assignment target."""
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            names: set = set()
            for elt in target.elts:
                names |= _target_names(elt)
            return names
        return set()

    def _update_scope(stmt, defined: set) -> None:
        """Widen *defined* with names unconditionally defined by *stmt*."""
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                defined |= _target_names(target)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(stmt.name)
        elif isinstance(stmt, ast.ClassDef):
            defined.add(stmt.name)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                defined.add(alias.asname if alias.asname else alias.name.split(".")[0])
        elif (
            isinstance(stmt, ast.AnnAssign)
            and stmt.value is not None
            and isinstance(stmt.target, ast.Name)
        ):
            defined.add(stmt.target.id)
        # AugAssign, For, While, If, Try, With, Delete: NOT added.

    def _walk_block(stmts, defined: set) -> set:
        """Walk *stmts* sequentially; collect loads; return updated scope."""
        local = set(defined)
        for stmt in stmts:
            _collect_loads(stmt, local)
            _update_scope(stmt, local)
        return local

    def _collect_loads(node, defined: set) -> None:
        """Recursively collect Name loads not already in *defined*.

        *defined* is the set of names definitely locally bound at this
        point.  Does not recurse into nested function/class definitions.
        """
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return  # don't recurse into nested scopes
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return  # imports only produce stores
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id not in defined:
                loads.add(node.id)
            return
        if isinstance(node, ast.AugAssign):
            # ``x += expr`` reads x before writing — treat target as a load.
            if isinstance(node.target, ast.Name) and node.target.id not in defined:
                loads.add(node.target.id)
            _collect_loads(node.target, defined)
            _collect_loads(node.value, defined)
            return
        if isinstance(node, ast.For):
            _collect_loads(node.iter, defined)
            local = defined | _target_names(node.target)
            _walk_block(node.body, local)
            _walk_block(node.orelse, defined)
            return
        if isinstance(node, ast.While):
            _collect_loads(node.test, defined)
            _walk_block(node.body, defined)
            _walk_block(node.orelse, defined)
            return
        if isinstance(node, ast.If):
            _collect_loads(node.test, defined)
            _walk_block(node.body, defined)
            _walk_block(node.orelse, defined)
            return
        if isinstance(node, ast.Try):
            local_try = _walk_block(node.body, defined)
            # Propagate try-body definitions to the outer scope: if execution
            # reaches past the try block the try body must have completed.
            defined |= local_try - defined
            for handler in node.handlers:
                _collect_loads(handler, defined)
            _walk_block(node.orelse, local_try)
            _walk_block(node.finalbody, defined)
            return
        if isinstance(node, ast.With):
            local = set(defined)
            for item in node.items:
                _collect_loads(item.context_expr, defined)
                if item.optional_vars:
                    local |= _target_names(item.optional_vars)
            _walk_block(node.body, local)
            return
        if isinstance(node, ast.ExceptHandler):
            local = set(defined)
            if node.name:
                local.add(node.name)
            if node.type is not None:
                _collect_loads(node.type, defined)
            _walk_block(node.body, local)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            _collect_comp(node.generators, [node.elt], defined)
            return
        if isinstance(node, ast.DictComp):
            _collect_comp(node.generators, [node.key, node.value], defined)
            return
        if isinstance(node, ast.Lambda):
            local = set(defined)
            args = node.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs:
                local.add(arg.arg)
            if args.vararg:
                local.add(args.vararg.arg)
            if args.kwarg:
                local.add(args.kwarg.arg)
            # Default values are evaluated in the enclosing (outer) scope.
            for default in args.defaults:
                _collect_loads(default, defined)
            for kw_default in args.kw_defaults:
                if kw_default is not None:
                    _collect_loads(kw_default, defined)
            _collect_loads(node.body, local)
            return
        for child in ast.iter_child_nodes(node):
            _collect_loads(child, defined)

    def _collect_comp(generators, elts, defined: set) -> None:
        """Walk a comprehension's generators then its element expression(s)."""
        local = set(defined)
        for gen in generators:
            _collect_loads(gen.iter, local)
            local |= _target_names(gen.target)
            for cond in gen.ifs:
                _collect_loads(cond, local)
        for elt in elts:
            _collect_loads(elt, local)

    _walk_block(tree.body, set())
    return sorted(loads - _BUILTINS)


# ---------------------------------------------------------------------------
# Source extraction helpers
# ---------------------------------------------------------------------------


def _stmts_source(
    stmts: list,
    source_lines: List[str],
    positions: Dict,
) -> str:
    """Extract and dedent the source for a list of statements."""
    if not stmts:
        return ""
    start = positions[stmts[0]].start.line
    end = positions[stmts[-1]].end.line
    raw = "".join(source_lines[start - 1 : end])
    return textwrap.dedent(raw)


def _head_effective_lines(
    body_stmts: list,
    split_idx: int,
    positions: Dict,
    has_docstring: bool,
) -> int:
    """Estimate head line count after splitting at split_idx (includes return call)."""
    first_non_doc_idx = 1 if has_docstring else 0
    if first_non_doc_idx >= split_idx:
        # Head only has the docstring (or nothing) — effective content is 1 return line
        return 1
    first_non_doc_stmt = body_stmts[first_non_doc_idx]
    last_stmt = body_stmts[split_idx - 1]
    # +1 for inclusive range, +1 for the inserted return call
    return positions[last_stmt].end.line - positions[first_non_doc_stmt].start.line + 2


def _find_valid_splits(
    body_stmts: list,
    positions: Dict,
    max_lines: int,
) -> List[int]:
    """Return valid split indices (latest first, max _MAX_SPLIT_CANDIDATES)."""
    if not body_stmts:
        return []
    has_doc = _is_docstring_stmt(body_stmts[0])
    candidates: List[int] = []

    # Restrict split point to before the first nested function def in the body.
    # Splitting across a closure boundary would produce NameErrors at runtime:
    # nested functions defined in HEAD can reference names only available in TAIL.
    upper = len(body_stmts) - 1
    for k, stmt in enumerate(body_stmts):
        if isinstance(stmt, cst.FunctionDef):
            upper = k
            break

    for i in range(upper, 0, -1):
        head_lines = _head_effective_lines(body_stmts, i, positions, has_doc)
        if head_lines > max_lines:
            continue
        candidates.append(i)
        if len(candidates) >= _MAX_SPLIT_CANDIDATES:
            break

    return candidates


def _update_best_split(best_idx, best_params, split_idx, params):
    if best_idx is None or len(params) < len(best_params):
        return split_idx, params
    return best_idx, best_params


def _choose_best_split(
    body_stmts: list,
    valid_splits: List[int],
    source_lines: List[str],
    positions: Dict,
    orig_params: List[str],
    module_globals: set = frozenset(),
) -> Optional[Tuple[int, List[str], bool]]:
    """Choose split with fewest free variables; ties broken by latest split.

    Returns a 3-tuple ``(split_idx, params, is_instance_method)``.

    * ``is_instance_method=False`` — the helper can be a ``@staticmethod``
      (tail does not reference ``self``).
    * ``is_instance_method=True`` — the tail references ``self``, so the
      helper must be a regular instance method; ``self`` is excluded from
      ``params`` because it is the implicit first argument.

    Static splits are preferred over instance-method splits.  Returns
    ``None`` only when ``valid_splits`` is empty.

    ``module_globals`` names are excluded from the helper's param list.
    """
    best_static_idx: Optional[int] = None
    best_static_params: List[str] = []
    best_instance_idx: Optional[int] = None
    best_instance_params: List[str] = []

    is_method = bool(orig_params) and orig_params[0] == "self"

    for split_idx in valid_splits:
        tail_stmts = body_stmts[split_idx:]
        tail_src = _stmts_source(tail_stmts, source_lines, positions)
        free = [v for v in _find_free_vars(tail_src) if v not in module_globals]
        if is_method and "self" in free:
            # Tail needs instance state — extract as a regular instance method.
            params_no_self = [v for v in free if v != "self"]
            best_instance_idx, best_instance_params = _update_best_split(
                best_instance_idx, best_instance_params, split_idx, params_no_self
            )
        else:
            best_static_idx, best_static_params = _update_best_split(
                best_static_idx, best_static_params, split_idx, free
            )

    # Prefer static (no self dependency) over instance method.
    if best_static_idx is not None:
        return (best_static_idx, best_static_params, False)
    if best_instance_idx is not None:
        return (best_instance_idx, best_instance_params, True)
    return None


# ---------------------------------------------------------------------------
# Code generation helpers
# ---------------------------------------------------------------------------


def _generate_helper_source(
    name: str,
    params: List[str],
    tail_source: str,
    func_indent: str,
    is_static: bool,
    add_docstring: bool,
    is_instance_method: bool = False,
) -> str:
    """Build the source text for the extracted helper function."""
    body_indent = func_indent + "    "
    parts = []
    if is_static:
        parts.append(func_indent + "@staticmethod\n")
    all_params = (["self"] + params) if is_instance_method else params
    parts.append(func_indent + f"def _{name}({', '.join(all_params)}):\n")
    if add_docstring:
        parts.append(body_indent + '"""..."""\n')
    body = textwrap.indent(tail_source.rstrip("\n"), body_indent)
    parts.append(body + "\n")
    return "".join(parts)


def _generate_call(
    name: str,
    params: List[str],
    class_name: Optional[str],
    body_indent: str,
    is_instance_method: bool = False,
) -> str:
    """Build the return-call line inserted into the head function."""
    args = ", ".join(params)
    if is_instance_method:
        return body_indent + f"return self._{name}({args})"
    if class_name:
        return body_indent + f"return {class_name}._{name}({args})"
    return body_indent + f"return _{name}({args})"


# ---------------------------------------------------------------------------
# LLM naming
# ---------------------------------------------------------------------------


def _llm_name_helpers(
    client,
    model: str,
    provider: str,
    tasks: List[_SplitTask],
) -> List[str]:
    """Single LLM call to name all helper functions. Falls back on error."""
    task_texts = []
    for i, task in enumerate(tasks):
        fi = task.func_info
        class_info = (
            f" in class '{fi.class_name}'" if fi.class_name else " at module level"
        )
        task_texts.append(
            f"Function {i} (id: '{i}'):\n"
            f"  Original name: '{fi.node.name.value}'{class_info}\n"
            f"  Helper params: {task.params}\n"
            f"  Tail code to extract:\n```python\n{task.tail_source.strip()}\n```"
        )

    prompt = (
        "Name these extracted Python helper functions. "
        "Provide a concise snake_case name (no leading underscore) for each.\n\n"
        + "\n\n".join(task_texts)
    )

    result = _llm_client.call_with_tool(
        client,
        provider,
        model,
        256,
        _NAME_TOOL,
        "name_helper_functions",
        [{"role": "user", "content": prompt}],
        caller="FunctionSplitter",
    )

    fallback = [f"{t.func_info.node.name.value}_helper" for t in tasks]

    if result is None or "names" not in result:
        return fallback

    name_map: Dict[str, str] = {}
    for item in result["names"]:
        try:
            raw = item["name"].lstrip("_")
            name_map[str(item["id"])] = raw if raw else "helper"
        except (KeyError, TypeError, AttributeError):
            pass

    return [
        name_map.get(str(i), f"{tasks[i].func_info.node.name.value}_helper")
        for i in range(len(tasks))
    ]


# ---------------------------------------------------------------------------
# Function range helpers
# ---------------------------------------------------------------------------


def _func_in_changed_range(
    func_info: _FuncInfo, changed_ranges: List[Tuple[int, int]]
) -> bool:
    """Return True if the function overlaps any changed range."""
    for start, end in changed_ranges:
        if func_info.start_line <= end and func_info.end_line >= start:
            return True
    return False


def _extract_func_source(func_info: _FuncInfo, source_lines: List[str]) -> str:
    """Return dedented source of the function (including decorators)."""
    raw = "".join(source_lines[func_info.start_line - 1 : func_info.end_line])
    return textwrap.dedent(raw)


# ---------------------------------------------------------------------------
# Pyflakes safety check
# ---------------------------------------------------------------------------


def _has_new_undefined_names(before_src: str, after_src: str) -> bool:
    """Return True if after_src introduces UndefinedName warnings not in before_src.

    Compares pyflakes output before and after a transformation.  Only newly
    introduced ``UndefinedName`` warnings (F821) count; pre-existing issues in
    the original source are ignored to avoid false positives.
    """
    import pyflakes.api
    import pyflakes.messages

    class _Collector:
        def __init__(self) -> None:
            self.names: set = set()

        def unexpectedError(self, filename, msg) -> None:  # pragma: no cover
            pass

        def syntaxError(
            self, filename, msg, lineno, offset, text
        ) -> None:  # pragma: no cover
            pass

        def flake(self, msg) -> None:
            if isinstance(msg, pyflakes.messages.UndefinedName):
                self.names.add(msg.message_args[0])

    try:
        before = _Collector()
        pyflakes.api.check(before_src, "<before>", reporter=before)
        after = _Collector()
        pyflakes.api.check(after_src, "<after>", reporter=after)
        return bool(after.names - before.names)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CST visitor to collect functions
# ---------------------------------------------------------------------------


class _FunctionCollector(cst.CSTVisitor):
    """Visit a module and collect _FuncInfo for each splittable function."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self) -> None:
        self.functions: List[_FuncInfo] = []
        self._class_stack: List[str] = []
        self._scope_kind_stack: List[str] = [
            "module"
        ]  # "module" | "class" | "function"

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self._class_stack.append(node.name.value)
        self._scope_kind_stack.append("class")
        return None

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self._class_stack.pop()
        self._scope_kind_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        parent_kind = self._scope_kind_stack[-1]
        if parent_kind in ("module", "class"):
            # Only collect top-level functions and class methods
            if (
                node.asynchronous is None
                and not _has_yield(node)
                and not _has_nested_funcdef(node)
            ):
                pos = self.get_metadata(PositionProvider, node)
                class_name = self._class_stack[-1] if self._class_stack else None
                indent = " " * pos.start.column
                params = [
                    p.name.value
                    for p in node.params.params
                    if isinstance(p.name, cst.Name)
                ]
                self.functions.append(
                    _FuncInfo(
                        node=node,
                        start_line=pos.start.line,
                        end_line=pos.end.line,
                        class_name=class_name,
                        indent=indent,
                        original_params=params,
                    )
                )
        self._scope_kind_stack.append("function")
        return None

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._scope_kind_stack.pop()


# ---------------------------------------------------------------------------
# Main refactor class
# ---------------------------------------------------------------------------


class FunctionSplitter(Refactor):
    """Split functions that exceed the line-count limit."""

    def __init__(
        self,
        changed_ranges: List[Tuple[int, int]],
        source: str = "",
        verbose: bool = True,
        max_lines: int = 75,
        model: str = _MODEL,
        provider: str = "anthropic",
        helper_docstrings: bool = False,
    ) -> None:
        super().__init__(changed_ranges, source=source, verbose=verbose)
        self._max_lines = max_lines
        self._model = model
        self._provider = provider
        self._helper_docstrings = helper_docstrings
        self._new_source: Optional[str] = None
        if source:
            self._analyze(source)

    def get_rewritten_source(self) -> Optional[str]:
        return self._new_source

    def _analyze(self, source: str) -> None:
        """Iteratively split oversized functions until stable or limit reached."""
        return self._split_large_functions(source)

    def _split_large_functions(self, source):
        current = source
        return self._process_split_iteration(current, source)

    def _process_split_iteration(self, current, source):
        for _iteration in range(_MAX_SPLIT_ITERATIONS):
            try:
                tree = cst.parse_module(current)
            except cst.ParserSyntaxError:
                return

            source_lines = current.splitlines(keepends=True)
            wrapper = MetadataWrapper(tree)
            positions = wrapper.resolve(PositionProvider)

            # 1. Collect all splittable functions
            collector = _FunctionCollector()
            wrapper.visit(collector)

            # 2. For each function in changed ranges: check limits, plan splits
            module_globals = _module_global_names(current)
            tasks: List[_SplitTask] = []
            for func_info in collector.functions:
                if not _func_in_changed_range(func_info, self.changed_ranges):
                    continue
                func_source = _extract_func_source(func_info, source_lines)
                body_lines = _count_body_lines(func_source)
                if body_lines <= self._max_lines:
                    continue
                body_stmts = list(func_info.node.body.body)
                if len(body_stmts) < 2:
                    continue
                valid_splits = _find_valid_splits(
                    body_stmts,
                    positions,
                    self._max_lines,
                )
                if not valid_splits:
                    continue
                best = _choose_best_split(
                    body_stmts,
                    valid_splits,
                    source_lines,
                    positions,
                    func_info.original_params,
                    module_globals,
                )
                split_idx, params, is_instance_method = best
                tail_stmts = body_stmts[split_idx:]
                tail_raw = _stmts_source(tail_stmts, source_lines, positions)
                tasks.append(
                    _SplitTask(
                        func_info,
                        split_idx,
                        params,
                        tail_source=tail_raw,
                        is_instance_method=is_instance_method,
                    )
                )

            if not tasks:
                break  # stable

            # 3. LLM call for names
            try:
                api_key = _llm_client.get_api_key(self._provider, "FunctionSplitter")
                client = _llm_client.make_client(self._provider, api_key)
                names = _run_with_timeout(
                    _llm_name_helpers,
                    _API_HARD_TIMEOUT,
                    client,
                    self._model,
                    self._provider,
                    tasks,
                )
            except Exception:
                names = [f"{t.func_info.node.name.value}_helper" for t in tasks]

            for task, name in zip(tasks, names):
                task.helper_name = name

            # Drop tasks whose helper name collides with an existing function
            # definition.  In Python, two defs with the same name keep only the
            # last, so a collision would cause a TypeError at every call site
            # that was written for the earlier (now-overridden) signature.
            existing_func_names = {
                node.name
                for node in ast.walk(ast.parse(current))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            tasks = [t for t in tasks if f"_{t.helper_name}" not in existing_func_names]
            if not tasks:
                break

            # 4. Build text edits (process bottom-up by start_line)
            tasks.sort(key=lambda t: t.func_info.start_line, reverse=True)
            edits: List[Tuple[int, int, str]] = []
            for task in tasks:
                fi = task.func_info
                body_stmts = list(fi.node.body.body)
                tail_stmts = body_stmts[task.split_idx :]
                tail_raw = _stmts_source(tail_stmts, source_lines, positions)
                body_indent = fi.indent + "    "
                call_line = _generate_call(
                    task.helper_name,
                    task.params,
                    fi.class_name,
                    body_indent,
                    task.is_instance_method,
                )
                tail_start = positions[tail_stmts[0]].start.line
                helper_src = _generate_helper_source(
                    task.helper_name,
                    task.params,
                    tail_raw,
                    fi.indent,
                    fi.class_name is not None and not task.is_instance_method,
                    self._helper_docstrings,
                    task.is_instance_method,
                )
                head_lines = source_lines[fi.start_line - 1 : tail_start - 1]
                replacement = (
                    "".join(head_lines).rstrip("\n")
                    + "\n"
                    + call_line
                    + "\n\n\n"
                    + helper_src
                )
                edits.append((fi.start_line, fi.end_line, replacement))

            # Apply edits bottom-up
            result_lines = list(source_lines)
            for start, end, replacement in edits:
                result_lines[start - 1 : end] = [replacement]
            new_source = "".join(result_lines)

            # 5. Verify syntax
            try:
                compile(new_source, "<string>", "exec")
            except SyntaxError:
                break  # bail, don't apply bad output

            # 6. Verify no new undefined names (catches NameError-producing splits)
            if _has_new_undefined_names(current, new_source):
                break

            for task in tasks:
                fi = task.func_info
                self.changes_made.append(
                    f"split {fi.node.name.value!r}: extracted _{task.helper_name}"
                )
            current = new_source

        if current != source:
            self._new_source = current
