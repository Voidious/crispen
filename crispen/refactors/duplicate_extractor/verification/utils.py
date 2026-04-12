from __future__ import annotations
from typing import List, Tuple
import ast
import re
import textwrap
from ..collectors import _SeqInfo


def _normalize_replacement_indentation(seq: _SeqInfo, replacement: str) -> str:
    """Re-indent *replacement* to match the original block's leading whitespace.

    The LLM sometimes returns replacements at column 0.  This function
    re-indents them to match the indentation of the corresponding original
    block, so the assembled edit remains valid Python.
    """
    orig_lines = [ln for ln in seq.source.splitlines() if ln.strip()]
    if not orig_lines:
        return replacement
    first = orig_lines[0]
    expected_indent = first[: len(first) - len(first.lstrip())]
    dedented = textwrap.dedent(replacement)
    if not expected_indent:
        return dedented
    return textwrap.indent(dedented, expected_indent)


def _collect_ast_store_names(node: ast.AST, names: List[str]) -> None:
    """Recursively collect Name ids from an assignment target (Store context)."""
    if isinstance(node, ast.Name):
        names.append(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            _collect_ast_store_names(elt, names)


def _replace_unused_in_target(
    target: ast.AST, following_src: str
) -> Tuple[ast.AST, bool, bool]:
    """Replace unused Name nodes in *target* with ``_``.

    Returns ``(new_target, all_replaced, any_replaced)`` where:
    - *all_replaced*: every name in the target was replaced (all unused).
    - *any_replaced*: at least one name was replaced.

    Non-Name, non-Tuple/List targets (Attribute, Subscript, …) are treated as
    *used* so we never accidentally strip an assignment we cannot analyse.
    """
    if isinstance(target, ast.Name):
        if re.search(r"\b" + re.escape(target.id) + r"\b", following_src):
            return target, False, False  # used → keep
        return ast.Name(id="_", ctx=ast.Store()), True, True  # unused → _
    if isinstance(target, (ast.Tuple, ast.List)):
        new_elts: List[ast.AST] = []
        all_replaced = True
        any_replaced = False
        for elt in target.elts:
            new_elt, elt_all, elt_any = _replace_unused_in_target(elt, following_src)
            new_elts.append(new_elt)
            if not elt_all:
                all_replaced = False
            if elt_any:
                any_replaced = True
        new_target = type(target)(elts=new_elts, ctx=ast.Store())
        return new_target, all_replaced, any_replaced
    # Attribute, Subscript, Starred, etc. — treat as used.
    return target, False, False


def _scope_end_line(source_lines: List[str], scope: str, after_line: int) -> int:
    """Return the exclusive slice index into *source_lines* for the end of *scope*.

    ``after_line`` is the 1-based line number of the last line of the replaced
    block.  The returned index is suitable for ``source_lines[after_line:idx]``
    to get only the lines inside the enclosing scope that follow the block.

    For ``"<module>"`` scope the whole rest of the file is in scope, so
    ``len(source_lines)`` is returned.  For named function/class scopes the
    innermost definition whose name matches *scope* and that contains
    *after_line* is located via the AST; its end line is returned as the
    exclusive slice bound (1-based end_lineno used directly as a 0-based
    exclusive index is correct because line N is at index N-1, so slicing up
    to index N includes line N).  Falls back to ``len(source_lines)`` on any
    parse error or if no matching scope is found.
    """
    if scope == "<module>":
        return len(source_lines)

    source = "".join(source_lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return len(source_lines)

    # ast.walk is BFS, so outer scopes are visited before inner ones.  Always
    # overwriting best_end means the last match wins — which is the innermost
    # (smallest) scope that still contains after_line.
    best_end: int = len(source_lines)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name != scope:
            continue
        if not (node.lineno <= after_line <= node.end_lineno):
            continue
        best_end = node.end_lineno

    return best_end


def _strip_unused_call_assignments(replacement: str, following_lines: List[str]) -> str:
    """Clean up unused assignment targets in a call-site replacement.

    For each ``Assign`` node whose right-hand side is a ``Call``:

    * **Single target** — unused ``Name`` elements in the target are replaced
      with ``_``.  If every element is unused the whole assignment is dropped
      and only the call expression is emitted.  Example::

          result = _helper(x)          →  _helper(x)
          a, b   = _helper(x)  (b used)  →  a, _ = _helper(x)

    * **Chained assignment** (``a = b = call()``) — stripped to just the call
      only when every name across every target is unused; otherwise left alone.

    Augmented (``+=``) and annotated assignments are never touched.  Assignment
    targets that are not plain names or tuples/lists (e.g. ``self.x``) are
    treated as *used* so we never accidentally remove live assignments.

    This prevents flake8 F841 "local variable assigned but never used" errors
    introduced by the extraction.
    """
    following_src = "".join(following_lines)
    try:
        dedented = textwrap.dedent(replacement)
        tree = ast.parse(dedented)
    except SyntaxError:
        return replacement

    # Build a list of (start_ln, end_ln, new_src) edits.  new_src is the
    # replacement text for that statement (without leading indentation).
    edits: List[Tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value_node = node.value
        if isinstance(value_node, ast.Await) and isinstance(value_node.value, ast.Call):
            pass  # treat `result = await helper(...)` like `result = helper(...)`
        elif not isinstance(value_node, ast.Call):
            continue

        call_src = ast.unparse(value_node)

        if len(node.targets) == 1:
            new_target, all_replaced, any_replaced = _replace_unused_in_target(
                node.targets[0], following_src
            )
            if all_replaced:
                edits.append((node.lineno, node.end_lineno, call_src))
            elif any_replaced:
                edits.append(
                    (
                        node.lineno,
                        node.end_lineno,
                        ast.unparse(new_target) + " = " + call_src,
                    )
                )
        else:
            # Chained assignment: strip only when every name is unused.
            all_names: List[str] = []
            for t in node.targets:
                _collect_ast_store_names(t, all_names)
            if not all_names:
                continue
            if not any(
                re.search(r"\b" + re.escape(n) + r"\b", following_src)
                for n in all_names
            ):
                edits.append((node.lineno, node.end_lineno, call_src))

    if not edits:
        return replacement

    # Determine the leading indentation from the first non-empty line.
    first_content = next((ln for ln in replacement.splitlines() if ln.strip()), "")
    indent = first_content[: len(first_content) - len(first_content.lstrip())]

    # Apply edits in reverse line order so earlier indices stay valid.
    dedented_lines = dedented.splitlines(keepends=True)
    for start_ln, end_ln, new_src in sorted(edits, key=lambda x: x[0], reverse=True):
        dedented_lines[start_ln - 1 : end_ln] = [new_src + "\n"]

    return textwrap.indent("".join(dedented_lines), indent)


_MUTABLE_CONSTRUCTORS = frozenset({"set", "list", "dict", "frozenset", "bytearray"})


def _has_mutable_literal_is_check(source: str) -> bool:
    """Return True if *source* contains identity checks against mutable literals.

    Patterns like ``x is set()``, ``x is []``, or ``x is {}`` are always
    False in Python because each literal creates a new object at runtime.
    Such patterns are a common LLM mistake when using a ``set()`` sentinel.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for op, comp in zip(node.ops, node.comparators):
            if not isinstance(op, (ast.Is, ast.IsNot)):
                continue
            if isinstance(comp, (ast.List, ast.Set, ast.Dict, ast.Tuple)):
                return True
            if (
                isinstance(comp, ast.Call)
                and isinstance(comp.func, ast.Name)
                and comp.func.id in _MUTABLE_CONSTRUCTORS
            ):
                return True
    return False


def _collect_attribute_names(source: str) -> set:
    """Return all attribute names (dot-access names) anywhere in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def _collect_called_attr_names(source: str) -> set:
    """Return attribute names used as method calls in *source*.

    Unlike :func:`_collect_attribute_names`, this only returns names that
    appear as the attribute of a call expression (i.e. ``obj.method(...)``).
    Plain attribute reads and type annotations like ``ast.AST`` are ignored,
    so the new-method-call check does not produce false positives for
    standard-library type references.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def _has_funcdef(func_name: str, source: str) -> bool:
    """Return True if func_name is defined anywhere in source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        ):
            return True
    return False


def _has_call_to(func_name: str, source: str) -> bool:
    """Return True if func_name is called anywhere in source.

    Checks both direct calls (``func_name(...)``) and attribute calls
    (``obj.func_name(...)``), covering both module-level helpers and
    staticmethod calls.  Returns False if source cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == func_name:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
            return True
    return False


def _is_pure_literal(node: ast.expr) -> bool:
    """Return True if *node* is a side-effect-free literal expression.

    Covers ``ast.Constant`` (numbers, strings, bytes, True/False/None) and
    recursively-pure container literals (list, tuple, set, dict).  Anything
    involving a function call or attribute access returns False.
    """
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_pure_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (k is None or _is_pure_literal(k)) and _is_pure_literal(v)
            for k, v in zip(node.keys, node.values)
        )
    return False


def _names_in_edit_texts(extraction_groups) -> set:
    """Return all bare ``Name`` ids found in every edit text of *extraction_groups*.

    ``extraction_groups`` is the list of ``(func_name, group_edits, msg)``
    tuples accepted at the end of ``DuplicateExtractor._transform``.  Each
    ``group_edits`` entry is a ``(start, end, text)`` triple; *text* may be
    the helper function source or a call-site replacement.  Collecting names
    from all of them gives the set of variables that the extraction actually
    touched.
    """
    names: set = set()
    for _, g_edits, _ in extraction_groups:
        for _start, _end, text in g_edits:
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    names.add(node.id)
    return names


def _seq_ends_with_return(seq: _SeqInfo) -> bool:
    """Return True if the last top-level statement is a non-None return.

    Detects the case where the LLM includes a ``return`` statement inside the
    duplicate block but the generated replacement omits it, producing a
    function that silently returns ``None`` instead of the original value.

    Bare ``return`` and ``return None`` are excluded: both are semantically
    equivalent to falling off the end of a function, so dropping them in a
    replacement causes no behavioral change.
    """
    try:
        tree = ast.parse(textwrap.dedent(seq.source))
    except SyntaxError:
        return False
    if not tree.body:
        return False
    last = tree.body[-1]
    if not isinstance(last, ast.Return):
        return False
    # Bare `return` and `return None` are equivalent to implicit None return.
    if last.value is None:
        return False
    if isinstance(last.value, ast.Constant) and last.value.value is None:
        return False
    return True


def _replacement_contains_return(replacement: str) -> bool:
    """Return True if *replacement* contains any return statement.

    Wraps the replacement in a dummy function before parsing so that
    ``return`` statements — which are legal inside a function body — do not
    cause false SyntaxError rejections.
    """
    try:
        wrapped = "def _check():\n" + textwrap.indent(
            textwrap.dedent(replacement), "    "
        )
        tree = ast.parse(wrapped)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            return True
    return False


def _names_assigned_in(block_source: str) -> set:
    """Return names assigned at the top level of block_source.

    Covers bare ``x = ...`` (ast.Assign) and augmented ``x += ...``
    (ast.AugAssign) statements only; other assignment forms are ignored.
    """
    try:
        tree = ast.parse(textwrap.dedent(block_source))
    except SyntaxError:
        return set()
    names: set = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for n in ast.walk(target):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(node, ast.AugAssign):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    names.add(n.id)
    return names


def _extract_defined_names(source: str) -> set:
    """Return all function and class names defined anywhere in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
