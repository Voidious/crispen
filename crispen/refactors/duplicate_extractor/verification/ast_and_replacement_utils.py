from __future__ import annotations
from typing import List, Tuple
import ast
import re
import textwrap
from ..common import _FunctionInfo, _SeqInfo


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


def _missing_free_vars(
    block_src: str, call_srcs: List[str], helper_src: str, source: str
) -> set:
    """Return locally-scoped free variable names from block_src absent from the
    replacement.

    Free variables are names that are *read* (appear in a ``Load`` context) but
    not locally *assigned* (``Store``/``Del``) within the original block.  To
    avoid false positives from module-level names (imported symbols, globally-
    defined functions) that the extracted helper can reference directly, the
    check is restricted to names that appear as assignment targets or function
    parameters somewhere in *source* — these are variables that live in a local
    scope and cannot be reached by the helper without being threaded through as
    arguments.

    After this filtering, every remaining name must appear as a bare ``Name``
    node somewhere in the call-site replacements or the helper body.  A name
    that vanishes from both indicates the LLM silently changed the data flow —
    for example by turning a local variable reference into an attribute access
    on one of the parameters (``new_source`` → ``transformer.new_source``).

    Returns the set of names that are absent from both.  An empty set means the
    check passes.  Returns an empty set on any ``SyntaxError`` so a parse
    failure does not block the extraction — the later ``compile()`` guard will
    catch real syntax problems.
    """
    try:
        block_tree = ast.parse(textwrap.dedent(block_src))
    except SyntaxError:
        return set()

    reads: set = set()
    stores: set = set()
    for node in ast.walk(block_tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                reads.add(node.id)
            else:
                stores.add(node.id)

    free_vars = reads - stores
    if not free_vars:
        return set()

    # Restrict to names that are locally assigned or are function/lambda
    # parameters somewhere in the full source.  Module-level names that are
    # only ever read (e.g. imported functions, global constants) are in scope
    # from the helper definition too and do not need to be passed as args.
    try:
        source_tree = ast.parse(source)
    except SyntaxError:
        return set()
    source_locals: set = set()
    for node in ast.walk(source_tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            source_locals.add(node.id)
        elif isinstance(node, ast.arg):
            source_locals.add(node.arg)

    free_vars = free_vars & source_locals
    if not free_vars:
        return set()

    replacement_names: set = set()
    for src in list(call_srcs) + [helper_src]:
        try:
            repl_tree = ast.parse(textwrap.dedent(src))
        except SyntaxError:
            return set()
        for node in ast.walk(repl_tree):
            if isinstance(node, ast.Name):
                replacement_names.add(node.id)

    return free_vars - replacement_names


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


def _replacement_steals_post_block_line(
    group: List[_SeqInfo], call_replacements: List[str], source_lines: List[str]
) -> bool:
    """Return True if any replacement's last line duplicates the line after its block.

    The LLM occasionally appends the first statement *after* the replaced block
    to the end of the replacement text.  When applied, that statement then appears
    twice in the assembled output: once inside the replacement and once as the
    original untouched line.
    """
    for seq, replacement in zip(group, call_replacements):
        next_idx = seq.end_line  # 0-based index of the first line after the block
        # Scan forward past blank lines to find the first real post-block line.
        while next_idx < len(source_lines) and not source_lines[next_idx].strip():
            next_idx += 1
        if next_idx >= len(source_lines):
            continue
        post_block = source_lines[next_idx].strip()
        repl_lines = [ln.strip() for ln in replacement.splitlines() if ln.strip()]
        if repl_lines and repl_lines[-1] == post_block:
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


def _find_escaping_vars(group: List[_SeqInfo], source_lines: List[str]) -> set:
    """Return names assigned in any group sequence that are referenced after it.

    A variable "escapes" when the block assigns it and subsequent code in the
    same scope (at the same or deeper indentation level) references it.
    The helper must return these variables so callers that need them can
    capture the return value.
    """
    escaping: set = set()
    for seq in group:
        block_src = "".join(source_lines[seq.start_line - 1 : seq.end_line])
        assigned = _names_assigned_in(block_src)
        if not assigned:
            continue

        # Infer the block's indentation level from its first non-empty line.
        first_line = next(
            (
                ln
                for ln in source_lines[seq.start_line - 1 : seq.end_line]
                if ln.strip()
            ),
            "",
        )
        block_indent = len(first_line) - len(first_line.lstrip())

        # Collect lines that follow the block within the same scope.
        # For indented blocks: stop when indentation falls below block_indent.
        # For module-level (indent 0): stop at the next def/class statement.
        after_lines: List[str] = []
        for line in source_lines[seq.end_line :]:
            if not line.strip():
                after_lines.append(line)
                continue
            line_indent = len(line) - len(line.lstrip())
            if block_indent == 0:
                if re.match(r"def |class ", line):
                    break
            elif line_indent < block_indent:
                break
            after_lines.append(line)

        if not after_lines:
            continue

        after_src = "".join(after_lines)
        try:
            after_tree = ast.parse(textwrap.dedent(after_src))
        except SyntaxError:
            continue

        used_after = {n.id for n in ast.walk(after_tree) if isinstance(n, ast.Name)}
        escaping |= assigned & used_after

    return escaping


def _would_create_proxy_wrappers(
    group: List[_SeqInfo], all_functions: List[_FunctionInfo]
) -> bool:
    """Return True if extracting this group would leave *some but not all* members
    as trivial proxy wrappers.

    A function becomes a trivial proxy wrapper when its entire body is the
    extracted block — after extraction it would contain only a single call to
    the new helper, with no meaningful logic of its own.

    When *every* member of the group would become a proxy, extraction is still
    worthwhile: all functions delegate to the same helper, which eliminates the
    duplication.  The problematic case is a mixed group where some members lose
    all their logic while others keep meaningful bodies.
    """
    proxy_count = 0
    non_module_count = 0
    for seq in group:
        if seq.scope == "<module>":
            continue
        non_module_count += 1
        func_outer_scope = (
            seq.class_scope if seq.class_scope is not None else "<module>"
        )
        for func in all_functions:
            if func.name == seq.scope and func.scope == func_outer_scope:
                if len(seq.stmts) == func.body_stmt_count:
                    proxy_count += 1
                break
    return 0 < proxy_count < non_module_count
