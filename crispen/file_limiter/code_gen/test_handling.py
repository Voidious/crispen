from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import ast
import re
from ..entity_parser import Entity, EntityKind
from .cross_file_deps import _relative_import_prefix, _target_module_name


def _test_names_in_decorators(source: str, names: Set[str]) -> Set[str]:
    """Return the subset of *names* that appear as Name loads inside a decorator.

    Decorators are evaluated before function bodies run, so a symbol that
    only reaches a file via an inline import (injected into the function body)
    will not be in scope when the decorator is evaluated.  This helper detects
    that situation so callers can abort the split rather than generate broken
    code.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                for child in ast.walk(dec):
                    if (
                        isinstance(child, ast.Name)
                        and isinstance(child.ctx, ast.Load)
                        and child.id in names
                    ):
                        found.add(child.id)
    return found


def _class_has_test_methods(entity_src: str) -> bool:
    """Return True if *entity_src* defines a class with any ``test_`` methods.

    Used to suppress re-exports of test classes: pytest discovers test classes
    by scanning the filesystem, so re-exporting them from the original file
    causes every test inside to run twice.
    """
    try:
        tree = ast.parse(entity_src)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("test_"):
                        return True
    return False


def _is_test_name(name: str) -> bool:
    """Return True if *name* matches pytest's test-discovery patterns.

    Pytest collects classes named ``Test*`` and functions named ``test_*``.
    Importing such names at module level in a test file causes every test
    inside to be discovered — and run — a second time.
    """
    return name.startswith("Test") or name.startswith("test_")


def _is_pytest_fixture(entity_src: str) -> bool:
    """Return True if *entity_src* defines a function with a @pytest.fixture decorator.

    Handles all common forms: ``@fixture``, ``@fixture()``, ``@pytest.fixture``,
    and ``@pytest.fixture(scope=...)``.
    """
    try:
        tree = ast.parse(entity_src)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                # Unwrap call forms like @pytest.fixture(...) to get the base reference.
                ref = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(ref, ast.Name) and ref.id == "fixture":
                    return True
                if isinstance(ref, ast.Attribute) and ref.attr == "fixture":
                    return True
    return False


def _file_has_only_fixtures(source: str) -> bool:
    """Return True if *source* has at least one @pytest.fixture and nothing else.

    "Nothing else" means no test functions (``def test_*``), no test classes
    (``class Test*``), no other function/class definitions, and no non-import
    module-level statements other than a module docstring.  Import statements
    and a leading docstring are allowed because they are needed to support the
    fixture definitions.

    Returns False on syntax errors (be conservative).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    lines = source.splitlines(keepends=True)
    has_fixture = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # module docstring or standalone string literal
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            first_line = (
                node.decorator_list[0].lineno if node.decorator_list else node.lineno
            )
            fn_src = "".join(lines[first_line - 1 : node.end_lineno]).rstrip()
            if _is_pytest_fixture(fn_src):
                has_fixture = True
                continue
            return False  # non-fixture function (including test_ functions)
        return False  # class, assignment, or other statement
    return has_fixture


def _split_cross_imports_by_test(
    imports: List[str],
) -> Tuple[List[str], List[str]]:
    """Split cross-file import statements into (non_test, test_named) groups.

    Import statements that name pytest-discoverable symbols (``Test*`` or
    ``test_*``) are returned as inline imports so callers can inject them
    into function/class bodies rather than emitting them at module level.
    Mixed imports (some test, some non-test names) are split into two
    separate statements.
    """
    non_test: List[str] = []
    test_named: List[str] = []
    for imp in imports:
        m = re.match(r"^(from\s+\S+\s+import\s+)(.*)", imp)
        if not m:
            non_test.append(imp)
            continue
        prefix = m.group(1)
        names = [n.strip() for n in m.group(2).split(",")]
        t_names = sorted(n for n in names if _is_test_name(n))
        nt_names = sorted(n for n in names if not _is_test_name(n))
        if t_names:
            test_named.append(f"{prefix}{', '.join(t_names)}")
        if nt_names:
            non_test.append(f"{prefix}{', '.join(nt_names)}")
    return non_test, test_named


def _inject_inline_imports(entity_src: str, imports: List[str]) -> str:
    """Inject *imports* at the top of a function or class body in *entity_src*.

    The imports are inserted after any leading docstring.  Returns
    *entity_src* unchanged when it cannot be parsed or the top-level node
    is not a function or class (TOP_LEVEL entities have no body scope).
    """
    if not imports:
        return entity_src
    try:
        tree = ast.parse(entity_src)
    except SyntaxError:
        return entity_src
    if not tree.body:
        return entity_src
    top = tree.body[0]
    if not isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return entity_src
    first_stmt = top.body[0]
    insert_line = first_stmt.lineno
    if (
        isinstance(first_stmt, ast.Expr)
        and isinstance(first_stmt.value, ast.Constant)
        and isinstance(first_stmt.value.value, str)
        and len(top.body) > 1
    ):
        insert_line = top.body[1].lineno
    lines = entity_src.splitlines(keepends=True)
    body_line = lines[insert_line - 1]
    indent = body_line[: len(body_line) - len(body_line.lstrip())]
    import_lines = [f"{indent}{imp}\n" for imp in imports]
    return "".join(lines[: insert_line - 1] + import_lines + lines[insert_line - 1 :])


def _find_main_block_entity(
    entities: List[Entity],
    entity_source_map: Dict[str, str],
) -> Optional[str]:
    """Return the entity name of the ``if __name__ == '__main__':`` block.

    Returns ``None`` when no such block is present.
    """
    for entity in entities:
        if entity.kind != EntityKind.TOP_LEVEL:
            continue
        src = entity_source_map.get(entity.name, "")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq)
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "__main__"
            ):
                return entity.name
    return None


def _find_main_direct_callees(
    main_src: str, function_entity_names: Set[str]
) -> Set[str]:
    """Return function entity names called directly in the ``__main__`` block.

    Only names that appear in *function_entity_names* (i.e. are defined as
    top-level FUNCTION entities in the same file) are returned, so the
    caller can keep those functions sticky to the original file alongside
    the ``__main__`` block.
    """
    try:
        tree = ast.parse(main_src)
    except SyntaxError:
        return set()
    callees: Set[str] = set()
    for node in tree.body:
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"
        ):
            continue
        for subnode in ast.walk(node):
            if (
                isinstance(subnode, ast.Call)
                and isinstance(subnode.func, ast.Name)
                and subnode.func.id in function_entity_names
            ):
                callees.add(subnode.func.id)
    return callees


def _inject_inline_test_imports_original(
    source: str,
    migrated_test_symbols: Dict[str, str],
    abs_pkg: Optional[str],
    original_basename: str,
) -> str:
    """Inject inline imports for migrated test-named symbols into function/class bodies.

    After a split, test-named symbols (``Test*`` / ``test_*``) that were
    migrated to new files are not re-exported at module level (to avoid
    pytest double-discovery).  This function finds every top-level
    function or class in *source* that still references such symbols and
    injects the required ``from … import …`` statement at the top of
    each body, after any docstring.

    *migrated_test_symbols* maps each migrated test name to its target
    file (relative path).  *abs_pkg* and *original_basename* are used to
    build the correct import prefix (absolute for test files, relative
    otherwise).
    """
    if not migrated_test_symbols:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    # Maps 1-based line number → list of import lines to insert before it.
    insertions: Dict[int, List[str]] = {}

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body_names: Set[str] = set()
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Name) and isinstance(subnode.ctx, ast.Load):
                body_names.add(subnode.id)
        needed: Dict[str, List[str]] = {}
        for name in body_names:
            tfile = migrated_test_symbols.get(name)
            if tfile:
                needed.setdefault(tfile, []).append(name)
        if not needed:
            continue
        import_stmts: List[str] = []
        for tfile, names in sorted(needed.items()):
            if abs_pkg is not None:
                mod = _target_module_name(tfile)
                prefix = f"{abs_pkg}.{mod}" if abs_pkg else mod
            else:
                prefix = _relative_import_prefix(original_basename, tfile)
            import_stmts.append(f"from {prefix} import {', '.join(sorted(names))}")
        first_stmt = node.body[0]
        insert_line = first_stmt.lineno
        if (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
            and len(node.body) > 1
        ):
            insert_line = node.body[1].lineno
        body_line = lines[insert_line - 1]
        indent = body_line[: len(body_line) - len(body_line.lstrip())]
        insertions.setdefault(insert_line, [])
        insertions[insert_line] = [f"{indent}{s}\n" for s in import_stmts] + insertions[
            insert_line
        ]

    if not insertions:
        return source
    result: List[str] = []
    for i, line in enumerate(lines, 1):
        if i in insertions:
            result.extend(insertions[i])
        result.append(line)
    return "".join(result)
