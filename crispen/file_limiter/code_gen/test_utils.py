from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
from .import_utils import _relative_import_prefix, _target_module_name
import ast
import re


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
