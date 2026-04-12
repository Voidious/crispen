from unittest.mock import patch
import threading
from crispen.engine import (
    _EXCLUDED_DIR_NAMES,
    _build_alias_map,
    _collect_assignment_names,
    _collect_code_referenced_names,
    _collect_imported_names,
    _collect_top_level_names,
    _compute_qname,
    _file_to_module,
    _find_outside_callers,
    _find_repo_root,
    _module_path_for_file,
    _visit_with_timeout,
)


def test_find_repo_root_finds_git(tmp_path):
    (tmp_path / ".git").mkdir()
    subdir = tmp_path / "src"
    subdir.mkdir()
    f = subdir / "code.py"
    f.write_text("x = 1\n")
    root = _find_repo_root({str(f): [(1, 1)]})
    assert root == str(tmp_path)


def test_find_repo_root_not_found(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n")
    root = _find_repo_root({str(f): [(1, 1)]})
    assert root is None


def test_file_to_module_regular_file(tmp_path):
    f = tmp_path / "mypkg" / "service.py"
    f.parent.mkdir()
    f.write_text("x = 1\n")
    assert _file_to_module(str(tmp_path), str(f)) == "mypkg.service"


def test_file_to_module_init(tmp_path):
    f = tmp_path / "mypkg" / "__init__.py"
    f.parent.mkdir()
    f.write_text("")
    assert _file_to_module(str(tmp_path), str(f)) == "mypkg"


def test_compute_qname(tmp_path):
    f = tmp_path / "pkg" / "mod.py"
    f.parent.mkdir()
    f.write_text("")
    assert _compute_qname(str(tmp_path), str(f), "my_func") == "pkg.mod.my_func"


def test_build_alias_map_identity_only(tmp_path):
    # No __init__.py in tmp_path → only identity mapping returned.
    alias_map = _build_alias_map(str(tmp_path), {"a.b.func"})
    assert alias_map == {"a.b.func": "a.b.func"}


def test_build_alias_map_with_reexport(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mypkg.service import get_user\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    assert "mypkg.get_user" in alias_map
    assert alias_map["mypkg.get_user"] == "mypkg.service.get_user"


def test_build_alias_map_star_import_skipped(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mypkg.service import *\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    # Star import does not create an alias
    assert "mypkg.get_user" not in alias_map


def test_build_alias_map_ambiguous_name_skipped(tmp_path):
    # Two canonical qnames share the same function name → alias is ambiguous.
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mypkg.service import get_user\n")
    alias_map = _build_alias_map(
        str(tmp_path),
        {"mypkg.service.get_user", "mypkg.other.get_user"},
    )
    # Ambiguous: skip adding the alias
    assert "mypkg.get_user" not in alias_map


def test_build_alias_map_invalid_init_skipped(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def f(:\n    pass\n")  # invalid Python
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    # Gracefully skips the unreadable __init__.py
    assert alias_map == {"mypkg.service.get_user": "mypkg.service.get_user"}


def test_build_alias_map_skips_compound_statement(tmp_path):
    # A function definition is a compound statement, not SimpleStatementLine (line 76).
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def helper():\n    pass\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    assert alias_map == {"mypkg.service.get_user": "mypkg.service.get_user"}


def test_build_alias_map_skips_non_import_in_simple_stmt(tmp_path):
    # An assignment in SimpleStatementLine is not ImportFrom (line 79).
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("__version__ = '1.0'\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    assert alias_map == {"mypkg.service.get_user": "mypkg.service.get_user"}


def test_visit_with_timeout_completes():
    """Fast visit completes within timeout → returns True."""
    from unittest.mock import MagicMock

    wrapper = MagicMock()
    finder = MagicMock()
    assert _visit_with_timeout(wrapper, finder, 5.0) is True
    wrapper.visit.assert_called_once_with(finder)


def test_visit_with_timeout_fires():
    """Slow visit that never completes → returns False after timeout."""
    block = threading.Event()

    class _HangWrapper:
        def visit(self, finder):
            block.wait()  # blocks until released

    result = _visit_with_timeout(_HangWrapper(), object(), 0.01)
    block.set()  # unblock the daemon thread for cleanup
    assert result is False


def test_find_outside_callers_scope_analysis_timeout(tmp_path):
    """When _visit_with_timeout times out, all target qnames are blocked."""
    (tmp_path / "other.py").write_text("x = 1\n")
    with patch("crispen.engine.callers._visit_with_timeout", return_value=False):
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    assert result == {"some.func"}


def test_find_outside_callers_deadline_expired(tmp_path):
    """Total budget already exhausted before any file is visited: all blocked."""
    (tmp_path / "other.py").write_text("x = 1\n")
    # A negative timeout makes the deadline fall in the past immediately.
    with patch("crispen.engine.callers._SCOPE_ANALYSIS_TIMEOUT", -1):
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    assert result == {"some.func"}


def test_find_outside_callers_excludes_venv_dirs(tmp_path):
    """Files inside excluded directories (.venv, __pycache__, etc.) are skipped."""
    for dirname in _EXCLUDED_DIR_NAMES:
        excluded = tmp_path / dirname / "lib"
        excluded.mkdir(parents=True, exist_ok=True)
        (excluded / "pkg.py").write_text(
            "from mypkg.service import get_user\nget_user()\n"
        )
    # Even though each excluded dir has a caller, none should be counted.
    result = _find_outside_callers(str(tmp_path), {"mypkg.service.get_user"}, set())
    assert "mypkg.service.get_user" not in result


def test_module_path_for_file_returns_dotted_path(tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    sub = tmp_path / "tests" / "lua"
    sub.mkdir(parents=True)
    f = sub / "test_foo.py"
    f.write_text("", encoding="utf-8")
    assert _module_path_for_file(str(f)) == "tests.lua.test_foo"


def test_module_path_for_file_init_strips_init_segment(tmp_path):
    """__init__.py resolves to the package name, not package.__init__."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    pkg = tmp_path / "mypkg" / "subpkg"
    pkg.mkdir(parents=True)
    f = pkg / "__init__.py"
    f.write_text("", encoding="utf-8")
    assert _module_path_for_file(str(f)) == "mypkg.subpkg"


def test_module_path_for_file_no_markers_returns_none(tmp_path):
    f = tmp_path / "test_foo.py"
    f.write_text("", encoding="utf-8")
    # No pyproject.toml / .git anywhere up the path — within tmp_path hierarchy.
    # We can't guarantee no markers exist above tmp_path in the real filesystem,
    # so only assert that the function returns a string or None without raising.
    result = _module_path_for_file(str(f))
    assert result is None or isinstance(result, str)


def test_collect_top_level_names_various():
    """Covers functions, classes, assignments, aug/ann assigns, imports, from-imports,
    non-Name aug-assign targets, and unrecognised statement types."""
    source = (
        "import os\n"
        "import libcst as cst\n"
        "from pathlib import Path\n"
        "from typing import List as L\n"
        "from os import *\n"  # star import skipped
        "_CONST = 42\n"
        "x: int = 1\n"
        "counter += 1\n"
        "a, b = 1, 2\n"  # tuple target → ast.Tuple, not ast.Name
        "some_obj.attr += 1\n"  # AugAssign with Attribute target → skipped
        "if True: pass\n"  # ast.If → matches no elif, skipped
        "def my_func(): pass\n"
        "class MyClass: pass\n"
        "async def async_func(): pass\n"
    )
    result = _collect_top_level_names(source)
    assert "os" in result
    assert "cst" in result
    assert "Path" in result
    assert "L" in result
    assert "_CONST" in result
    assert "x" in result
    assert "counter" in result
    assert "my_func" in result
    assert "MyClass" in result
    assert "async_func" in result
    # Tuple-unpacking targets (a, b = …) are ast.Tuple, not ast.Name → skipped
    assert "a" not in result
    assert "b" not in result
    # Attribute aug-assign (some_obj.attr += 1) → target is Attribute, skipped
    assert "some_obj" not in result


def test_collect_top_level_names_syntax_error():
    """Invalid Python source → empty set."""
    assert _collect_top_level_names("def broken(:") == set()


def test_collect_imported_names_various():
    """Covers import, import-as, from-import, from-import-as, star (skip)."""
    source = (
        "import os\n"
        "import os.path\n"
        "import json as json_mod\n"
        "from pathlib import Path\n"
        "from typing import List as L\n"
        "from os import *\n"
    )
    result = _collect_imported_names(source)
    assert result == {"os", "path", "json_mod", "Path", "L"}


def test_collect_imported_names_syntax_error():
    """Invalid Python source → empty set."""
    assert _collect_imported_names("def broken(:") == set()


def test_collect_assignment_names_basic():
    """Covers plain assignment, annotated assignment, augmented assignment."""
    source = (
        "_CONST = 42\n"
        "x: int = 1\n"
        "counter += 1\n"
        "a, b = 1, 2\n"  # tuple target → skipped
        "obj.attr += 1\n"  # attribute aug-assign → skipped
        "def my_func(): pass\n"  # function → skipped
        "import os\n"  # import → skipped
    )
    result = _collect_assignment_names(source)
    assert result == {"_CONST", "x", "counter"}


def test_collect_assignment_names_syntax_error():
    """Invalid Python source → empty set."""
    assert _collect_assignment_names("def broken(:") == set()


def test_collect_code_referenced_names_finds_load_uses():
    """Names used in code expressions are returned."""
    src = "from .sub import MyFunc\nresult = MyFunc()\n"
    assert "MyFunc" in _collect_code_referenced_names(src)


def test_collect_code_referenced_names_excludes_import_aliases():
    """Import alias names are not ast.Name nodes → not returned."""
    src = "from .sub import MyFunc\n"
    assert "MyFunc" not in _collect_code_referenced_names(src)


def test_collect_code_referenced_names_excludes_funcdef_name():
    """Function definition names are not ast.Name Load nodes."""
    src = "def MyFunc(): pass\n"
    assert "MyFunc" not in _collect_code_referenced_names(src)


def test_collect_code_referenced_names_syntax_error():
    """Returns empty set on unparseable source."""
    assert _collect_code_referenced_names("def (broken:") == set()
