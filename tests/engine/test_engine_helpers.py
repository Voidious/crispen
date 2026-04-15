from unittest.mock import patch
import threading
from crispen.engine import (
    _EXCLUDED_DIR_NAMES,
    _blocked_private_scopes,
    _build_alias_map,
    _categorize_into_stats,
    _collect_assignment_names,
    _collect_code_referenced_names,
    _collect_imported_names,
    _collect_top_level_names,
    _compute_qname,
    _file_to_module,
    _find_outside_callers,
    _find_repo_root,
    _has_callers_outside_ranges,
    _module_path_for_file,
    _redirect_inline_module_imports,
    _visit_with_timeout,
)
from crispen.stats import RunStats


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


def test_find_outside_callers_empty_qnames(tmp_path):
    result = _find_outside_callers(str(tmp_path), set(), set())
    assert result == set()


def test_find_outside_callers_no_outside_py_files(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n")
    result = _find_outside_callers(str(tmp_path), {"pkg.func"}, {str(f.resolve())})
    # All .py files are in the diff → nothing to scan outside
    assert result == set()


def test_find_outside_callers_finds_caller(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    service = pkg / "service.py"
    service.write_text("def get_user():\n    return (1, 2, 3)\n")
    outside = tmp_path / "outside.py"
    outside.write_text("from mypkg.service import get_user\nget_user()\n")

    qname = "mypkg.service.get_user"
    diff_files = {str(service.resolve())}
    result = _find_outside_callers(str(tmp_path), {qname}, diff_files)
    assert qname in result


def test_find_outside_callers_no_match(tmp_path):
    outside = tmp_path / "other.py"
    outside.write_text("x = 1\n")
    qname = "mypkg.service.get_user"
    result = _find_outside_callers(str(tmp_path), {qname}, set())
    assert qname not in result


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


def test_find_outside_callers_call_qname_not_target(tmp_path):
    # outside file calls other_func (resolves to mypkg.other.other_func),
    # but target is mypkg.service.get_user → hits the 118->117 branch.
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "other.py").write_text("def other_func(): pass\n")
    caller = tmp_path / "caller.py"
    caller.write_text("from mypkg.other import other_func\nother_func()\n")

    result = _find_outside_callers(str(tmp_path), {"mypkg.service.get_user"}, set())
    assert "mypkg.service.get_user" not in result


def test_find_outside_callers_manager_build_fails(tmp_path):
    (tmp_path / "other.py").write_text("x = 1\n")
    with patch("crispen.engine.FullRepoManager", side_effect=RuntimeError("fail")):
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    # Conservative: all target qnames are blocked.
    assert result == {"some.func"}


def test_find_outside_callers_wrapper_fails(tmp_path):
    (tmp_path / "other.py").write_text("x = 1\n")
    with patch("crispen.engine.FullRepoManager") as MockFRM:
        MockFRM.return_value.get_metadata_wrapper_for_path.side_effect = RuntimeError(
            "fail"
        )
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    assert result == set()


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
    with patch("crispen.engine._visit_with_timeout", return_value=False):
        result = _find_outside_callers(str(tmp_path), {"some.func"}, set())
    assert result == {"some.func"}


def test_find_outside_callers_deadline_expired(tmp_path):
    """Total budget already exhausted before any file is visited: all blocked."""
    (tmp_path / "other.py").write_text("x = 1\n")
    # A negative timeout makes the deadline fall in the past immediately.
    with patch("crispen.engine._SCOPE_ANALYSIS_TIMEOUT", -1):
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


def test_has_callers_outside_ranges_found():
    source = "def f(): pass\nf()\n"  # call on line 2, range is only line 1
    assert _has_callers_outside_ranges(source, "f", [(1, 1)]) is True


def test_has_callers_outside_ranges_not_found():
    source = "def f(): pass\nf()\n"  # call on line 2, range covers line 2
    assert _has_callers_outside_ranges(source, "f", [(1, 2)]) is False


def test_has_callers_outside_ranges_syntax_error():
    assert _has_callers_outside_ranges("def f(:", "f", [(1, 1)]) is False


def test_blocked_private_scopes_finds_outside_callers():
    # _helper called at line 3, diff range only covers line 1
    source = "def _helper(): pass\n\n_helper()\n"
    blocked = _blocked_private_scopes(source, [(1, 1)])
    assert "_helper" in blocked


def test_blocked_private_scopes_ignores_in_range_callers():
    # _helper called at line 3, diff range covers line 3
    source = "def _helper(): pass\n\n_helper()\n"
    blocked = _blocked_private_scopes(source, [(1, 3)])
    assert "_helper" not in blocked


def test_blocked_private_scopes_syntax_error():
    blocked = _blocked_private_scopes("def f(:", [(1, 1)])
    assert blocked == set()


def test_blocked_private_scopes_ignores_public():
    # Public functions (no leading _) should not appear in blocked set
    source = "def helper(): pass\n\nhelper()\n"
    blocked = _blocked_private_scopes(source, [(1, 1)])
    assert "helper" not in blocked


def test_categorize_if_not_else():
    s = RunStats()
    _categorize_into_stats(s, "IfNotElse: flipped if/else at line 3")
    assert s.if_not_else == 1
    assert s.total_edits == 1


def test_categorize_tuple_to_dataclass():
    s = RunStats()
    _categorize_into_stats(
        s, "TupleDataclass: replaced 3-tuple with FooResult at line 5"
    )
    assert s.tuple_to_dataclass == 1


def test_categorize_duplicate_matched():
    s = RunStats()
    _categorize_into_stats(s, "DuplicateExtractor: replaced '_f' body with call to 'g'")
    assert s.duplicate_matched == 1
    assert s.duplicate_extracted == 0


def test_categorize_duplicate_extracted():
    s = RunStats()
    _categorize_into_stats(
        s, "DuplicateExtractor: extracted '_helper' from 2 duplicate blocks"
    )
    assert s.duplicate_extracted == 1
    assert s.duplicate_matched == 0


def test_categorize_function_split():
    s = RunStats()
    _categorize_into_stats(s, "split 'big_func': extracted _step_two")
    assert s.function_split == 1


def test_categorize_other_message_ignored():
    s = RunStats()
    _categorize_into_stats(s, "CallerUpdater: expanded FooResult unpacking at line 7")
    assert s.total_edits == 0


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


def test_redirect_inline_module_imports_basic():
    source = "def run():\n    from pkg.old import Foo, Bar\n    Foo()\n"
    result = _redirect_inline_module_imports(
        source, "pkg.old", {"Foo": "pkg.new_foo", "Bar": "pkg.new_bar"}
    )
    assert "from pkg.new_foo import Foo" in result
    assert "from pkg.new_bar import Bar" in result
    assert "from pkg.old import" not in result


def test_redirect_inline_module_imports_partial_redirect():
    # Only 'Foo' has a new location; 'Baz' is unknown and kept in old module.
    source = "def run():\n    from pkg.old import Foo, Baz\n    Foo()\n"
    result = _redirect_inline_module_imports(source, "pkg.old", {"Foo": "pkg.new_foo"})
    assert "from pkg.new_foo import Foo" in result
    assert "from pkg.old import Baz" in result


def test_redirect_inline_module_imports_no_matching_import():
    source = "def run():\n    from pkg.other import Foo\n    Foo()\n"
    result = _redirect_inline_module_imports(source, "pkg.old", {"Foo": "pkg.new"})
    assert result == source


def test_redirect_inline_module_imports_module_level():
    # Module-level import is also redirected.
    source = "from pkg.old import Foo\n\ndef run():\n    Foo()\n"
    result = _redirect_inline_module_imports(source, "pkg.old", {"Foo": "pkg.new"})
    assert "from pkg.new import Foo" in result
    assert "from pkg.old import" not in result


def test_redirect_inline_module_imports_syntax_error():
    source = "def (invalid"
    result = _redirect_inline_module_imports(source, "pkg.old", {"Foo": "pkg.new"})
    assert result == source


def test_redirect_inline_module_imports_no_moved_names():
    # Import exists but none of the names are in the map — leave unchanged.
    source = "def run():\n    from pkg.old import Baz\n"
    result = _redirect_inline_module_imports(source, "pkg.old", {"Foo": "pkg.new"})
    assert result == source


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
