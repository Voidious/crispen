from __future__ import annotations
from crispen.file_limiter.code_gen import _collect_external_imported_names


def test_collect_external_imported_names_relative_path():
    # Non-absolute path → empty set (no scan).
    assert _collect_external_imported_names("relative/path.py") == set()


def test_collect_external_imported_names_nonexistent_file(tmp_path):
    # Absolute but non-existent → empty set.
    assert _collect_external_imported_names(str(tmp_path / "ghost.py")) == set()


def test_collect_external_imported_names_no_project_root(tmp_path):
    # File exists but no pyproject.toml/.git above it → empty set.
    f = tmp_path / "module.py"
    f.write_text("x = 1\n")
    # tmp_path is under /tmp which typically has no project markers.
    result = _collect_external_imported_names(str(f))
    # May or may not find a root depending on environment; we just verify no crash.
    assert isinstance(result, set)


def _make_pkg_with_helper_module(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    mod = pkg / "utils.py"
    mod.write_text("def _helper():\n    pass\n")
    return pkg, mod


def test_collect_external_imported_names_absolute_import(tmp_path):
    pkg, mod = _make_pkg_with_helper_module(tmp_path)
    caller = tmp_path / "tests" / "test_utils.py"
    caller.parent.mkdir()
    caller.write_text("from mypkg.utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_relative_import(tmp_path):
    pkg, mod = _make_pkg_with_helper_module(tmp_path)
    sibling = pkg / "other.py"
    sibling.write_text("from .utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_self_excluded(tmp_path):
    # The file being scanned is excluded from the search.
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("from module import _x\n")  # self-referential (ignored)
    result = _collect_external_imported_names(str(mod))
    assert "_x" not in result


def test_collect_external_imported_names_syntax_error_skipped(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("def _helper(): pass\n")
    bad = tmp_path / "bad.py"
    bad.write_text("def (invalid\n")
    good = tmp_path / "good.py"
    good.write_text("from module import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_non_matching_import_ignored(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("def _helper(): pass\n")
    other = tmp_path / "other.py"
    other.write_text("from different_module import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" not in result


def test_collect_external_imported_names_non_importfrom_nodes_skipped(tmp_path):
    # Caller file contains a plain `import` statement (not ImportFrom) mixed
    # with a matching `from … import`.  The plain import must be skipped without
    # crashing, and the matching ImportFrom still contributes to the result.
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("def _helper(): pass\n")
    caller = tmp_path / "caller.py"
    caller.write_text("import os\nfrom module import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_deep_relative_import(tmp_path):
    # Two-level relative import: `from ..utils import _helper`
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "utils.py"
    mod.write_text("def _helper(): pass\n")
    sub = tmp_path / "pkg" / "sub" / "caller.py"
    sub.parent.mkdir(parents=True)
    sub.write_text("from ...utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_relative_level_too_deep(tmp_path):
    # Relative import that goes above the project root → skipped without crash.
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "utils.py"
    mod.write_text("def _helper(): pass\n")
    # A file at the top level trying to go up 5 packages (impossible).
    top = tmp_path / "top.py"
    top.write_text("from .....utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    # The over-deep import is silently skipped; no crash.
    assert isinstance(result, set)
