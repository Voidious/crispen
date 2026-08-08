"""Direct tests for crispen.repo_index's public API.

Exhaustive edge-case coverage (relative import resolution depth, module/
package path derivation, etc.) already lives in test_patch_rewriter.py,
which exercises this module through patch_rewriter's `_cg_*` aliases. These
tests just cover the module's own public surface directly, since
duplicate_extractor will depend on it too.
"""

from __future__ import annotations

from crispen.repo_index import (
    EXCLUDED_DIR_NAMES,
    RepoIndex,
    build_repo_index,
    collect_defined_names,
    file_to_module_and_package,
    parse_imports,
)


def test_collect_defined_names():
    assert collect_defined_names("def foo(): pass\nclass Bar: pass\n") == {
        "foo",
        "Bar",
    }


def test_collect_defined_names_syntax_error():
    assert collect_defined_names("def f(:\n") == set()


def test_file_to_module_and_package(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    f = pkg / "helpers.py"
    f.touch()
    assert file_to_module_and_package(f, tmp_path) == ("pkg.helpers", "pkg")


def test_file_to_module_and_package_init(tmp_path):
    f = tmp_path / "pkg"
    f.mkdir()
    init = f / "__init__.py"
    init.touch()
    assert file_to_module_and_package(init, tmp_path) == ("pkg", "pkg")


def test_parse_imports_basic():
    assert parse_imports("from pkg.sub import foo\n", "pkg") == {
        "foo": ("pkg.sub", "foo")
    }


def test_parse_imports_syntax_error():
    assert parse_imports("def f(:\n", "pkg") == {}


def test_repo_index_get_imports_cached():
    index = RepoIndex(
        module_to_source={"pkg.mod": "import os\n"},
        module_to_package={"pkg.mod": "pkg"},
        module_to_defs={"pkg.mod": set()},
        file_to_module={},
    )
    r1 = index.get_imports("pkg.mod")
    r2 = index.get_imports("pkg.mod")
    assert r1 == r2 == {"os": ("os", "os")}
    assert "pkg.mod" in index._import_cache


def test_repo_index_get_imports_missing_module():
    index = RepoIndex(
        module_to_source={}, module_to_package={}, module_to_defs={}, file_to_module={}
    )
    assert index.get_imports("nonexistent") == {}


def test_build_repo_index_none_root():
    index = build_repo_index(None)
    assert index.module_to_source == {}
    assert index.file_to_module == {}


def test_build_repo_index_scans_repo(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text("def foo(): pass\n", encoding="utf-8")

    index = build_repo_index(str(tmp_path))

    assert index.module_to_source["pkg.mod"] == "def foo(): pass\n"
    assert index.module_to_package["pkg.mod"] == "pkg"
    assert index.module_to_defs["pkg.mod"] == {"foo"}


def test_build_repo_index_per_file_override(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    f = pkg / "mod.py"
    f.write_text("def foo(): pass\n", encoding="utf-8")
    abs_path = str(f.resolve())

    index = build_repo_index(str(tmp_path), {abs_path: "def bar(): pass\n"})

    assert index.module_to_source["pkg.mod"] == "def bar(): pass\n"
    assert index.module_to_defs["pkg.mod"] == {"bar"}


def test_build_repo_index_excludes_dir_names(tmp_path):
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "ignored.py").write_text("def nope(): pass\n", encoding="utf-8")

    index = build_repo_index(str(tmp_path))

    assert index.module_to_source == {}


def test_excluded_dir_names_contains_common_env_dirs():
    assert {".venv", "venv", "__pycache__"} <= EXCLUDED_DIR_NAMES
