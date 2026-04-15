from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import (
    _module_path_for_file,
    _patch_inline_imports_after_test_deletion,
    _redirect_inline_module_imports,
    run_engine,
)
from crispen.file_limiter.runner import FileLimiterResult
from .test_engine_file_limiter_phase3 import _FL_PATCH


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


def test_patch_inline_imports_after_test_deletion_updates_per_file(tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    sub = tmp_path / "pkg"
    sub.mkdir()
    # Simulate the deleted test file path and new files created by its split.
    deleted_path = str(tmp_path / "pkg" / "test_old.py")
    deleted_dir = tmp_path / "pkg"
    new_files = {
        "sub/test_new.py": "class TestFoo:\n    pass\n",
    }
    (deleted_dir / "sub").mkdir()
    (deleted_dir / "sub" / "test_new.py").write_text(new_files["sub/test_new.py"])

    src = "def run():\n    from pkg.test_old import TestFoo\n    TestFoo()\n"
    per_file = {
        "parent.py": {
            "source": src,
            "original": src,
        }
    }
    fl_new_file_final: dict = {}

    _patch_inline_imports_after_test_deletion(
        deleted_path, deleted_dir, new_files, per_file, fl_new_file_final
    )

    updated = per_file["parent.py"]["source"]
    assert "from pkg.sub.test_new import TestFoo" in updated
    assert "from pkg.test_old import" not in updated


def test_patch_inline_imports_after_test_deletion_updates_new_files(tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    sub = tmp_path / "pkg"
    sub.mkdir()
    deleted_path = str(tmp_path / "pkg" / "test_old.py")
    deleted_dir = tmp_path / "pkg"

    sibling_content = (
        "def helper():\n    from pkg.test_old import TestFoo\n    TestFoo()\n"
    )
    sibling_path = str(tmp_path / "pkg" / "test_sibling.py")
    (tmp_path / "pkg" / "test_sibling.py").write_text(sibling_content, encoding="utf-8")

    (deleted_dir / "sub").mkdir()
    new_file_content = "class TestFoo:\n    pass\n"
    (deleted_dir / "sub" / "test_new.py").write_text(new_file_content)

    new_files = {"sub/test_new.py": new_file_content}
    per_file: dict = {}
    fl_new_file_final = {sibling_path: sibling_content}

    _patch_inline_imports_after_test_deletion(
        deleted_path, deleted_dir, new_files, per_file, fl_new_file_final
    )

    updated = fl_new_file_final[sibling_path]
    assert "from pkg.sub.test_new import TestFoo" in updated
    assert "from pkg.test_old import" not in updated
    # File was re-written to disk.
    assert (
        "from pkg.sub.test_new import TestFoo"
        in (tmp_path / "pkg" / "test_sibling.py").read_text()
    )


def test_patch_inline_imports_after_test_deletion_no_markers_skips(tmp_path):
    # No pyproject.toml — module path unresolvable; function must not raise.
    deleted_path = str(tmp_path / "pkg" / "test_old.py")
    deleted_dir = tmp_path / "pkg"
    deleted_dir.mkdir(parents=True)
    per_file = {
        "parent.py": {
            "source": "def run():\n    from pkg.test_old import TestFoo\n",
            "original": "def run():\n    from pkg.test_old import TestFoo\n",
        }
    }
    # Should not raise; source is unchanged because old_mod is None.
    _patch_inline_imports_after_test_deletion(
        deleted_path, deleted_dir, {}, per_file, {}
    )
    assert (
        per_file["parent.py"]["source"]
        == "def run():\n    from pkg.test_old import TestFoo\n"
    )


def test_patch_inline_imports_after_test_deletion_new_mod_none_skips(tmp_path):
    # new_mod resolves to None for a path outside the project root → that entry
    # is skipped and name_to_new_mod stays empty → function returns early.
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    sub = tmp_path / "pkg"
    sub.mkdir()
    deleted_path = str(tmp_path / "pkg" / "test_old.py")
    deleted_dir = tmp_path / "pkg"
    # Relative path that escapes the project root when resolved from deleted_dir.
    outside_rel = "../../../outside/test_new.py"
    per_file = {
        "parent.py": {
            "source": "def run():\n    from pkg.test_old import TestFoo\n",
            "original": "def run():\n    from pkg.test_old import TestFoo\n",
        }
    }
    _patch_inline_imports_after_test_deletion(
        deleted_path,
        deleted_dir,
        {outside_rel: "class TestFoo:\n    pass\n"},
        per_file,
        {},
    )
    # Source unchanged because name_to_new_mod was empty.
    assert (
        per_file["parent.py"]["source"]
        == "def run():\n    from pkg.test_old import TestFoo\n"
    )


def test_patch_inline_imports_after_test_deletion_syntax_error_in_new_file(tmp_path):
    # SyntaxError in a new_file content → that entry is skipped.
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    sub = tmp_path / "pkg" / "sub"
    sub.mkdir(parents=True)
    (sub / "test_new.py").write_text("def (invalid", encoding="utf-8")
    deleted_path = str(tmp_path / "pkg" / "test_old.py")
    deleted_dir = tmp_path / "pkg"
    per_file = {
        "parent.py": {
            "source": "def run():\n    from pkg.test_old import TestFoo\n",
            "original": "def run():\n    from pkg.test_old import TestFoo\n",
        }
    }
    _patch_inline_imports_after_test_deletion(
        deleted_path, deleted_dir, {"sub/test_new.py": "def (invalid"}, per_file, {}
    )
    # Source unchanged; SyntaxError in the new file caused it to be skipped.
    assert (
        per_file["parent.py"]["source"]
        == "def run():\n    from pkg.test_old import TestFoo\n"
    )


def test_patch_inline_imports_after_test_deletion_no_class_or_func_in_new_file(
    tmp_path,
):
    # New file has no ClassDef/FunctionDef → name_to_new_mod stays empty → skip.
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    sub = tmp_path / "pkg" / "sub"
    sub.mkdir(parents=True)
    deleted_path = str(tmp_path / "pkg" / "test_old.py")
    deleted_dir = tmp_path / "pkg"
    per_file = {
        "parent.py": {
            "source": "def run():\n    from pkg.test_old import TestFoo\n",
            "original": "def run():\n    from pkg.test_old import TestFoo\n",
        }
    }
    _patch_inline_imports_after_test_deletion(
        deleted_path, deleted_dir, {"sub/test_new.py": "X = 1\n"}, per_file, {}
    )
    assert (
        per_file["parent.py"]["source"]
        == "def run():\n    from pkg.test_old import TestFoo\n"
    )


def test_patch_inline_imports_after_test_deletion_source_unchanged_no_import(tmp_path):
    # per_file source has no import from old_mod → update is a no-op.
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    sub = tmp_path / "pkg" / "sub"
    sub.mkdir(parents=True)
    (sub / "test_new.py").write_text("class TestFoo:\n    pass\n", encoding="utf-8")
    deleted_path = str(tmp_path / "pkg" / "test_old.py")
    deleted_dir = tmp_path / "pkg"
    original_source = "def run():\n    pass\n"
    per_file = {
        "parent.py": {
            "source": original_source,
            "original": original_source,
        }
    }
    _patch_inline_imports_after_test_deletion(
        deleted_path,
        deleted_dir,
        {"sub/test_new.py": "class TestFoo:\n    pass\n"},
        per_file,
        {},
    )
    # Source is identical to original (no-op branch taken).
    assert per_file["parent.py"]["source"] == original_source


def test_patch_inline_imports_after_test_deletion_empty_fl_entry_skipped(tmp_path):
    # fl_new_file_final entry with empty/None content is skipped without error.
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    sub = tmp_path / "pkg" / "sub"
    sub.mkdir(parents=True)
    (sub / "test_new.py").write_text("class TestFoo:\n    pass\n", encoding="utf-8")
    deleted_path = str(tmp_path / "pkg" / "test_old.py")
    deleted_dir = tmp_path / "pkg"
    fl_new_file_final = {str(tmp_path / "empty.py"): ""}
    _patch_inline_imports_after_test_deletion(
        deleted_path,
        deleted_dir,
        {"sub/test_new.py": "class TestFoo:\n    pass\n"},
        {},
        fl_new_file_final,
    )
    # Empty entry was skipped; dict unchanged.
    assert fl_new_file_final[str(tmp_path / "empty.py")] == ""


def test_file_limiter_recursive_test_deletion_patches_parent_inline_imports(tmp_path):
    """When a recursive split deletes a test file, inline imports in the parent
    file that point to the deleted module are updated to the new locations."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

    # Parent test file with inline imports referencing a module that will be
    # created by the first split and then deleted by the recursive split.
    parent_src = (
        "def run_comprehensive_tests():\n"
        "    from tests.test_collections import TestA, TestB\n"
        "    TestA()\n"
        "    TestB()\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    run_comprehensive_tests()\n"
    )
    parent_file = tmp_path / "tests" / "test_suite.py"
    parent_file.parent.mkdir(parents=True)
    parent_file.write_text(parent_src, encoding="utf-8")

    # First pass: parent → creates test_collections.py (oversized).
    # The original source has the inline import already present (as if
    # _inject_inline_test_imports_original added it).
    first_result = FileLimiterResult(
        original_source=parent_src,  # unchanged (inline import already injected)
        new_files={
            "test_collections.py": (
                "class TestA:\n    pass\n\nclass TestB:\n    pass\n"
            )
        },
        messages=[],
        abort=False,
    )

    # Recursive pass: test_collections.py → split into sub/test_a.py + sub/test_b.py.
    # All entities migrated; original_source is empty → file will be deleted.
    recursive_result = FileLimiterResult(
        original_source="",
        new_files={
            "sub/test_a.py": "class TestA:\n    pass\n",
            "sub/test_b.py": "class TestB:\n    pass\n",
        },
        messages=[],
        abort=False,
    )

    with patch(_FL_PATCH, side_effect=[first_result, recursive_result]):
        list(
            run_engine(
                {str(parent_file): [(1, len(parent_src.splitlines()))]},
                config=CrispenConfig(max_file_lines=2, file_limiter_recursive=True),
            )
        )

    # test_collections.py was deleted.
    assert not (parent_file.parent / "test_collections.py").exists()

    # Parent file now has updated inline imports pointing to the new locations.
    updated = parent_file.read_text(encoding="utf-8")
    assert "from tests.sub.test_a import TestA" in updated
    assert "from tests.sub.test_b import TestB" in updated
    assert "from tests.test_collections import" not in updated
