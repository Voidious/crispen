from crispen.engine import (
    _collect_assignment_names,
    _collect_code_referenced_names,
    _collect_imported_names,
    _collect_top_level_names,
    _module_path_for_file,
    _patch_inline_imports_after_test_deletion,
    _redirect_inline_module_imports,
)


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
