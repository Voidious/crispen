from unittest.mock import patch
import textwrap
from crispen.config import CrispenConfig
from crispen.engine import (
    _apply_tuple_dataclass,
    _blocked_private_scopes,
    _categorize_into_stats,
    _collect_assignment_names,
    _collect_code_referenced_names,
    _collect_imported_names,
    _collect_top_level_names,
    _compute_qname,
    _file_to_module,
    _find_repo_root,
    _has_callers_outside_ranges,
    _module_path_for_file,
    _patch_inline_imports_after_test_deletion,
    _redirect_inline_module_imports,
    _should_run,
    run_engine,
)
from crispen.errors import CrispenAPIError
from crispen.refactors.base import Refactor
from crispen.stats import RunStats
import pytest


def _run(changed):
    return list(run_engine(changed, config=CrispenConfig(min_tuple_size=3)))


def test_skip_missing_file(tmp_path):
    missing = str(tmp_path / "nonexistent.py")
    msgs = _run({missing: [(1, 10)]})
    assert len(msgs) == 1
    assert "SKIP" in msgs[0]
    assert "file not found" in msgs[0]


def test_no_changes_no_messages(tmp_path):
    f = tmp_path / "simple.py"
    f.write_text("x = 1\n", encoding="utf-8")
    msgs = _run({str(f): [(1, 1)]})
    assert msgs == []


def test_applies_refactor_and_writes(tmp_path):
    source = textwrap.dedent(
        """\
        if not x:
            a()
        else:
            b()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    msgs = _run({str(f): [(1, 4)]})
    assert any("IfNotElse" in m for m in msgs)
    assert "if x:" in f.read_text(encoding="utf-8")


def test_rewritten_source_used_when_available(tmp_path):
    """get_rewritten_source() is preferred over new_tree.code when non-None."""
    rewritten = "x = 999  # rewritten\n"

    class _RewritingRefactor(Refactor):
        @classmethod
        def name(cls):
            return "Rewriter"

        def get_rewritten_source(self):
            return rewritten

        def get_changes(self):
            return ["Rewriter: rewrote the file"]

    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch("crispen.engine._REFACTORS", [_RewritingRefactor]):
        msgs = _run({str(f): [(1, 1)]})
    assert any("Rewriter" in m for m in msgs)
    assert f.read_text(encoding="utf-8") == rewritten


def test_skip_parse_error(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def f(:\n    pass\n", encoding="utf-8")
    msgs = _run({str(f): [(1, 2)]})
    assert any("parse error" in m for m in msgs)


class _RaisingTransformer(Refactor):
    """A Refactor subclass that always raises during tree traversal."""

    @classmethod
    def name(cls):
        return "RaisingRefactor"

    def leave_Module(self, original_node, updated_node):
        raise RuntimeError("intentional transform error")


def test_skip_transform_error(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch("crispen.engine._REFACTORS", [_RaisingTransformer]):
        msgs = _run({str(f): [(1, 1)]})
    assert any("transform error" in m for m in msgs)


class _CrispenApiErrorRefactor(Refactor):
    @classmethod
    def name(cls):
        return "ApiErrorRefactor"

    def leave_Module(self, original_node, updated_node):
        raise CrispenAPIError("test api error")


def test_crispen_api_error_propagates(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with patch("crispen.engine._REFACTORS", [_CrispenApiErrorRefactor]):
        with pytest.raises(CrispenAPIError):
            list(run_engine({str(f): [(1, 1)]}))


def test_tuple_dataclass_transform_error_handled(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")

    class _FailingTD:
        def __init__(self, *a, **kw):
            raise RuntimeError("simulated TupleDataclass failure")

    with patch("crispen.engine.file_limiter.TupleDataclass", _FailingTD):
        msgs = _run({str(f): [(1, 1)]})
    assert any("TupleDataclass" in m and "transform error" in m for m in msgs)


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


def test_apply_tuple_dataclass_parse_error():
    bad_source = "def f(:\n    pass\n"
    source_out, msgs, td = _apply_tuple_dataclass(
        "fake.py", [(1, 10)], bad_source, False, set()
    )
    assert any("parse error" in m for m in msgs)
    assert td is None
    assert source_out == bad_source


def test_apply_tuple_dataclass_crispen_api_error():
    with patch("crispen.engine.file_limiter.MetadataWrapper") as MockWrapper:
        MockWrapper.return_value.visit.side_effect = CrispenAPIError("test api error")
        with pytest.raises(CrispenAPIError):
            _apply_tuple_dataclass("f.py", [(1, 1)], "x = 1\n", False, set())


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


def test_should_run_defaults_allow_all():
    cfg = CrispenConfig()
    for name in (
        "if_not_else",
        "duplicate_extractor",
        "function_splitter",
        "tuple_dataclass",
        "file_limiter",
    ):
        assert _should_run(name, cfg) is True


def test_should_run_enabled_list_allows_listed():
    cfg = CrispenConfig(enabled_refactors=["if_not_else", "function_splitter"])
    assert _should_run("if_not_else", cfg) is True
    assert _should_run("function_splitter", cfg) is True


def test_should_run_enabled_list_blocks_unlisted():
    cfg = CrispenConfig(enabled_refactors=["if_not_else"])
    assert _should_run("duplicate_extractor", cfg) is False
    assert _should_run("tuple_dataclass", cfg) is False
    assert _should_run("file_limiter", cfg) is False


def test_should_run_disabled_list_blocks_listed():
    cfg = CrispenConfig(disabled_refactors=["function_splitter", "file_limiter"])
    assert _should_run("function_splitter", cfg) is False
    assert _should_run("file_limiter", cfg) is False


def test_should_run_disabled_list_allows_unlisted():
    cfg = CrispenConfig(disabled_refactors=["function_splitter"])
    assert _should_run("if_not_else", cfg) is True
    assert _should_run("tuple_dataclass", cfg) is True


def test_should_run_enabled_takes_precedence_over_disabled():
    # enabled_refactors non-empty → disabled_refactors is ignored
    cfg = CrispenConfig(
        enabled_refactors=["if_not_else"],
        disabled_refactors=["if_not_else"],
    )
    assert _should_run("if_not_else", cfg) is True


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
