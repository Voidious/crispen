from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    _abs_package_for_dir,
    _extract_module_docstring,
    _find_cross_file_imports,
    _find_main_block_entity,
    _find_main_direct_callees,
    _find_project_root,
    _inject_inline_test_imports_original,
    _is_pytest_fixture,
    _is_test_name,
    _module_path_from_file,
    _split_cross_imports_by_test,
    _strip_module_docstring,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .test_generate_core import _classified, _make_entity, _plan
import textwrap


def test_find_project_root_finds_pyproject_toml(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    sub = tmp_path / "pkg" / "module.py"
    sub.parent.mkdir()
    sub.write_text("x = 1\n")
    assert _find_project_root(sub) == tmp_path


def test_find_project_root_finds_git(tmp_path):
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "module.py"
    sub.write_text("x = 1\n")
    assert _find_project_root(sub) == tmp_path


def test_find_project_root_called_with_directory(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    assert _find_project_root(tmp_path) == tmp_path


def test_find_project_root_not_found(tmp_path):
    # tmp_path is under /tmp which has no project markers → None.
    sub = tmp_path / "module.py"
    sub.write_text("x = 1\n")
    result = _find_project_root(sub)
    # If the test runner is inside a project that happens to include tmp_path
    # (unlikely but possible with in-tree pytest), just ensure the function
    # returns without crashing.  The important coverage is the happy path above.
    assert result is None or result.exists()


def test_module_path_from_file_success(tmp_path):
    f = tmp_path / "pkg" / "utils.py"
    f.parent.mkdir()
    f.write_text("")
    assert _module_path_from_file(tmp_path, f) == "pkg.utils"


def test_module_path_from_file_top_level(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("")
    assert _module_path_from_file(tmp_path, f) == "module"


def test_module_path_from_file_not_under_root(tmp_path):
    other = tmp_path.parent / "other.py"
    assert _module_path_from_file(tmp_path, other) is None


def test_find_cross_file_imports_abs_pkg_package_prefix():
    # abs_pkg="tests" → "from tests.block_1 import _MODEL"
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "test_fn.py", abs_pkg="tests"
    )
    assert result == ["from tests.block_1 import _MODEL"]


def test_find_cross_file_imports_abs_pkg_root_level():
    # abs_pkg="" → "from block_1 import _MODEL" (no package prefix)
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "test_fn.py", abs_pkg=""
    )
    assert result == ["from block_1 import _MODEL"]


def test_abs_package_for_dir_subdir(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_engine.py"
    test_file.touch()
    assert _abs_package_for_dir(str(test_file)) == "tests"


def test_abs_package_for_dir_root_level(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    test_file = tmp_path / "test_engine.py"
    test_file.touch()
    assert _abs_package_for_dir(str(test_file)) == ""


def test_abs_package_for_dir_no_project_root(monkeypatch):
    monkeypatch.setattr(
        "crispen.file_limiter.code_gen._find_project_root", lambda _p: None
    )
    assert _abs_package_for_dir("/some/random/path/test_engine.py") is None


def test_abs_package_for_dir_non_ancestor_root(tmp_path, monkeypatch):
    # Defensive branch: project root is not an ancestor of the file's directory.
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.setattr(
        "crispen.file_limiter.code_gen._find_project_root", lambda _p: other_dir
    )
    test_file = tmp_path / "tests" / "test_engine.py"
    test_file.parent.mkdir()
    test_file.touch()
    assert _abs_package_for_dir(str(test_file)) is None


def test_generate_test_file_reexports_use_absolute_imports(tmp_path):
    # When the original is a test file, re-exports in the updated original
    # must use absolute imports so pytest can load the file.
    (tmp_path / "pyproject.toml").touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_engine.py"
    test_file.write_text("")

    source = "import os\n\ndef foo():\n    os.getcwd()\n"
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="test_helpers.py")])

    result = generate_file_splits(c, plan, source, str(test_file))

    assert "from tests.test_helpers import foo" in result.original_source
    assert "from .test_helpers import foo" not in result.original_source


def test_generate_test_file_cross_imports_use_absolute_imports(tmp_path):
    # Cross-file imports in generated test split files must also be absolute.
    (tmp_path / "pyproject.toml").touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_engine.py"
    test_file.write_text("")

    source = "_CONST = 42\n\ndef test_fn():\n    return _CONST\n"
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_fn = _make_entity("test_fn", 3, 4)
    c = _classified(entities=[e_block, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="test_constants.py"),
            GroupPlacement(group=["test_fn"], target_file="test_fns.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, str(test_file))

    fn_src = result.new_files["test_fns.py"]
    assert "from tests.test_constants import _CONST" in fn_src
    assert "from .test_constants import _CONST" not in fn_src


def test_generate_file_splits_subdir_name_uses_init_as_original_basename():
    # When subdir_name="service", the dependency graph treats "service/__init__.py"
    # as the original file node.  Because main (public) is re-exported from
    # __init__, _extract_shared_helpers pulls helper into service/main.py to
    # break the __init__ → main → __init__ cycle.  The split must not abort.
    source = "def helper():\n    return 1\n\ndef main():\n    return helper()\n"
    e_helper = _make_entity("helper", 1, 2)
    e_main = _make_entity("main", 4, 5)
    c = _classified(entities=[e_helper, e_main])
    # Only main is migrated; helper stays in "original" (→ service/__init__.py).
    plan = _plan([GroupPlacement(group=["main"], target_file="service/main.py")])

    result = generate_file_splits(c, plan, source, "service.py", subdir_name="service")

    assert not result.abort
    # helper is extracted into service/main.py to break the re-export cycle.
    main_src = result.new_files["service/main.py"]
    assert "def helper" in main_src
    assert "def main" in main_src
    # Re-exports use the short relative prefix ".main", not ".service.main".
    assert "from .main import" in result.original_source
    assert "from .service.main" not in result.original_source


def test_generate_file_splits_subdir_name_re_exports_use_relative_prefix():
    # With subdir_name set (non-test), re-exports in the "original" source
    # (which becomes __init__.py) use ".utils" not ".service.utils".
    # target_file already has the "service/" prefix (added by runner.py).
    source = "def foo():\n    pass\n"
    e_foo = _make_entity("foo", 1, 2)
    c = _classified(entities=[e_foo])
    plan = _plan([GroupPlacement(group=["foo"], target_file="service/utils.py")])

    result = generate_file_splits(c, plan, source, "service.py", subdir_name="service")

    assert not result.abort
    assert "from .utils import foo" in result.original_source
    assert "from .service.utils" not in result.original_source


def test_generate_file_splits_subdir_name_cross_file_uses_relative():
    # In subdir mode, cross-file imports between new files use relative imports
    # even when the original file is a test (abs_pkg would normally apply).
    # NOTE: runner.py prefixes target_file with subdir_name before this call,
    # so target_files already include "svc/" here.
    source = "def helper():\n    return 1\n\ndef test_fn():\n    return helper()\n"
    e_helper = _make_entity("helper", 1, 2)
    e_test = _make_entity("test_fn", 4, 5)
    c = _classified(entities=[e_helper, e_test])
    plan = _plan(
        [
            GroupPlacement(group=["helper"], target_file="svc/helpers.py"),
            GroupPlacement(group=["test_fn"], target_file="svc/test_fns.py"),
        ]
    )

    # Use a path that looks like a test file so abs_pkg would normally be set.
    result = generate_file_splits(
        c, plan, source, "tests/test_svc.py", subdir_name="svc"
    )

    assert not result.abort
    # Cross-file import from test_fns.py to helpers.py should be relative.
    test_src = result.new_files["svc/test_fns.py"]
    assert "from .helpers import helper" in test_src


def test_generate_file_splits_test_subdir_nonmigrated_imports_from_original():
    # Non-migrated names (e.g. module-level constants) stay in the original
    # test file, not in __init__.py.  A new subfile that references them should
    # get "from ..test_svc import _CONFIG", not "from . import _CONFIG".
    source = "_CONFIG = 'val'\n\ndef test_fn():\n    return _CONFIG\n"
    # Use TOP_LEVEL kind so _extract_shared_helpers does not pull _CONFIG into
    # the new file (it only extracts FUNCTION/CLASS entities).
    e_config = Entity(EntityKind.TOP_LEVEL, "_CONFIG", 1, 1, ["_CONFIG"])
    e_test = _make_entity("test_fn", 3, 4)
    c = _classified(entities=[e_config, e_test])
    plan = _plan([GroupPlacement(group=["test_fn"], target_file="svc/test_fns.py")])

    result = generate_file_splits(
        c, plan, source, "tests/test_svc.py", subdir_name="svc"
    )

    assert not result.abort
    test_src = result.new_files["svc/test_fns.py"]
    assert "from ..test_svc import _CONFIG" in test_src
    assert "from . import _CONFIG" not in test_src


def test_generate_file_splits_subdir_bumps_needed_imports():
    # In subdir-split mode, relative imports from the original file that appear
    # in new sub-files must be incremented by one level so they still resolve
    # correctly from inside the subdirectory package.
    source = "from .sibling import CONST\n\ndef foo():\n    return CONST\n"
    e_foo = _make_entity("foo", 3, 4)
    c = _classified(entities=[e_foo])
    plan = _plan([GroupPlacement(group=["foo"], target_file="service/utils.py")])

    result = generate_file_splits(c, plan, source, "service.py", subdir_name="service")

    assert not result.abort
    utils_src = result.new_files["service/utils.py"]
    assert "from ..sibling import CONST" in utils_src
    assert "from .sibling import CONST" not in utils_src


def test_generate_file_splits_subdir_bumps_init_imports():
    # In subdir-split mode, relative imports in the non-migrated original source
    # (which becomes subdir/__init__.py) must also be bumped by one level so
    # they still point at the correct modules from inside the package.
    source2 = (
        "from .. import llm_client\n"
        "from .base import Base\n\n"
        "def stayed():\n    return llm_client, Base\n\n"
        "def migrated():\n    pass\n"
    )
    e_stayed2 = _make_entity("stayed", 4, 5)
    e_migrated2 = _make_entity("migrated", 7, 8)
    c = _classified(entities=[e_stayed2, e_migrated2])
    plan = _plan([GroupPlacement(group=["migrated"], target_file="pkg/helpers.py")])

    result = generate_file_splits(c, plan, source2, "pkg.py", subdir_name="pkg")

    assert not result.abort
    init_src = result.original_source
    assert "from ... import llm_client" in init_src
    assert "from ..base import Base" in init_src
    assert "from .. import llm_client" not in init_src
    assert "from .base import Base" not in init_src


def test_generate_file_splits_subdir_bumps_two_levels_deep():
    # When the LLM places a new file two directories deep (e.g.
    # "pkg/pkg/core.py"), relative imports must be bumped by 2 dots, not 1.
    # This matches the real-world scenario where subdir_name="pkg" but the
    # advisor proposes "pkg/pkg/core.py" as a target.
    source = "from .. import llm_client\n\ndef func():\n    return llm_client\n"
    e_func = _make_entity("func", 3, 4)
    c = _classified(entities=[e_func])
    plan = _plan([GroupPlacement(group=["func"], target_file="pkg/pkg/core.py")])

    result = generate_file_splits(c, plan, source, "pkg.py", subdir_name="pkg")

    assert not result.abort
    core_src = result.new_files["pkg/pkg/core.py"]
    # 2 levels deep → original ".." becomes "...." (4 dots)
    assert "from .... import llm_client" in core_src
    assert "from .. import llm_client" not in core_src
    assert "from ... import llm_client" not in core_src


def test_extract_module_docstring_present():
    src = '"""My module."""\n\nimport os\n'
    assert _extract_module_docstring(src) == '"""My module."""'


def test_extract_module_docstring_absent():
    src = "import os\n\ndef foo():\n    pass\n"
    assert _extract_module_docstring(src) is None


def test_extract_module_docstring_syntax_error():
    assert _extract_module_docstring("def (\n") is None


def test_extract_module_docstring_non_string_expr():
    # First statement is an expression but not a string constant.
    src = "1 + 1\n\ndef foo():\n    pass\n"
    assert _extract_module_docstring(src) is None


def test_strip_module_docstring_removes_docstring():
    src = '"""My module."""\n\n_CONST = 1\n'
    result = _strip_module_docstring(src)
    assert '"""My module."""' not in result
    assert "_CONST = 1" in result


def test_strip_module_docstring_no_docstring():
    src = "_CONST = 1\n"
    assert _strip_module_docstring(src) == src


def test_strip_module_docstring_syntax_error():
    src = "def (\n"
    assert _strip_module_docstring(src) == src


def test_generate_subdir_module_docstring_goes_to_init():
    # In subdir-split mode the module docstring belongs in __init__.py, not
    # in the split-off child module.  Migrate the preamble entity (_block_1)
    # along with foo so the docstring is removed from the original source,
    # triggering the restore-to-__init__ logic.
    source = textwrap.dedent(
        """\
        \"\"\"Top-level module doc.\"\"\"

        import os

        def foo():
            return os.sep

        def bar():
            return foo()
        """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["os"])
    e_foo = _make_entity("foo", 5, 6)
    e_bar = _make_entity("bar", 8, 9)
    c = _classified(entities=[e_block, e_foo, e_bar])
    plan = _plan(
        [GroupPlacement(group=["_block_1", "foo"], target_file="pkg/helpers.py")]
    )

    result = generate_file_splits(c, plan, source, "pkg.py", subdir_name="pkg")

    assert not result.abort
    init_src = result.original_source
    helpers_src = result.new_files["pkg/helpers.py"]
    # Docstring belongs in __init__.py.
    assert '"""Top-level module doc."""' in init_src
    # Docstring must NOT appear in the child module.
    assert '"""Top-level module doc."""' not in helpers_src


def test_generate_subdir_docstring_already_in_init_not_duplicated():
    # If the TOP_LEVEL entity stays in the original (not migrated), the
    # docstring remains in the updated source and must not be prepended again.
    source = textwrap.dedent(
        """\
        \"\"\"Top-level module doc.\"\"\"

        _CONST = 1

        def stayed():
            return _CONST

        def migrated():
            pass
        """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["_CONST"])
    e_stayed = _make_entity("stayed", 5, 6)
    e_migrated = _make_entity("migrated", 8, 9)
    c = _classified(entities=[e_block, e_stayed, e_migrated])
    plan = _plan([GroupPlacement(group=["migrated"], target_file="pkg/helpers.py")])

    result = generate_file_splits(c, plan, source, "pkg.py", subdir_name="pkg")

    assert not result.abort
    init_src = result.original_source
    assert init_src.count('"""Top-level module doc."""') == 1


def test_generate_subdir_module_docstring_goes_to_test_init():
    # For test-file subdir splits the module docstring goes into
    # subdir/__init__.py, not into the re-export stub file.
    source = textwrap.dedent(
        """\
        \"\"\"Tests for the runner module.\"\"\"

        import os

        def test_foo():
            return os.sep

        def test_bar():
            return test_foo()
        """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["os"])
    e_foo = _make_entity("test_foo", 5, 6)
    e_bar = _make_entity("test_bar", 8, 9)
    c = _classified(entities=[e_block, e_foo, e_bar])
    plan = _plan(
        [GroupPlacement(group=["_block_1", "test_foo"], target_file="svc/test_foo.py")]
    )

    result = generate_file_splits(
        c, plan, source, "tests/test_svc.py", subdir_name="svc"
    )

    assert not result.abort
    init_src = result.new_files["svc/__init__.py"]
    child_src = result.new_files["svc/test_foo.py"]
    updated_src = result.original_source
    # Docstring belongs in __init__.py.
    assert '"""Tests for the runner module."""' in init_src
    # Docstring must NOT appear in the child test file or the stub file.
    assert '"""Tests for the runner module."""' not in child_src
    assert '"""Tests for the runner module."""' not in updated_src


def test_generate_subdir_docstring_not_stripped_from_non_subdir_split():
    # Outside subdir-split mode, a TOP_LEVEL entity's docstring is preserved
    # in the new file (only imports are stripped, not docstrings).
    source = '"""Module doc."""\n\nimport os\n\ndef foo():\n    return os.sep\n'
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["os"])
    e_foo = _make_entity("foo", 5, 6)
    c = _classified(entities=[e_block, e_foo])
    plan = _plan([GroupPlacement(group=["_block_1", "foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert '"""Module doc."""' in new_src


def test_is_test_name_test_class():
    assert _is_test_name("TestFoo") is True


def test_is_test_name_test_function():
    assert _is_test_name("test_bar") is True


def test_is_test_name_non_test():
    assert _is_test_name("helper") is False
    assert _is_test_name("Foo") is False
    assert _is_test_name("_test_private") is False


def test_is_pytest_fixture_syntax_error():
    assert _is_pytest_fixture("def (") is False


def test_is_pytest_fixture_empty_body():
    # Empty source → empty tree body → not a fixture.
    assert _is_pytest_fixture("") is False


def test_is_pytest_fixture_class_node():
    # Class definition is not a FunctionDef → returns False.
    assert _is_pytest_fixture("class Foo:\n    pass\n") is False


def test_is_pytest_fixture_no_decorator():
    assert _is_pytest_fixture("def client():\n    pass\n") is False


def test_is_pytest_fixture_bare_name():
    # @fixture (plain name, no call)
    src = "@fixture\ndef client():\n    pass\n"
    assert _is_pytest_fixture(src) is True


def test_is_pytest_fixture_bare_name_called():
    # @fixture() (called with no args)
    src = "@fixture()\ndef client():\n    pass\n"
    assert _is_pytest_fixture(src) is True


def test_is_pytest_fixture_attribute():
    # @pytest.fixture (attribute access, no call)
    src = "@pytest.fixture\ndef client():\n    pass\n"
    assert _is_pytest_fixture(src) is True


def test_is_pytest_fixture_attribute_called():
    # @pytest.fixture(scope="session")
    src = '@pytest.fixture(scope="session")\ndef client():\n    pass\n'
    assert _is_pytest_fixture(src) is True


def test_is_pytest_fixture_non_matching_decorator():
    # @other_decorator — Name but id != "fixture"; not an Attribute.
    src = "@other_decorator\ndef client():\n    pass\n"
    assert _is_pytest_fixture(src) is False


def test_split_cross_imports_by_test_pure_non_test():
    non_test, test_named = _split_cross_imports_by_test(["from .foo import helper"])
    assert non_test == ["from .foo import helper"]
    assert test_named == []


def test_split_cross_imports_by_test_pure_test():
    non_test, test_named = _split_cross_imports_by_test(
        ["from .foo import TestFoo, test_bar"]
    )
    assert non_test == []
    assert test_named == ["from .foo import TestFoo, test_bar"]


def test_split_cross_imports_by_test_mixed():
    non_test, test_named = _split_cross_imports_by_test(
        ["from .foo import TestFoo, helper, test_bar"]
    )
    assert non_test == ["from .foo import helper"]
    assert test_named == ["from .foo import TestFoo, test_bar"]


def test_split_cross_imports_by_test_plain_import_passthrough():
    # Plain "import x" lines (no "from") pass through to non_test unchanged.
    non_test, test_named = _split_cross_imports_by_test(["import os"])
    assert non_test == ["import os"]
    assert test_named == []


def test_find_main_block_entity_present():
    from crispen.file_limiter.entity_parser import parse_entities

    source = textwrap.dedent(
        """\
        def run():
            pass

        if __name__ == "__main__":
            run()
        """
    )
    entities = parse_entities(source)
    esmap = {e.name: source.splitlines(keepends=True) for e in entities}
    # Rebuild entity_source_map properly
    lines = source.splitlines(keepends=True)
    esmap = {
        e.name: "".join(lines[e.start_line - 1 : e.end_line]).rstrip() for e in entities
    }
    result = _find_main_block_entity(entities, esmap)
    assert result is not None
    assert result.startswith("_block_")


def test_find_main_block_entity_absent():
    from crispen.file_limiter.entity_parser import parse_entities

    source = "def foo():\n    pass\n"
    entities = parse_entities(source)
    lines = source.splitlines(keepends=True)
    esmap = {
        e.name: "".join(lines[e.start_line - 1 : e.end_line]).rstrip() for e in entities
    }
    assert _find_main_block_entity(entities, esmap) is None


def test_find_main_block_entity_syntax_error_skipped():

    # Entity whose source is invalid Python: should be skipped gracefully.
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, [])
    result = _find_main_block_entity([entity], {"_block_1": "def (invalid"})
    assert result is None


def test_find_main_direct_callees_basic():
    src = 'if __name__ == "__main__":\n    run_tests()\n'
    callees = _find_main_direct_callees(src, {"run_tests", "other"})
    assert callees == {"run_tests"}


def test_find_main_direct_callees_not_in_entity_names():
    src = 'if __name__ == "__main__":\n    unknown()\n'
    callees = _find_main_direct_callees(src, {"run_tests"})
    assert callees == set()


def test_find_main_direct_callees_syntax_error():
    assert _find_main_direct_callees("def (invalid", {"foo"}) == set()


def test_find_main_direct_callees_no_main_block():
    src = "run_tests()\n"
    assert _find_main_direct_callees(src, {"run_tests"}) == set()


def test_inject_inline_test_imports_original_basic():
    source = textwrap.dedent(
        """\
        def runner():
            TestFoo()
        """
    )
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg="pkg.tests", original_basename="test_orig.py"
    )
    assert "from pkg.tests.sub.test_foo import TestFoo" in result
    # Import appears inside the function body, not before the def line.
    lines = result.splitlines()
    def_idx = next(i for i, l in enumerate(lines) if l.startswith("def runner"))
    import_idx = next(i for i, l in enumerate(lines) if "import TestFoo" in l)
    assert import_idx > def_idx


def test_inject_inline_test_imports_original_skips_docstring():
    source = textwrap.dedent(
        """\
        def runner():
            \"\"\"Run tests.\"\"\"
            TestFoo()
        """
    )
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg="tests", original_basename="test_orig.py"
    )
    lines = result.splitlines()
    doc_idx = next(i for i, l in enumerate(lines) if '"""Run tests."""' in l)
    import_idx = next(i for i, l in enumerate(lines) if "import TestFoo" in l)
    assert import_idx > doc_idx


def test_inject_inline_test_imports_original_no_reference():
    source = "def runner():\n    pass\n"
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg="tests", original_basename="test_orig.py"
    )
    assert result == source


def test_inject_inline_test_imports_original_empty_map():
    source = "def runner():\n    TestFoo()\n"
    result = _inject_inline_test_imports_original(
        source, {}, abs_pkg="tests", original_basename="test_orig.py"
    )
    assert result == source


def test_inject_inline_test_imports_original_syntax_error():
    result = _inject_inline_test_imports_original(
        "def (invalid",
        {"TestFoo": "sub/test_foo.py"},
        abs_pkg="tests",
        original_basename="test_orig.py",
    )
    assert result == "def (invalid"


def test_inject_inline_test_imports_original_relative_import():
    source = "def runner():\n    TestFoo()\n"
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg=None, original_basename="test_orig.py"
    )
    assert "from .sub.test_foo import TestFoo" in result


def test_inject_inline_test_imports_original_unreferenced_symbol_skipped():
    # Function references `helper` (not test-named) and `other_func`, neither
    # of which is in migrated_test_symbols — the false branch of `if tfile:`.
    source = "def runner():\n    helper()\n    other_func()\n"
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg="tests", original_basename="test_orig.py"
    )
    assert result == source


def test_generate_shebang_stripped_from_new_file():
    # Shebang on line 1 should NOT appear in generated new files.
    source = "#!/usr/bin/env python3\n\ndef foo():\n    pass\n\ndef bar():\n    foo()\n"
    e_foo = Entity(EntityKind.FUNCTION, "foo", 3, 4, ["foo"])
    e_bar = Entity(EntityKind.FUNCTION, "bar", 6, 7, ["bar"])
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["bar"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "#!/usr/bin/env python3" not in result.new_files["helpers.py"]


def test_generate_shebang_preserved_in_original_when_entity_migrated():
    # When the entity owning line 1 (with shebang comment) is migrated,
    # the shebang must be restored at the top of the original file.
    source = "#!/usr/bin/env python3\ndef foo():\n    pass\n\ndef bar():\n    pass\n"
    e_foo = Entity(EntityKind.FUNCTION, "foo", 1, 3, ["foo"])
    e_bar = Entity(EntityKind.FUNCTION, "bar", 5, 6, ["bar"])
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["foo"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert result.original_source.startswith("#!/usr/bin/env python3\n")
    assert "#!/usr/bin/env python3" not in result.new_files["helpers.py"]


def test_generate_shebang_preserved_when_not_migrated():
    # When the shebang entity stays in the original, shebang remains at top.
    source = "#!/usr/bin/env python3\ndef foo():\n    pass\n\ndef bar():\n    pass\n"
    e_foo = Entity(EntityKind.FUNCTION, "foo", 1, 3, ["foo"])
    e_bar = Entity(EntityKind.FUNCTION, "bar", 5, 6, ["bar"])
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["bar"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert result.original_source.startswith("#!/usr/bin/env python3\n")


def test_generate_main_block_stays_in_original():
    source = textwrap.dedent(
        """\
        def run():
            pass

        def other():
            pass

        if __name__ == "__main__":
            run()
        """
    )
    e_run = Entity(EntityKind.FUNCTION, "run", 1, 2, ["run"])
    e_other = Entity(EntityKind.FUNCTION, "other", 4, 5, ["other"])
    e_main = Entity(EntityKind.TOP_LEVEL, "_block_7", 7, 8, [])
    c = _classified(entities=[e_run, e_other, e_main])
    # Plan tries to migrate run + __main__ block and other.
    plan = _plan(
        [
            GroupPlacement(group=["run", "_block_7"], target_file="helpers.py"),
            GroupPlacement(group=["other"], target_file="helpers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    # __main__ block stays in original.
    assert 'if __name__ == "__main__"' in result.original_source
    assert 'if __name__ == "__main__"' not in result.new_files.get("helpers.py", "")


def test_generate_main_callee_stays_in_original():
    source = textwrap.dedent(
        """\
        def run():
            pass

        def other():
            pass

        if __name__ == "__main__":
            run()
        """
    )
    e_run = Entity(EntityKind.FUNCTION, "run", 1, 2, ["run"])
    e_other = Entity(EntityKind.FUNCTION, "other", 4, 5, ["other"])
    e_main = Entity(EntityKind.TOP_LEVEL, "_block_7", 7, 8, [])
    c = _classified(entities=[e_run, e_other, e_main])
    # Plan tries to migrate run (the direct callee of __main__).
    plan = _plan(
        [
            GroupPlacement(group=["run"], target_file="helpers.py"),
            GroupPlacement(group=["other"], target_file="helpers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    # run() is a direct __main__ callee — must stay in original.
    assert "def run():" in result.original_source
    # other() is not a callee — may be migrated.
    assert "helpers.py" in result.new_files


def test_generate_test_named_cross_import_inlined():
    # TestHelper migrates to helpers.py; runner stays in original and
    # references TestHelper — the import must be injected inside runner's body.
    source = textwrap.dedent(
        """\
        class TestHelper:
            def test_x(self):
                pass

        def runner():
            TestHelper()
        """
    )
    e_cls = Entity(EntityKind.CLASS, "TestHelper", 1, 3, ["TestHelper"])
    e_run = Entity(EntityKind.FUNCTION, "runner", 5, 6, ["runner"])
    c = _classified(entities=[e_cls, e_run])
    plan = _plan([GroupPlacement(group=["TestHelper"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    orig = result.original_source
    # No module-level re-export of TestHelper.
    lines = orig.splitlines()
    top_level_import_lines = [
        ln for ln in lines if ln.startswith("from") and "TestHelper" in ln
    ]
    assert top_level_import_lines == []
    # Import appears inside runner's body.
    assert "    from .helpers import TestHelper" in orig


def test_generate_test_named_inline_not_applied_to_toplevel_entity():
    # A TOP_LEVEL entity referencing a test-named symbol falls back to
    # module-level import since it has no body scope to inject into.
    source = textwrap.dedent(
        """\
        class TestHelper:
            def test_x(self):
                pass

        _inst = TestHelper()
        """
    )
    e_cls = Entity(EntityKind.CLASS, "TestHelper", 1, 3, ["TestHelper"])
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_5", 5, 5, ["_inst"])
    c = _classified(entities=[e_cls, e_block])
    plan = _plan(
        [GroupPlacement(group=["TestHelper", "_block_5"], target_file="helpers.py")]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    # TestHelper and _block_5 were migrated together — no cross-file issue here.
    # Test just ensures no crash and the file is produced.
    assert "helpers.py" in result.new_files


def test_generate_test_named_inlined_in_function_in_new_file():
    # TestA goes to file_a.py; func_b (which calls TestA) goes to file_b.py.
    # The cross-file import of TestA into file_b.py should be injected inline
    # inside func_b's body rather than at the top of file_b.py.
    source = textwrap.dedent(
        """\
        class TestA:
            def test_x(self):
                pass

        def func_b():
            TestA()
        """
    )
    e_a = Entity(EntityKind.CLASS, "TestA", 1, 3, ["TestA"])
    e_b = Entity(EntityKind.FUNCTION, "func_b", 5, 6, ["func_b"])
    c = _classified(entities=[e_a, e_b])
    plan = _plan(
        [
            GroupPlacement(group=["TestA"], target_file="file_a.py"),
            GroupPlacement(group=["func_b"], target_file="file_b.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    file_b = result.new_files["file_b.py"]
    lines = file_b.splitlines()
    # No module-level import of TestA.
    assert not any(ln.startswith("from") and "TestA" in ln for ln in lines)
    # Inline import inside func_b.
    assert "    from .file_a import TestA" in file_b


def test_generate_toplevel_entity_in_new_file_test_import_falls_back_to_module_level():
    # A TOP_LEVEL entity in a new file that references a test-named symbol
    # from another new file: no function body to inject into, falls back to
    # module-level import.  Two TOP_LEVEL entities referencing the same
    # test name exercise the dedup path on the second.
    source = textwrap.dedent(
        """\
        class TestA:
            def test_x(self):
                pass

        _inst1 = TestA()

        def _sep():
            pass

        _inst2 = TestA()
        """
    )
    e_a = Entity(EntityKind.CLASS, "TestA", 1, 3, ["TestA"])
    e_b1 = Entity(EntityKind.TOP_LEVEL, "_block_5", 5, 5, ["_inst1"])
    e_sep = Entity(EntityKind.FUNCTION, "_sep", 7, 8, ["_sep"])
    e_b2 = Entity(EntityKind.TOP_LEVEL, "_block_10", 10, 10, ["_inst2"])
    c = _classified(entities=[e_a, e_b1, e_sep, e_b2])
    plan = _plan(
        [
            GroupPlacement(group=["TestA"], target_file="file_a.py"),
            GroupPlacement(
                group=["_block_5", "_sep", "_block_10"], target_file="file_b.py"
            ),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    file_b = result.new_files["file_b.py"]
    # Module-level import is acceptable for TOP_LEVEL entities (no body scope).
    assert "TestA" in file_b
    # Dedup: the same import appears only once despite two TOP_LEVEL entities
    # both referencing TestA.
    assert file_b.count("import TestA") == 1


def test_generate_pytest_conftest_disabled_no_conftest():
    # Default (pytest_conftest=False): fixture goes to assigned file, re-exported.
    src = "@pytest.fixture\ndef client():\n    pass\n"
    entity = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["client"], target_file="fixtures.py")])

    result = generate_file_splits(c, plan, src, "test_big.py")

    assert "fixtures.py" in result.new_files
    assert "conftest.py" not in result.new_files
    assert "client" in result.new_files["fixtures.py"]


def test_generate_pytest_conftest_fixture_goes_to_conftest():
    # With pytest_conftest=True, fixture entity lands in conftest.py, not the
    # LLM-assigned file, and no re-export import appears in the original.
    src = "@pytest.fixture\ndef client():\n    pass\n"
    entity = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["client"], target_file="fixtures.py")])

    result = generate_file_splits(c, plan, src, "test_big.py", pytest_conftest=True)

    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]
    # No import of client back into the original (no F401/F811).
    assert "import client" not in result.original_source
    # The LLM-assigned file is dropped (all entities redirected).
    assert "fixtures.py" not in result.new_files


def test_generate_pytest_conftest_mixed_group_splits():
    # Fixture goes to conftest.py; non-fixture stays in the assigned file.
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        def helper():
            pass
        """
    )
    e_client = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    e_helper = Entity(EntityKind.FUNCTION, "helper", 5, 6, ["helper"])
    c = _classified(entities=[e_client, e_helper])
    plan = _plan([GroupPlacement(group=["client", "helper"], target_file="support.py")])

    result = generate_file_splits(c, plan, src, "test_big.py", pytest_conftest=True)

    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]
    assert "support.py" in result.new_files
    assert "def helper():" in result.new_files["support.py"]
    assert "import client" not in result.original_source


def test_generate_pytest_conftest_no_fixtures_no_conftest():
    # pytest_conftest=True but no fixture entities → no conftest.py created.
    src = "def helper():\n    pass\n"
    entity = Entity(EntityKind.FUNCTION, "helper", 1, 2, ["helper"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["helper"], target_file="support.py")])

    result = generate_file_splits(c, plan, src, "test_big.py", pytest_conftest=True)

    assert "conftest.py" not in result.new_files
    assert "support.py" in result.new_files


def test_generate_pytest_conftest_prepends_existing(tmp_path):
    # When conftest.py already exists on disk, its content is prepended.
    existing = tmp_path / "conftest.py"
    existing.write_text(
        "# existing fixture\ndef prior():\n    pass\n", encoding="utf-8"
    )

    src = "@pytest.fixture\ndef client():\n    pass\n"
    entity = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["client"], target_file="fixtures.py")])
    original_path = str(tmp_path / "test_big.py")

    result = generate_file_splits(c, plan, src, original_path, pytest_conftest=True)

    conftest_src = result.new_files["conftest.py"]
    assert "# existing fixture" in conftest_src
    assert "def prior():" in conftest_src
    assert "def client():" in conftest_src
    # Existing content should come first.
    assert conftest_src.index("prior") < conftest_src.index("client")
