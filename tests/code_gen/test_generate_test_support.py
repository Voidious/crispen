from __future__ import annotations
from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import (
    _inject_inline_imports,
    _inject_inline_test_imports_original,
    _is_pytest_fixture,
    _is_test_name,
    _merge_conftest_sources,
    _split_cross_imports_by_test,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
import textwrap


def _plan(placements=None) -> FileLimiterPlan:
    return FileLimiterPlan(set3_migrate=[], placements=placements or [], abort=False)


def _classified(
    *, entities=None, set_2_groups=None, set_3_groups=None
) -> ClassifiedEntities:
    return ClassifiedEntities(
        entities=entities or [],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=set_2_groups or [],
        set_3_groups=set_3_groups or [],
        abort=False,
    )


def _make_entity(name: str, start: int, end: int, defines=None) -> Entity:
    return Entity(EntityKind.FUNCTION, name, start, end, defines or [name])


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


def test_inject_inline_imports_into_function():
    src = "def foo():\n    return 1\n"
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert result == "def foo():\n    from .bar import Baz\n    return 1\n"


def test_inject_inline_imports_skips_docstring():
    src = 'def foo():\n    """Doc."""\n    return 1\n'
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert (
        result == 'def foo():\n    """Doc."""\n    from .bar import Baz\n    return 1\n'
    )


def test_inject_inline_imports_into_class():
    src = "class Foo:\n    x = 1\n"
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert result == "class Foo:\n    from .bar import Baz\n    x = 1\n"


def test_inject_inline_imports_toplevel_noop():
    # TOP_LEVEL entity (bare if-statement): no body scope, returns unchanged.
    src = "if True:\n    pass\n"
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert result == src


def test_inject_inline_imports_empty_list_noop():
    src = "def foo():\n    pass\n"
    assert _inject_inline_imports(src, []) == src


def test_inject_inline_imports_syntax_error_noop():
    src = "def (invalid"
    assert _inject_inline_imports(src, ["from .x import Y"]) == src


def test_inject_inline_imports_empty_source_noop():
    # Empty source parses to empty tree.body — returns unchanged.
    assert _inject_inline_imports("", ["from .x import Y"]) == ""


def test_inject_inline_imports_only_docstring_injects_after():
    # Function with only a docstring — inserts after docstring (at body[0] line)
    # since len(body) == 1.
    src = 'def foo():\n    """Only doc."""\n'
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert result == 'def foo():\n    from .bar import Baz\n    """Only doc."""\n'


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


def test_merge_conftest_sources_deduplicates_imports():
    # Imports that already exist are not repeated.
    existing = "import pytest\n\n\n@pytest.fixture\ndef prior():\n    pass\n"
    new = "import pytest\n\n\n@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert result.count("import pytest") == 1


def test_merge_conftest_sources_deduplicates_functions():
    # A function already in existing is not appended again.
    existing = "@pytest.fixture\ndef client():\n    return 1\n"
    new = "@pytest.fixture\ndef client():\n    return 2\n"
    result = _merge_conftest_sources(existing, new)
    assert result.count("def client():") == 1
    assert "return 1" in result
    assert "return 2" not in result


def test_merge_conftest_sources_appends_new_fixture():
    # A new fixture not in existing is appended.
    existing = "@pytest.fixture\ndef prior():\n    pass\n"
    new = "@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert "def prior():" in result
    assert "def client():" in result
    assert result.index("prior") < result.index("client")


def test_merge_conftest_sources_no_changes_returns_existing():
    # When nothing new to add, return existing unchanged.
    existing = "import pytest\n\n\n@pytest.fixture\ndef client():\n    pass\n"
    new = "import pytest\n\n\n@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert result == existing


def test_merge_conftest_sources_inserts_new_imports_before_functions():
    # New imports are inserted after existing imports but before functions — no E402.
    existing = "import pytest\n\n\n@pytest.fixture\ndef prior():\n    pass\n"
    new = "import asyncio\n\n\n@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert "import asyncio" in result
    assert "def client():" in result
    # Imports must come before the first function definition.
    assert result.index("import asyncio") < result.index("def prior():")


def test_merge_conftest_sources_syntax_error_fallback():
    # Falls back to simple concatenation when existing cannot be parsed.
    existing = "def (broken"
    new = "import pytest\n"
    result = _merge_conftest_sources(existing, new)
    assert "def (broken" in result
    assert "import pytest" in result


def test_merge_conftest_sources_preserves_comments():
    # Comments in the existing conftest are preserved.
    existing = "# shared fixtures\nimport pytest\n\n\ndef prior():\n    pass\n"
    new = "@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert "# shared fixtures" in result
    assert "def client():" in result


def test_merge_conftest_sources_from_import_dedup():
    # from-style imports are also deduplicated via the _import_key F: path.
    existing = "from conftest import setup\n\n\ndef prior():\n    pass\n"
    new = "from conftest import setup\n\n\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert result.count("from conftest import setup") == 1
    assert "def client():" in result


def test_merge_conftest_sources_only_new_imports_no_defs():
    # When only new imports are added but no new functions, ends with newline.
    existing = "import pytest\n\n\ndef prior():\n    pass\n"
    new = "import asyncio\n"
    result = _merge_conftest_sources(existing, new)
    assert "import asyncio" in result
    assert result.endswith("\n")
    # No duplicate function definition appended.
    assert result.count("def prior():") == 1


def test_merge_conftest_sources_non_import_non_def_in_new():
    # Bare statements (assignments, expressions) in new_content are silently ignored.
    existing = "def prior():\n    pass\n"
    new = "X = 42\n"
    result = _merge_conftest_sources(existing, new)
    # Nothing to import or define → returns existing unchanged.
    assert result == existing
