from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    _file_has_only_fixtures,
    _is_pytest_fixture,
    _test_names_in_decorators,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .helpers import _classified, _make_entity, _plan


def test_test_names_in_decorators_finds_name_in_decorator():
    src = (
        "@pytest.mark.parametrize('x', TestFixture.PARAMS)\ndef test_fn(x):\n    pass\n"
    )
    assert _test_names_in_decorators(src, {"TestFixture"}) == {"TestFixture"}


def test_test_names_in_decorators_name_only_in_body_not_found():
    src = "def test_fn():\n    TestFixture.setup()\n"
    assert _test_names_in_decorators(src, {"TestFixture"}) == set()


def test_test_names_in_decorators_syntax_error_returns_empty():
    assert _test_names_in_decorators("def (invalid", {"TestFixture"}) == set()


def test_test_names_in_decorators_class_decorator():
    src = "@TestFixture.mark\nclass TestSomething:\n    pass\n"
    assert _test_names_in_decorators(src, {"TestFixture"}) == {"TestFixture"}


def test_generate_aborts_when_test_class_used_in_decorator():
    # TestFixture (a Test* class) provides PARAMS used in a parametrize decorator
    # on test_fn.  If they are split into different files, TestFixture would need
    # to be imported inline (to avoid pytest duplicate collection), but that
    # import would not be in scope when the decorator is evaluated.
    source = textwrap.dedent(
        """\
        import pytest

        class TestFixture:
            PARAMS = [1, 2, 3]

        @pytest.mark.parametrize("x", TestFixture.PARAMS)
        def test_fn(x):
            assert x
        """
    )
    e_fixture = Entity(EntityKind.CLASS, "TestFixture", 3, 4, ["TestFixture"])
    e_fn = _make_entity("test_fn", 6, 8)
    c = _classified(entities=[e_fixture, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["TestFixture"], target_file="test_fixture.py"),
            GroupPlacement(group=["test_fn"], target_file="test_fns.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "tests/test_original.py")

    assert result.abort
    assert "TestFixture" in result.abort_reason
    assert "decorator" in result.abort_reason


def test_generate_non_migrated_helper_extracted_to_new_file():
    # _run is non-migrated; test_fn is migrated and references _run.
    # _run is extracted into test_helpers.py to prevent an O→F→O cycle.
    source = textwrap.dedent(
        """\
        import textwrap

        def _run(x):
            return x

        def test_fn():
            return _run(1)
    """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["textwrap"])
    e_run = _make_entity("_run", 3, 4)
    e_test = _make_entity("test_fn", 6, 7)
    c = _classified(entities=[e_block, e_run, e_test])
    plan = _plan([GroupPlacement(group=["test_fn"], target_file="test_helpers.py")])

    result = generate_file_splits(c, plan, source, "original.py")

    new_src = result.new_files["test_helpers.py"]
    # _run is defined in the new file (extracted), not imported from original
    assert "def _run" in new_src
    assert "from .original import _run" not in new_src
    # import textwrap is not referenced by either entity
    assert "from .original import textwrap" not in new_src


def test_generate_self_referential_placement_dropped():
    # LLM names a target file the same as the original → would create a
    # circular import.  The placement must be silently dropped so the entity
    # stays in the original file and no self-import is added.
    source = "class Foo:\n    pass\n\nclass Bar:\n    pass\n"
    e_foo = _make_entity("Foo", 1, 2)
    e_bar = _make_entity("Bar", 4, 5)
    c = _classified(entities=[e_foo, e_bar])
    # "mymodule.py" is also the original filename → self-referential
    plan = _plan(
        [
            GroupPlacement(group=["Foo"], target_file="mymodule.py"),
            GroupPlacement(group=["Bar"], target_file="helpers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "mymodule.py")

    # Foo stays in the original — no circular self-import
    assert "from .mymodule import Foo" not in result.original_source
    assert "mymodule.py" not in result.new_files
    # Bar is still moved normally
    assert "helpers.py" in result.new_files
    assert "class Bar" in result.new_files["helpers.py"]
    # Foo remains in the original source (not removed)
    assert "class Foo" in result.original_source


def test_generate_all_placements_self_referential():
    # All placements target the original file → nothing is moved.
    source = "def foo():\n    pass\n"
    e_foo = _make_entity("foo", 1, 2)
    c = _classified(entities=[e_foo])
    plan = _plan([GroupPlacement(group=["foo"], target_file="original.py")])

    result = generate_file_splits(c, plan, source, "original.py")

    assert result.new_files == {}
    assert "from .original import foo" not in result.original_source
    assert "def foo" in result.original_source


def test_generate_aborts_on_cross_file_import_cycle():
    # fn_a references fn_b (in b.py) and fn_b references fn_a (in a.py).
    # This creates a circular import a.py ↔ b.py that Python cannot load.
    # generate_file_splits must detect the cycle and abort rather than emit
    # broken code.
    source = "def fn_a():\n    return fn_b()\n\ndef fn_b():\n    return fn_a()\n"
    e_a = _make_entity("fn_a", 1, 2)
    e_b = _make_entity("fn_b", 4, 5)
    c = _classified(entities=[e_a, e_b])
    plan = _plan(
        [
            GroupPlacement(group=["fn_a"], target_file="a.py"),
            GroupPlacement(group=["fn_b"], target_file="b.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    assert result.abort is True
    assert result.new_files == {}


def test_generate_aborts_on_cycle_through_original():
    # _CONST is a TOP_LEVEL constant (stays in original).
    # _worker is migrated to helpers.py and references _CONST.
    # main() (non-migrated) calls _worker → original will re-export _worker.
    # Cycle: original → helpers.py (re-export of _worker)
    #              → original (via `from .original import _CONST`).
    source = textwrap.dedent(
        """\
        _CONST = "value"

        def _worker():
            return _CONST

        def main():
            return _worker()
    """
    )
    e_const = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_worker = _make_entity("_worker", 3, 4)
    e_main = _make_entity("main", 6, 7)
    c = _classified(entities=[e_const, e_worker, e_main])
    plan = _plan([GroupPlacement(group=["_worker"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "original.py")

    # helpers.py would need `from .original import _CONST` while original
    # re-exports _worker from helpers.py → circular import → must abort.
    assert result.abort is True
    assert result.new_files == {}


def test_generate_aborts_on_cycle_through_original_test_subdir():
    # In a test-file subdir split non_migrated_home ("test_svc.py") differs
    # from original_basename ("svc/__init__.py").  The cycle detection must
    # treat the original test file as its own graph node:
    #
    # _CONFIG stays in test_svc.py (TOP_LEVEL, non-migrated).
    # _helper is migrated to svc/test_helpers.py and references _CONFIG.
    # test_fn (non-migrated) calls _helper → test_svc.py re-exports _helper.
    # Cycle: test_svc.py → svc/test_helpers.py (re-export of _helper)
    #              → test_svc.py (via `from ..test_svc import _CONFIG`).
    source = textwrap.dedent(
        """\
        _CONFIG = "value"

        def _helper():
            return _CONFIG

        def test_fn():
            return _helper()
    """
    )
    e_config = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONFIG"])
    e_helper = _make_entity("_helper", 3, 4)
    e_test = _make_entity("test_fn", 6, 7)
    c = _classified(entities=[e_config, e_helper, e_test])
    plan = _plan([GroupPlacement(group=["_helper"], target_file="svc/test_helpers.py")])

    result = generate_file_splits(
        c, plan, source, "tests/test_svc.py", subdir_name="svc"
    )

    # svc/test_helpers.py imports _CONFIG from test_svc.py, and test_svc.py
    # re-exports _helper from svc/test_helpers.py → circular import → abort.
    assert result.abort is True
    assert result.new_files == {}


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


def test_generate_pytest_conftest_subdir_parent_conftest_imports_only(tmp_path):
    # When parent conftest exists but contains only imports (no function defs),
    # no conflict is detected and the fixture routes to parent conftest normally.
    parent_conftest = tmp_path / "conftest.py"
    parent_conftest.write_text("import pytest\n", encoding="utf-8")
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        def test_big(client):
            pass
        """
    )
    e_client = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    e_test = Entity(EntityKind.FUNCTION, "test_big", 5, 6, ["test_big"])
    c = _classified(entities=[e_client, e_test])
    plan = _plan([GroupPlacement(group=["client"], target_file="expr/fixtures.py")])
    original_path = str(tmp_path / "test_big.py")

    result = generate_file_splits(
        c, plan, src, original_path, subdir_name="expr", pytest_conftest=True
    )

    # No conflict in parent conftest → fixture routes to parent conftest.
    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]
    assert "expr/conftest.py" not in result.new_files


def test_generate_pytest_conftest_subdir_parent_conftest_syntax_error(tmp_path):
    # When parent conftest has a syntax error, the OSError/SyntaxError handler
    # silently ignores it (no names loaded), so no conflict is detected and the
    # fixture routes to parent conftest normally.
    parent_conftest = tmp_path / "conftest.py"
    parent_conftest.write_text("def (broken syntax", encoding="utf-8")
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        def test_big(client):
            pass
        """
    )
    e_client = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    e_test = Entity(EntityKind.FUNCTION, "test_big", 5, 6, ["test_big"])
    c = _classified(entities=[e_client, e_test])
    plan = _plan([GroupPlacement(group=["client"], target_file="expr/fixtures.py")])
    original_path = str(tmp_path / "test_big.py")

    result = generate_file_splits(
        c, plan, src, original_path, subdir_name="expr", pytest_conftest=True
    )

    # Unreadable parent conftest → no conflict detected → parent conftest.
    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]
    assert "expr/conftest.py" not in result.new_files


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


def test_generate_pytest_conftest_name_conflict_keeps_in_target(tmp_path):
    # When conftest.py already defines a function with the same name as the
    # fixture being routed, the fixture stays in its LLM-assigned target file
    # instead of being dropped by _merge_conftest_sources.  This preserves the
    # entity in the split output so that _verify_preservation passes.
    existing = tmp_path / "conftest.py"
    existing.write_text(
        "@pytest.fixture\nasync def client():\n    return 'old'\n", encoding="utf-8"
    )

    src = "@pytest.fixture\nasync def client():\n    return 'new'\n"
    entity = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["client"], target_file="fixtures.py")])
    original_path = str(tmp_path / "test_big.py")

    result = generate_file_splits(c, plan, src, original_path, pytest_conftest=True)

    # Fixture must appear in the output — in the LLM-assigned file, not conftest.
    assert "fixtures.py" in result.new_files
    assert "def client():" in result.new_files["fixtures.py"]
    # conftest.py should not be created/modified (no new fixtures were routed there).
    assert "conftest.py" not in result.new_files


def test_generate_pytest_conftest_name_conflict_mixed_group(tmp_path):
    # When a placement group contains both a conftest-conflict fixture AND a
    # regular function, the fixture is excluded from re-exports but the regular
    # function is still re-exported.  This covers the branch that rebuilds the
    # GroupPlacement with only the non-conflict names.
    existing = tmp_path / "conftest.py"
    existing.write_text(
        "@pytest.fixture\ndef client():\n    return 'old'\n", encoding="utf-8"
    )

    src = (
        "@pytest.fixture\ndef client():\n    return 'new'\n\n"
        "def helper():\n    pass\n"
    )
    e_client = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    e_helper = Entity(EntityKind.FUNCTION, "helper", 5, 6, ["helper"])
    c = _classified(entities=[e_client, e_helper])
    plan = _plan([GroupPlacement(group=["client", "helper"], target_file="helpers.py")])
    original_path = str(tmp_path / "test_big.py")

    result = generate_file_splits(c, plan, src, original_path, pytest_conftest=True)

    # Both entities migrate to helpers.py.
    assert "helpers.py" in result.new_files
    assert "def client():" in result.new_files["helpers.py"]
    assert "def helper():" in result.new_files["helpers.py"]
    # helper is re-exported (public non-fixture); client is not (conftest conflict).
    assert "helper" in result.original_source
    assert "client" not in result.original_source


def test_generate_pytest_conftest_unreadable_conftest_falls_through(tmp_path):
    # When conftest.py exists but has a syntax error, the OSError/SyntaxError
    # handler silently ignores it and routes the fixture to conftest normally.
    existing = tmp_path / "conftest.py"
    existing.write_text("def (broken syntax", encoding="utf-8")

    src = "@pytest.fixture\ndef client():\n    pass\n"
    entity = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["client"], target_file="fixtures.py")])
    original_path = str(tmp_path / "test_big.py")

    result = generate_file_splits(c, plan, src, original_path, pytest_conftest=True)

    # With unreadable conftest, routing proceeds normally → fixture goes to conftest.
    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]


def test_file_has_only_fixtures_syntax_error():
    assert _file_has_only_fixtures("def (") is False


def test_file_has_only_fixtures_empty():
    assert _file_has_only_fixtures("") is False


def test_file_has_only_fixtures_no_fixture():
    # Regular function only — not a fixture.
    assert _file_has_only_fixtures("def helper():\n    pass\n") is False


def test_file_has_only_fixtures_with_test_function():
    # Has both a fixture and a test function → not fixture-only.
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        def test_foo(client):
            pass
        """
    )
    assert _file_has_only_fixtures(src) is False


def test_file_has_only_fixtures_with_test_class():
    # Has both a fixture and a Test class → not fixture-only.
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        class TestFoo:
            pass
        """
    )
    assert _file_has_only_fixtures(src) is False


def test_file_has_only_fixtures_with_non_fixture_function():
    # Has a fixture and a plain helper function → not fixture-only.
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        def helper():
            pass
        """
    )
    assert _file_has_only_fixtures(src) is False


def test_file_has_only_fixtures_with_class():
    # Has a fixture and a regular class → not fixture-only.
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        class Config:
            pass
        """
    )
    assert _file_has_only_fixtures(src) is False


def test_file_has_only_fixtures_single_fixture():
    # Just a fixture and an import → fixture-only.
    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        def client():
            pass
        """
    )
    assert _file_has_only_fixtures(src) is True


def test_file_has_only_fixtures_multiple_fixtures():
    # Multiple fixtures with no tests → fixture-only.
    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        def client():
            pass

        @pytest.fixture
        def db():
            pass
        """
    )
    assert _file_has_only_fixtures(src) is True


def test_file_has_only_fixtures_async_fixture():
    # Async fixture → fixture-only.
    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        async def client():
            pass
        """
    )
    assert _file_has_only_fixtures(src) is True


def test_file_has_only_fixtures_with_docstring():
    # Module docstring + fixture → fixture-only (docstring is allowed).
    src = textwrap.dedent(
        """\
        \"\"\"Module docstring.\"\"\"

        import pytest

        @pytest.fixture
        def client():
            pass
        """
    )
    assert _file_has_only_fixtures(src) is True


def test_generate_stays_fixture_emptied_when_tests_migrated():
    # When a fixture "stays" in the original test file but all tests migrate
    # out, the original becomes fixture-only → route fixture to conftest.py
    # and empty the original so the engine deletes it.
    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        def client():
            pass

        def test_foo(client):
            pass
        """
    )
    e_fixture = Entity(EntityKind.FUNCTION, "client", 3, 5, ["client"])
    e_test = Entity(EntityKind.FUNCTION, "test_foo", 7, 8, ["test_foo"])
    c = _classified(entities=[e_fixture, e_test])
    # Only test_foo is migrated; client "stays" in original.
    plan = _plan(
        [GroupPlacement(group=["test_foo"], target_file="expression/test_foo.py")]
    )

    result = generate_file_splits(c, plan, src, "test_big.py", pytest_conftest=True)

    # Original should be empty (engine will delete it).
    assert result.original_source == ""
    # Fixture should be routed to conftest.py.
    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]


def test_generate_stays_fixture_merged_with_existing_conftest(tmp_path):
    # If conftest.py already exists on disk (e.g. same fixture already there),
    # the merge deduplicates so the fixture is not repeated.
    existing = tmp_path / "conftest.py"
    existing.write_text(
        "import pytest\n\n\n@pytest.fixture\ndef client():\n    pass\n",
        encoding="utf-8",
    )

    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        def client():
            pass

        def test_foo(client):
            pass
        """
    )
    e_fixture = Entity(EntityKind.FUNCTION, "client", 3, 5, ["client"])
    e_test = Entity(EntityKind.FUNCTION, "test_foo", 7, 8, ["test_foo"])
    c = _classified(entities=[e_fixture, e_test])
    plan = _plan(
        [GroupPlacement(group=["test_foo"], target_file="expression/test_foo.py")]
    )
    original_path = str(tmp_path / "test_expression.py")

    result = generate_file_splits(c, plan, src, original_path, pytest_conftest=True)

    assert result.original_source == ""
    # Fixture should appear exactly once in conftest.py (deduplicated).
    assert result.new_files["conftest.py"].count("def client():") == 1


def test_generate_stays_fixture_not_emptied_when_tests_remain():
    # If test functions still remain in the original, the fixture-only cleanup
    # does NOT trigger — the file should keep both fixture and test.
    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        def client():
            pass

        def test_foo(client):
            pass

        def test_bar(client):
            pass
        """
    )
    e_fixture = Entity(EntityKind.FUNCTION, "client", 3, 5, ["client"])
    e_test_foo = Entity(EntityKind.FUNCTION, "test_foo", 7, 8, ["test_foo"])
    e_test_bar = Entity(EntityKind.FUNCTION, "test_bar", 10, 11, ["test_bar"])
    c = _classified(entities=[e_fixture, e_test_foo, e_test_bar])
    # Only test_foo migrates; test_bar stays → original still has a test.
    plan = _plan(
        [GroupPlacement(group=["test_foo"], target_file="expression/test_foo.py")]
    )

    result = generate_file_splits(c, plan, src, "test_big.py", pytest_conftest=True)

    # Original should still contain the remaining test and fixture.
    assert "def test_bar" in result.original_source
    assert result.original_source != ""


def test_generate_stays_fixture_not_emptied_when_conftest_disabled():
    # When pytest_conftest=False, stranded fixtures are left in the original.
    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        def client():
            pass

        def test_foo(client):
            pass
        """
    )
    e_fixture = Entity(EntityKind.FUNCTION, "client", 3, 5, ["client"])
    e_test = Entity(EntityKind.FUNCTION, "test_foo", 7, 8, ["test_foo"])
    c = _classified(entities=[e_fixture, e_test])
    plan = _plan(
        [GroupPlacement(group=["test_foo"], target_file="expression/test_foo.py")]
    )

    result = generate_file_splits(c, plan, src, "test_big.py", pytest_conftest=False)

    # Original keeps the fixture (conftest routing disabled).
    assert "def client():" in result.original_source
    assert "conftest.py" not in result.new_files


def test_generate_stays_fixture_merged_with_already_written_conftest():
    # If conftest.py was already written by this same split run (e.g. another
    # entity was already routed there), merge into it rather than reading disk.
    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        def client():
            pass

        @pytest.fixture
        def db():
            pass

        def test_foo(client):
            pass
        """
    )
    e_client = Entity(EntityKind.FUNCTION, "client", 3, 5, ["client"])
    e_db = Entity(EntityKind.FUNCTION, "db", 7, 9, ["db"])
    e_test = Entity(EntityKind.FUNCTION, "test_foo", 11, 12, ["test_foo"])
    c = _classified(entities=[e_client, e_db, e_test])
    # db migrates (and goes to conftest.py via pytest routing); test_foo migrates;
    # client stays but is then stranded.
    plan = _plan(
        [
            GroupPlacement(group=["db"], target_file="fixtures.py"),
            GroupPlacement(group=["test_foo"], target_file="expression/test_foo.py"),
        ]
    )

    result = generate_file_splits(c, plan, src, "test_big.py", pytest_conftest=True)

    assert result.original_source == ""
    conftest_src = result.new_files["conftest.py"]
    # Both migrated db and stranded client fixtures should be in conftest.
    assert "def db():" in conftest_src
    assert "def client():" in conftest_src
