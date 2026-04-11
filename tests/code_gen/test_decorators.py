from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
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


def test_generate_pytest_conftest_subdir_routes_to_subdir_conftest():
    # With pytest_conftest=True AND subdir_name set, fixtures go to
    # <subdir>/conftest.py (not the parent conftest.py).  This prevents
    # multiple test files in the same directory from conflicting when they
    # each have a fixture of the same name.
    src = "@pytest.fixture\ndef client():\n    pass\n"
    entity = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["client"], target_file="expr/fixtures.py")])

    result = generate_file_splits(
        c, plan, src, "test_big.py", subdir_name="expr", pytest_conftest=True
    )

    assert "expr/conftest.py" in result.new_files
    assert "def client():" in result.new_files["expr/conftest.py"]
    assert "conftest.py" not in result.new_files  # parent conftest untouched
    assert "import client" not in result.original_source


def test_generate_pytest_conftest_subdir_fixture_referenced_in_remaining_goes_to_parent():  # noqa: E501
    # When a fixture is migrated from a subdir split but its name still appears
    # in entities that remain in the original file, route it to the parent
    # conftest.py (not the subdir conftest) so those tests can find it.
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
    # Only the fixture is migrated; the test stays in the original.
    c = _classified(entities=[e_client, e_test])
    plan = _plan([GroupPlacement(group=["client"], target_file="expr/fixtures.py")])

    result = generate_file_splits(
        c, plan, src, "test_big.py", subdir_name="expr", pytest_conftest=True
    )

    # Fixture goes to parent conftest.py, not the subdir one.
    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]
    assert "expr/conftest.py" not in result.new_files
    # No import of client back into the original.
    assert "import client" not in result.original_source


def test_generate_pytest_conftest_subdir_fixture_overrides_parent_conftest(tmp_path):
    # When the fixture is referenced in remaining source AND the parent conftest
    # already has a fixture with the same name (the module was overriding it),
    # the fixture is *copied* (not moved) to the subdir conftest so migrated
    # tests get the override; the entity also stays in the original file so
    # the original test discovers it from its own module.
    parent_conftest = tmp_path / "conftest.py"
    parent_conftest.write_text(
        "@pytest.fixture\ndef client():\n    return 'base'\n", encoding="utf-8"
    )
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            return 'override'

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

    # Fixture goes to subdir conftest for migrated tests.
    assert "expr/conftest.py" in result.new_files
    assert "def client():" in result.new_files["expr/conftest.py"]
    assert "return 'override'" in result.new_files["expr/conftest.py"]
    # Parent conftest is NOT modified (would drop the override via merge).
    assert "conftest.py" not in result.new_files
    # Fixture stays in original file so the original test finds the override.
    assert "def client():" in result.original_source
    assert "return 'override'" in result.original_source
    # No re-export import injected.
    assert "import client" not in result.original_source


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
