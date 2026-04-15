from __future__ import annotations
import textwrap
from crispen.file_limiter.code_gen import (
    _file_has_only_fixtures,
    _is_pytest_fixture,
    _is_test_name,
    _split_cross_imports_by_test,
)


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
