from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import _merge_conftest_sources, generate_file_splits
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .helpers import _classified, _plan


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
