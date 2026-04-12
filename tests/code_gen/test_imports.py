from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    _inject_inline_test_imports_original,
    _is_test_name,
    _test_names_in_decorators,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .shared_helpers import _classified, _plan


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


def test_is_test_name_test_class():
    assert _is_test_name("TestFoo") is True


def test_is_test_name_test_function():
    assert _is_test_name("test_bar") is True


def test_is_test_name_non_test():
    assert _is_test_name("helper") is False
    assert _is_test_name("Foo") is False
    assert _is_test_name("_test_private") is False


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


def test_generate_cross_import_dedup_across_entities():
    # helper goes to helpers.py; foo and bar both go to workers.py and both
    # reference helper — the cross-file import should appear once (dedup).
    source = textwrap.dedent(
        """\
        def helper():
            pass

        def foo():
            helper()

        def bar():
            helper()
        """
    )
    e_h = Entity(EntityKind.FUNCTION, "helper", 1, 2, ["helper"])
    e_foo = Entity(EntityKind.FUNCTION, "foo", 4, 5, ["foo"])
    e_bar = Entity(EntityKind.FUNCTION, "bar", 7, 8, ["bar"])
    c = _classified(entities=[e_h, e_foo, e_bar])
    plan = _plan(
        [
            GroupPlacement(group=["helper"], target_file="helpers.py"),
            GroupPlacement(group=["foo", "bar"], target_file="workers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    workers = result.new_files["workers.py"]
    # "from .helpers import helper" should appear exactly once.
    assert workers.count("import helper") == 1
