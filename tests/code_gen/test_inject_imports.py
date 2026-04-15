from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    _extract_module_docstring,
    _inject_inline_imports,
    _inject_inline_test_imports_original,
    _inject_module_level_imports,
    _inject_type_checking_imports,
    _strip_module_docstring,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .helpers import _classified, _make_entity, _plan


def test_inject_module_level_imports_docstring_only():
    # Source with only a docstring and no imports — insert after the docstring.
    src = '"""Module doc."""\n\nx = 1\n'
    result = _inject_module_level_imports(src, ["from . import converters"])
    assert '"""Module doc."""' in result
    assert "from . import converters" in result
    doc_pos = result.index('"""Module doc."""')
    imp_pos = result.index("from . import converters")
    assert doc_pos < imp_pos


def test_inject_module_level_imports_empty_list():
    src = "x = 1\n"
    assert _inject_module_level_imports(src, []) == src


def test_inject_module_level_imports_after_imports():
    src = "import os\n\nx = 1\n"
    result = _inject_module_level_imports(src, ["from . import converters"])
    assert result == "import os\nfrom . import converters\n\nx = 1\n"


def test_inject_module_level_imports_no_existing_imports():
    src = "x = 1\n"
    result = _inject_module_level_imports(src, ["from . import converters"])
    # Prepended before non-import content
    assert "from . import converters" in result
    assert result.index("from . import converters") < result.index("x = 1")


def test_inject_module_level_imports_sorted():
    src = "import os\n\nx = 1\n"
    result = _inject_module_level_imports(
        src, ["from . import z_mod", "from . import a_mod"]
    )
    lines = result.splitlines()
    import_lines = [ln for ln in lines if "import" in ln]
    assert import_lines.index("from . import a_mod") < import_lines.index(
        "from . import z_mod"
    )


def test_inject_module_level_imports_syntax_error_prepends():
    src = "def (broken:\n"
    result = _inject_module_level_imports(src, ["import os"])
    assert result.startswith("import os\n")


def test_inject_type_checking_imports_empty_list():
    src = "import os\n"
    assert _inject_type_checking_imports(src, []) == src


def test_inject_type_checking_imports_syntax_error():
    src = "def (broken:\n"
    assert _inject_type_checking_imports(src, ["from .config import Cfg"]) == src


def test_inject_type_checking_imports_all_already_present():
    # If every requested import is already in an existing TC block, no change.
    src = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .config import Cfg\n"
        "\n"
        "x = 1\n"
    )
    result = _inject_type_checking_imports(src, ["from .config import Cfg"])
    assert result == src


def test_inject_type_checking_imports_appends_to_existing_block():
    # New import should be appended inside the existing TYPE_CHECKING block.
    src = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .config import Cfg\n"
        "\n"
        "x = 1\n"
    )
    result = _inject_type_checking_imports(src, ["from .models import MyModel"])
    assert "from .models import MyModel" in result
    tc_start = result.index("if TYPE_CHECKING:")
    assert result.index("from .models import MyModel") > tc_start
    assert "x = 1" in result


def test_inject_type_checking_imports_creates_block_with_typing_import():
    # No existing TC block and TYPE_CHECKING not imported → add both.
    src = "from typing import List\n\ndef foo(x: 'Cfg') -> None:\n    pass\n"
    result = _inject_type_checking_imports(src, ["from .config import Cfg"])
    assert "from typing import TYPE_CHECKING" in result
    assert "if TYPE_CHECKING:" in result
    assert "    from .config import Cfg" in result


def test_inject_type_checking_imports_creates_block_type_checking_already_imported():
    # TYPE_CHECKING already in typing import → don't add it again.
    src = (
        "from typing import List, TYPE_CHECKING\n"
        "\n"
        "def foo(x: 'Cfg') -> None:\n"
        "    pass\n"
    )
    result = _inject_type_checking_imports(src, ["from .config import Cfg"])
    assert result.count("TYPE_CHECKING") == 2  # one in import, one in if-block
    assert "if TYPE_CHECKING:" in result
    assert "    from .config import Cfg" in result


def test_inject_type_checking_imports_block_after_last_import():
    # The new block should appear after the last import, before other code.
    src = "import os\nimport sys\n\nx = 1\n"
    result = _inject_type_checking_imports(src, ["from .config import Cfg"])
    lines = result.splitlines()
    sys_line = next(i for i, l in enumerate(lines) if "import sys" in l)
    if_line = next(i for i, l in enumerate(lines) if "if TYPE_CHECKING" in l)
    x_line = next(i for i, l in enumerate(lines) if "x = 1" in l)
    assert sys_line < if_line < x_line


def test_generate_cross_file_import():
    # fn_a goes to fn_module.py; _block_1 (defining _CONST) goes to constants.py.
    # _CONST is a TOP_LEVEL variable that is never reassigned → fn_module.py uses
    # a plain "from .constants import _CONST" (idiomatic Python; no module alias).
    source = "_CONST = 42\n\ndef fn_a():\n    return _CONST\n"
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_fn = _make_entity("fn_a", 3, 4)
    c = _classified(entities=[e_block, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["fn_a"], target_file="fn_module.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    fn_src = result.new_files["fn_module.py"]
    assert "from .constants import _CONST" in fn_src
    assert "from . import constants" not in fn_src
    assert "constants._CONST" not in fn_src
    # constants.py should NOT have a cross-import (it defines _CONST, not uses it)
    const_src = result.new_files["constants.py"]
    assert "from .fn_module" not in const_src


def test_generate_cross_file_import_no_duplicate_names():
    # Two entities (fn_a and fn_b) migrate to the same new file.
    # fn_a uses X and Z from helpers; fn_b uses Y and Z from helpers.
    # X, Y, Z are TOP_LEVEL variables that are never reassigned → the new file
    # gets ONE "from .constants import X, Y, Z" (no module alias needed).
    source = textwrap.dedent(
        """\
        X = 1
        Y = 2
        Z = 3

        def fn_a():
            return X + Z

        def fn_b():
            return Y + Z
        """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["X", "Y", "Z"])
    e_a = _make_entity("fn_a", 5, 6)
    e_b = _make_entity("fn_b", 8, 9)
    c = _classified(entities=[e_block, e_a, e_b])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["fn_a", "fn_b"], target_file="funcs.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    funcs_src = result.new_files["funcs.py"]
    # Both fn_a and fn_b are present
    assert "def fn_a" in funcs_src
    assert "def fn_b" in funcs_src
    # Plain from-import (no module alias) since none of X/Y/Z are reassigned
    assert "from .constants import" in funcs_src
    assert "from . import constants" not in funcs_src
    # Variables are referenced by their bare names, not as module attributes
    assert "constants.X" not in funcs_src
    assert "constants.Y" not in funcs_src
    assert "constants.Z" not in funcs_src


def test_generate_cross_file_import_reassigned_uses_module_alias():
    # _CONST is defined by _block_1 (→ constants.py) AND reassigned by _block_2
    # (non-migrated, stays in big.py).  Because _CONST is stored by a different
    # entity, fn_module.py must use the module-alias form so that any mutation of
    # _CONST propagates through the module reference rather than a stale copy.
    source = textwrap.dedent(
        """\
        _CONST = 42
        _CONST = int("99")

        def fn_a():
            return _CONST
        """
    )
    e_block1 = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_block2 = Entity(EntityKind.TOP_LEVEL, "_block_2", 2, 2, ["_CONST"])
    e_fn = _make_entity("fn_a", 4, 5)
    c = _classified(entities=[e_block1, e_block2, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["fn_a"], target_file="fn_module.py"),
            # _block_2 stays (non-migrated) — its store makes _CONST "reassigned"
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    fn_src = result.new_files["fn_module.py"]
    # _CONST is reassigned → module-alias import so mutations propagate.
    assert "from . import constants" in fn_src
    assert "constants._CONST" in fn_src
    assert "from .constants import _CONST" not in fn_src


def test_generate_cross_file_reassigned_original_file_uses_module_alias():
    # _CONST is defined by _block_1 (migrated) and reassigned by _block_2
    # (non-migrated).
    # The original file must rewrite both the load in fn_a AND the module-level
    # store in _block_2 to constants._CONST so that the reassignment updates the
    # value in constants.py rather than creating an orphaned local binding.
    source = textwrap.dedent(
        """\
        _CONST = 42
        _CONST = int("99")

        def fn_a():
            return _CONST
        """
    )
    e_block1 = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_block2 = Entity(EntityKind.TOP_LEVEL, "_block_2", 2, 2, ["_CONST"])
    e_fn = _make_entity("fn_a", 4, 5)
    c = _classified(entities=[e_block1, e_block2, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            # _block_2 and fn_a stay (non-migrated)
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    assert not result.abort
    orig = result.original_source
    # Module-level import added for the module alias.
    assert "from . import constants" in orig
    # Both the store (_block_2) and the load (fn_a) are rewritten.
    assert 'constants._CONST = int("99")' in orig
    assert "return constants._CONST" in orig
    # Must NOT bind _CONST as a bare name via from-import (would shadow the rewrite)
    assert "from .constants import _CONST" not in orig


def test_generate_reassigned_all_entities_migrated_no_original_processing():
    # When ALL entities are migrated, non_migrated_entity_names is empty and the
    # original-file module-alias processing block must be skipped without error.
    source = "_CONST = 42\n_CONST = 99\n\ndef fn_a():\n    return _CONST\n"
    e_block1 = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_block2 = Entity(EntityKind.TOP_LEVEL, "_block_2", 2, 2, ["_CONST"])
    e_fn = _make_entity("fn_a", 4, 5)
    c = _classified(entities=[e_block1, e_block2, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["_block_2", "fn_a"], target_file="funcs.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")
    # Does not abort or crash; original source may be minimal.
    assert not result.abort


def test_generate_reassigned_two_entities_same_file_single_module_import():
    # Two entities in the same new file both reference a reassigned constant.
    # The same "from . import constants" import must appear only once
    # (seen_top_cross deduplication).
    source = textwrap.dedent(
        """\
        _CONST = 42
        _CONST = 99

        def fn_a():
            return _CONST

        def fn_b():
            return _CONST
        """
    )
    e_block1 = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_block2 = Entity(EntityKind.TOP_LEVEL, "_block_2", 2, 2, ["_CONST"])
    e_fn_a = _make_entity("fn_a", 4, 5)
    e_fn_b = _make_entity("fn_b", 7, 8)
    c = _classified(entities=[e_block1, e_block2, e_fn_a, e_fn_b])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["fn_a", "fn_b"], target_file="funcs.py"),
            # _block_2 stays non-migrated → makes _CONST "reassigned"
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    funcs_src = result.new_files["funcs.py"]
    # The module import must appear exactly once despite two entities needing it.
    import_lines = [ln for ln in funcs_src.splitlines() if "import constants" in ln]
    assert len(import_lines) == 1


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


def test_generate_subdir_test_docstring_only_remaining_clears_original():
    # Regression: when a test-file subdir split migrates all entities and the
    # only thing left in the original is the module docstring (a TOP_LEVEL
    # entity that is not migrated by _remove_entity_lines), the docstring must
    # be routed to __init__.py and the original file must be cleared for
    # deletion by the engine.
    source = textwrap.dedent(
        """\
        \"\"\"Tests for the widget module.
        Covers edge cases.
        \"\"\"

        def test_alpha():
            pass

        def test_beta():
            pass
        """
    )
    # The module docstring is a TOP_LEVEL entity spanning lines 1-3.
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, [])
    e_alpha = _make_entity("test_alpha", 5, 6)
    e_beta = _make_entity("test_beta", 8, 9)
    c = _classified(entities=[e_block, e_alpha, e_beta])
    # Only the test functions are migrated; the TOP_LEVEL entity stays.
    plan = _plan(
        [
            GroupPlacement(group=["test_alpha"], target_file="widget/test_alpha.py"),
            GroupPlacement(group=["test_beta"], target_file="widget/test_beta.py"),
        ]
    )

    result = generate_file_splits(
        c, plan, source, "tests/test_widget.py", subdir_name="widget"
    )

    assert not result.abort
    # Docstring must end up in __init__.py.
    init_src = result.new_files["widget/__init__.py"]
    assert '"""Tests for the widget module.' in init_src
    # Original source must be empty so the engine deletes it.
    assert result.original_source == ""


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
