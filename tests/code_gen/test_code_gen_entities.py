from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import _add_re_exports, generate_file_splits
from crispen.file_limiter.entity_parser import Entity, EntityKind


def _make_entity(name: str, start: int, end: int, defines=None) -> Entity:
    return Entity(EntityKind.FUNCTION, name, start, end, defines or [name])


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


def _plan(placements=None) -> FileLimiterPlan:
    return FileLimiterPlan(set3_migrate=[], placements=placements or [], abort=False)


def _abort_plan() -> FileLimiterPlan:
    return FileLimiterPlan(set3_migrate=[], placements=[], abort=True)


def test_add_re_exports_all_private_no_change():
    # Private name not called anywhere in remaining source → no import added.
    source = "import os\n\ndef _helper():\n    pass\n"
    entity = _make_entity("_helper", 3, 4)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"_helper": entity}, {})
    assert result == source


def test_add_re_exports_private_referenced_in_source():
    # Private name still called in remaining source → import is added.
    source = "import os\n\n_helper()\n"
    entity = _make_entity("_helper", 3, 3)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"_helper": entity}, {})
    assert "from .utils import _helper" in result


def test_add_re_exports_public_inserted_after_imports():
    source = "import os\n\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 3, 4)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {})
    assert "from .utils import foo" in result
    # Re-export line should come after "import os"
    lines = result.splitlines()
    import_idx = next(i for i, l in enumerate(lines) if "import os" in l)
    reexport_idx = next(i for i, l in enumerate(lines) if "from .utils import foo" in l)
    assert reexport_idx > import_idx


def test_add_re_exports_no_import_in_source():
    # No imports in source → re-export inserted at beginning.
    source = "\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 2, 3)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {})
    assert "from .utils import foo" in result


def test_add_re_exports_from_import_line():
    # "from pathlib import Path" should be detected as an import line.
    source = "from pathlib import Path\n\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 3, 4)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {})
    lines = result.splitlines()
    from_import_idx = next(
        i for i, l in enumerate(lines) if "from pathlib import Path" in l
    )
    reexport_idx = next(i for i, l in enumerate(lines) if "from .utils import foo" in l)
    assert reexport_idx > from_import_idx


def test_add_re_exports_multiple_targets_sorted():
    source = "import os\n"
    e1 = _make_entity("foo", 1, 2)
    e2 = _make_entity("bar", 3, 4)
    placements = [
        GroupPlacement(group=["foo"], target_file="b_module.py"),
        GroupPlacement(group=["bar"], target_file="a_module.py"),
    ]
    result = _add_re_exports(source, placements, {"foo": e1, "bar": e2}, {})
    # a_module comes before b_module (sorted)
    a_idx = result.index("a_module")
    b_idx = result.index("b_module")
    assert a_idx < b_idx


def test_add_re_exports_mixed_public_private():
    source = "import os\n"
    entity_map = {
        "pub": _make_entity("pub", 1, 2),
        "_priv": _make_entity("_priv", 3, 4),
    }
    placement = GroupPlacement(group=["pub", "_priv"], target_file="utils.py")
    result = _add_re_exports(source, [placement], entity_map, {})
    # Only "pub" in re-export, not "_priv"
    assert "pub" in result
    assert "_priv" not in result


def test_add_re_exports_test_function_not_re_exported():
    # test_ functions must never get a proxy import — pytest would discover and
    # run them twice (once from the original file, once from the new file).
    source = "import os\n"
    entity = _make_entity("test_something", 1, 3)
    placement = GroupPlacement(group=["test_something"], target_file="tests/helpers.py")
    result = _add_re_exports(source, [placement], {"test_something": entity}, {})
    assert result == source


def test_add_re_exports_test_function_re_exported_when_referenced():
    # If something in the remaining source actually calls test_something (unusual
    # but possible), the proxy import is still added.
    source = "import os\n\ntest_something()\n"
    entity = _make_entity("test_something", 1, 3)
    placement = GroupPlacement(group=["test_something"], target_file="tests/helpers.py")
    result = _add_re_exports(source, [placement], {"test_something": entity}, {})
    assert "from .tests.helpers import test_something" in result


def test_add_re_exports_top_level_block_private_names_referenced():
    # TOP_LEVEL block entity name (_block_1) differs from its defined names.
    # Both defined names are still loaded in remaining source → re-imported.
    source = "import os\n\n_DUP_SOURCE\n_DUP_RANGES\n"
    entity = Entity(
        EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_DUP_SOURCE", "_DUP_RANGES"]
    )
    placement = GroupPlacement(group=["_block_1"], target_file="test_helpers.py")
    result = _add_re_exports(source, [placement], {"_block_1": entity}, {})
    assert "from .test_helpers import _DUP_RANGES, _DUP_SOURCE" in result


def test_add_re_exports_entity_not_in_map_falls_back_to_entity_name():
    # Entity name in group is missing from entity_map → falls back to entity name.
    source = "import os\n\nghost()\n"  # 'ghost' is still referenced
    placement = GroupPlacement(group=["ghost"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {}, {})
    assert "from .utils import ghost" in result


def test_add_re_exports_top_level_block_private_names_not_referenced():
    # TOP_LEVEL block entity whose defined name is private and not used → no import.
    source = "import os\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    placement = GroupPlacement(group=["_block_1"], target_file="constants.py")
    result = _add_re_exports(source, [placement], {"_block_1": entity}, {})
    assert result == source


def test_add_re_exports_indented_local_import_not_treated_as_last_import():
    # Functions with local (indented) imports must not cause re-exports to be
    # inserted inside the function body.  The re-export should appear after the
    # top-level "import os" line, not after the indented "from x import y".
    source = (
        "import os\n"
        "\n"
        "def foo():\n"
        "    from unittest.mock import MagicMock\n"
        "    MagicMock()\n"
        "\n"
        "def bar():\n"
        "    pass\n"
    )
    entity = _make_entity("baz", 7, 8)
    placement = GroupPlacement(group=["baz"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"baz": entity}, {})
    # Re-export must appear immediately after "import os", not inside foo().
    lines = result.splitlines()
    os_idx = next(i for i, l in enumerate(lines) if l == "import os")
    reexport_idx = next(i for i, l in enumerate(lines) if "from .utils import baz" in l)
    assert reexport_idx == os_idx + 1
    # The function body must remain intact (local import line must still be there).
    assert "    from unittest.mock import MagicMock" in result


def test_add_re_exports_syntax_error_returns_source_unchanged():
    # If the source has a SyntaxError, _add_re_exports cannot determine where
    # to insert re-exports and must return the source unchanged.
    source = "import os\ndef (invalid\n"
    entity = _make_entity("baz", 1, 1)
    placement = GroupPlacement(group=["baz"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"baz": entity}, {})
    assert result == source


def test_add_re_exports_abs_pkg_package_prefix():
    # abs_pkg="tests" → absolute import: "from tests.utils import foo"
    source = "import os\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {}, abs_pkg="tests")
    assert "from tests.utils import foo" in result
    assert "from .utils import foo" not in result


def test_add_re_exports_abs_pkg_root_level():
    # abs_pkg="" → root-level absolute import: "from utils import foo"
    source = "import os\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {}, abs_pkg="")
    assert "from utils import foo" in result
    assert "from .utils import foo" not in result


def test_generate_abort_plan():
    plan = _abort_plan()
    c = _classified()
    result = generate_file_splits(c, plan, "def foo():\n    pass\n", "big.py")
    assert result.abort is True
    assert result.new_files == {}
    assert result.original_source == "def foo():\n    pass\n"


def test_generate_empty_placements():
    plan = _plan()  # placements=[]
    c = _classified()
    source = "def foo():\n    pass\n"
    result = generate_file_splits(c, plan, source, "big.py")
    assert result.abort is False
    assert result.new_files == {}
    assert result.original_source == source


def test_generate_single_entity_migration():
    source = "import os\n\ndef foo():\n    os.getcwd()\n"
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert result.abort is False
    assert "utils.py" in result.new_files
    new_src = result.new_files["utils.py"]
    assert "import os" in new_src
    assert "def foo():" in new_src
    # Original should not have foo's def anymore
    assert "def foo():" not in result.original_source
    # But should have a re-export
    assert "from .utils import foo" in result.original_source


def test_generate_private_entity_no_reexport():
    source = "def _helper():\n    pass\n"
    entity = _make_entity("_helper", 1, 2)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["_helper"], target_file="private.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "from .private import" not in result.original_source


def test_generate_entity_not_in_source_map():
    # Group has entity name not in classified.entities → entity skipped in new file.
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    c = _classified(entities=[entity])
    # "ghost" is in the group but has no matching entity
    plan = _plan([GroupPlacement(group=["foo", "ghost"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "utils.py" in result.new_files
    # "ghost" produces no source so only "foo" appears
    new_src = result.new_files["utils.py"]
    assert "def foo():" in new_src


def test_generate_no_imports_needed():
    # Entity uses no imports → no import section in new file.
    source = "def add(a, b):\n    return a + b\n"
    entity = _make_entity("add", 1, 2)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["add"], target_file="math_utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["math_utils.py"]
    # No "import" prefix expected
    assert not new_src.startswith("import")
    assert "def add" in new_src


def test_generate_multiple_groups_same_file():
    source = textwrap.dedent(
        """\
        import os

        def foo():
            pass

        def bar():
            pass
        """
    )
    e_foo = _make_entity("foo", 3, 4)
    e_bar = _make_entity("bar", 6, 7)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan(
        [
            GroupPlacement(group=["foo"], target_file="utils.py"),
            GroupPlacement(group=["bar"], target_file="utils.py"),
        ]
    )
    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert "def foo():" in new_src
    assert "def bar():" in new_src


def test_generate_multiple_different_target_files():
    source = "def foo():\n    pass\n\ndef bar():\n    pass\n"
    e_foo = _make_entity("foo", 1, 2)
    e_bar = _make_entity("bar", 4, 5)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan(
        [
            GroupPlacement(group=["foo"], target_file="foo_module.py"),
            GroupPlacement(group=["bar"], target_file="bar_module.py"),
        ]
    )
    result = generate_file_splits(c, plan, source, "big.py")

    assert "foo_module.py" in result.new_files
    assert "bar_module.py" in result.new_files
    assert "def foo():" in result.new_files["foo_module.py"]
    assert "def bar():" in result.new_files["bar_module.py"]
    assert "from .bar_module import bar" in result.original_source
    assert "from .foo_module import foo" in result.original_source


def test_generate_future_import_not_duplicated_when_in_entity_source():
    # Entity source itself contains `from __future__ import annotations`
    # (e.g. the _block_1 TOP_LEVEL entity which IS the file's import block).
    # It must appear only once at the top of the new file, not again inside
    # the entity source, which would cause a SyntaxError.
    source = textwrap.dedent(
        """\
        from __future__ import annotations

        \"\"\"Module docstring.\"\"\"

        from __future__ import annotations

        import os

        _CONST = 42
    """
    )
    # _block_1 spans the whole file and contains the future import + constants.
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 9, ["_CONST"])
    c = _classified(entities=[e_block])
    plan = _plan([GroupPlacement(group=["_block_1"], target_file="constants.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["constants.py"]
    assert new_src.count("from __future__ import annotations") == 1
    # Must be at the very start of the file (before any other code).
    first_non_blank = next(line for line in new_src.splitlines() if line.strip())
    assert first_non_blank == "from __future__ import annotations"


def test_generate_future_import_always_included():
    source = "from __future__ import annotations\n\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert "from __future__ import annotations" in new_src


def test_generate_no_circular_import_when_helper_referenced_by_migrated():
    # Integration test: _run stays in original and is used by test_fn (migrated).
    # Without the fix: original → helpers.py (re-export) and helpers.py → original.
    # With the fix: _run is moved into helpers.py; original imports _run from helpers.
    source = textwrap.dedent(
        """\
        def _run(x):
            return x

        def test_fn(tmp_path):
            return _run(tmp_path)
    """
    )
    e_run = _make_entity("_run", 1, 2)
    e_test = _make_entity("test_fn", 4, 5)
    c = _classified(entities=[e_run, e_test])
    plan = _plan([GroupPlacement(group=["test_fn"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "original.py")

    helpers_src = result.new_files["helpers.py"]
    # _run is defined in helpers.py (extracted), not imported from original
    assert "def _run" in helpers_src
    assert "from .original import _run" not in helpers_src
    # original re-imports _run from helpers.py (since it's still used there via
    # non-migrated code — but in this minimal example there's nothing left)
    # At minimum, no circular self-import exists
    assert "from .original import" not in helpers_src
