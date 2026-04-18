from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    _find_main_block_entity,
    _find_main_direct_callees,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from ..helpers import _classified, _make_entity, _plan


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
