from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    _find_main_block_entity,
    _find_main_direct_callees,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from ..test_generate_file_splits_core import _classified, _make_entity, _plan


def test_generate_file_splits_type_checking_for_quoted_annotation():
    # _advise_set3 uses Optional["_LLMAccumulator"] (quoted annotation).
    # _LLMAccumulator is migrated to models.py; _advise_set3 goes to placements.py.
    # placements.py must get:
    #   from typing import TYPE_CHECKING
    #   if TYPE_CHECKING:
    #       from .models import _LLMAccumulator
    source = textwrap.dedent(
        """\
        from typing import Optional

        class _LLMAccumulator:
            pass

        def _advise_set3(acc: Optional["_LLMAccumulator"]) -> None:
            pass
        """
    )
    e_acc = Entity(EntityKind.CLASS, "_LLMAccumulator", 3, 4, ["_LLMAccumulator"])
    e_fn = Entity(EntityKind.FUNCTION, "_advise_set3", 6, 7, ["_advise_set3"])
    c = _classified(entities=[e_acc, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_LLMAccumulator"], target_file="models.py"),
            GroupPlacement(group=["_advise_set3"], target_file="placements.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "advisor.py")

    placements_src = result.new_files["placements.py"]
    # TYPE_CHECKING may be merged into an existing "from typing import ..." line.
    assert "TYPE_CHECKING" in placements_src
    assert "if TYPE_CHECKING:" in placements_src
    assert "from .models import _LLMAccumulator" in placements_src


def test_generate_file_splits_type_checking_deduplication():
    # Two functions in the same target file both reference "_LLMAccumulator"
    # in quoted annotations.  The TYPE_CHECKING import should appear only once
    # even though both entities trigger _find_cross_file_type_checking_imports.
    source = textwrap.dedent(
        """\
        from typing import Optional

        class _LLMAccumulator:
            pass

        def _fn_a(x: Optional["_LLMAccumulator"]) -> None:
            pass

        def _fn_b(y: Optional["_LLMAccumulator"]) -> None:
            pass
        """
    )
    e_acc = Entity(EntityKind.CLASS, "_LLMAccumulator", 3, 4, ["_LLMAccumulator"])
    e_fna = Entity(EntityKind.FUNCTION, "_fn_a", 6, 7, ["_fn_a"])
    e_fnb = Entity(EntityKind.FUNCTION, "_fn_b", 9, 10, ["_fn_b"])
    c = _classified(entities=[e_acc, e_fna, e_fnb])
    plan = _plan(
        [
            GroupPlacement(group=["_LLMAccumulator"], target_file="models.py"),
            GroupPlacement(group=["_fn_a", "_fn_b"], target_file="placements.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "advisor.py")

    placements_src = result.new_files["placements.py"]
    assert placements_src.count("from .models import _LLMAccumulator") == 1


def test_generate_file_splits_tc_dedup_drops_when_already_in_regular():
    # Entity A uses _Acc at runtime (unquoted annotation → regular cross-file import).
    # Entity B uses _Acc only in a quoted annotation → would normally get a TC import.
    # Both go to workers.py.  The dedup step must remove the TC import entirely since
    # _Acc is already covered by the regular import.
    source = textwrap.dedent(
        """\
        from typing import Optional

        class _Acc:
            pass

        def fn_runtime(x) -> None:
            a: _Acc = x

        def fn_quoted(x: Optional["_Acc"]) -> None:
            pass
        """
    )
    e_acc = Entity(EntityKind.CLASS, "_Acc", 3, 4, ["_Acc"])
    e_rt = Entity(EntityKind.FUNCTION, "fn_runtime", 6, 7, ["fn_runtime"])
    e_qt = Entity(EntityKind.FUNCTION, "fn_quoted", 9, 10, ["fn_quoted"])
    c = _classified(entities=[e_acc, e_rt, e_qt])
    plan = _plan(
        [
            GroupPlacement(group=["_Acc"], target_file="models.py"),
            GroupPlacement(group=["fn_runtime", "fn_quoted"], target_file="workers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "advisor.py")

    workers_src = result.new_files["workers.py"]
    # Regular import must be present, TYPE_CHECKING block must NOT be.
    assert "from .models import _Acc" in workers_src
    assert "if TYPE_CHECKING:" not in workers_src


def test_generate_file_splits_tc_dedup_plain_import_branches():
    # Covers the non-from-import branches in the dedup loop:
    #   • "import sys" in needed → _FROM_IMPORT_RE does not match (2633->2631 branch)
    #   • "import typing_extensions" in needed_tc (annotation-only) → TC import is a
    #     plain import statement, not a from-import (2655 branch)
    source = textwrap.dedent(
        """\
        import sys
        import typing_extensions
        from typing import Optional

        class _Acc:
            pass

        def fn(x: Optional["_Acc"]) -> None:
            sys.exit(0)

        def fn2() -> "typing_extensions.Literal":
            pass
        """
    )
    e_acc = Entity(EntityKind.CLASS, "_Acc", 5, 6, ["_Acc"])
    e_fn = Entity(EntityKind.FUNCTION, "fn", 8, 9, ["fn"])
    e_fn2 = Entity(EntityKind.FUNCTION, "fn2", 11, 12, ["fn2"])
    c = _classified(entities=[e_acc, e_fn, e_fn2])
    plan = _plan(
        [
            GroupPlacement(group=["_Acc"], target_file="models.py"),
            GroupPlacement(group=["fn", "fn2"], target_file="workers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "advisor.py")

    workers_src = result.new_files["workers.py"]
    # TC import for _Acc (cross-file, quoted annotation) must still be present.
    assert "if TYPE_CHECKING:" in workers_src
    assert "_Acc" in workers_src
    # Plain import for typing_extensions preserved in TC block.
    assert "typing_extensions" in workers_src


def test_generate_prunes_unused_names_from_multiname_import():
    # foo uses only List, not Dict; the new file's import should be narrowed.
    source = "from typing import Dict, List\n\ndef foo(x: List):\n    return x\n"
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert "from typing import List" in new_src
    assert "Dict" not in new_src


def test_generate_prunes_fully_unused_import_from_original():
    # import os is only used by foo; after foo migrates the original no longer
    # needs os, so the import should be removed.
    source = "import os\n\ndef foo():\n    os.getcwd()\n\ndef bar():\n    return 1\n"
    e_foo = _make_entity("foo", 3, 4)
    e_bar = _make_entity("bar", 6, 7)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "from .utils import foo" in result.original_source
    assert "import os" not in result.original_source
    assert "def bar():" in result.original_source


def test_generate_narrows_partial_unused_import_in_original():
    # foo uses Dict; bar uses List.  After foo migrates, Dict should be
    # stripped from the original's import while List is kept.
    source = (
        "from typing import Dict, List\n\n"
        "def foo(x: Dict):\n    return x\n\n"
        "def bar(x: List):\n    return x\n"
    )
    e_foo = _make_entity("foo", 3, 4)
    e_bar = _make_entity("bar", 6, 7)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "from typing import List" in result.original_source
    assert "Dict" not in result.original_source


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
