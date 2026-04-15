from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import generate_file_splits
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .test_shared_helpers_extraction import _classified, _make_entity, _plan


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


def test_generate_migrated_top_level_import_names_not_in_cross_file_imports():
    # Regression: when a TOP_LEVEL entity containing "from dataclasses import
    # dataclass" is migrated, the name "dataclass" must NOT be added to the
    # name→target-file map.  A FUNCTION entity in a separate new file that also
    # uses dataclass should get "from dataclasses import dataclass" (via
    # _find_needed_imports) rather than "from .constants import dataclass" (a
    # spurious cross-file import that would fail at runtime because constants.py
    # never exports dataclass).
    source = (
        "from dataclasses import dataclass\n\n"
        "_CONST = 42\n\n"
        "def make():\n    return dataclass\n"
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["dataclass", "_CONST"])
    e_make = _make_entity("make", 5, 6)
    c = _classified(entities=[e_block, e_make])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["make"], target_file="utils.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    utils_src = result.new_files["utils.py"]
    # Must import dataclass from the stdlib, not from constants.py
    assert "from dataclasses import dataclass" in utils_src
    assert "from .constants import dataclass" not in utils_src


def test_generate_file_splits_removes_inline_redundant_imports():
    # When a split new file has both a top-level import and an inline re-import
    # of the same name, the inline one should be removed.
    source = textwrap.dedent(
        """\
        from mymod import Helper

        def test_uses_helper():
            from mymod import Helper
            assert Helper()
        """
    )
    entity = _make_entity("test_uses_helper", 3, 5)
    c = _classified(entities=[entity])
    plan = _plan(
        [GroupPlacement(group=["test_uses_helper"], target_file="test_split.py")]
    )
    result = generate_file_splits(c, plan, source, "big.py")
    new_src = result.new_files["test_split.py"]
    # The inline re-import should be removed; the module-level one covers it.
    assert new_src.count("from mymod import Helper") == 1


def test_generate_file_splits_subdir_injects_tc_import_for_nonmigrated_entity():
    # When a _block_N TOP_LEVEL entity that holds the `if TYPE_CHECKING:` block
    # is migrated to a sub-file, any non-migrated entity that references the
    # guarded name in a quoted annotation must receive the TYPE_CHECKING import
    # in the updated original (__init__.py).
    #
    # The original file has three entities:
    #   _block_1 — the TYPE_CHECKING block (migrated to sub.py)
    #   helper   — migrated to sub.py
    #   entry    — stays in __init__.py, references "MyConfig" in annotation
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .config import MyConfig\n"
        "\n"
        "def helper():\n"
        "    pass\n"
        "\n"
        "def entry(cfg: 'MyConfig') -> None:\n"
        "    helper()\n"
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, [])
    e_helper = _make_entity("helper", 5, 6)
    e_entry = _make_entity("entry", 8, 9)
    c = _classified(entities=[e_block, e_helper, e_entry])
    plan = _plan(
        [GroupPlacement(group=["_block_1", "helper"], target_file="pkg/sub.py")]
    )

    result = generate_file_splits(c, plan, source, "pkg.py", subdir_name="pkg")

    assert not result.abort
    init_src = result.original_source
    # The TYPE_CHECKING import must be injected and bumped for the new depth.
    assert "if TYPE_CHECKING:" in init_src
    assert "from ..config import MyConfig" in init_src
