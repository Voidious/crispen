from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import generate_file_splits
from crispen.file_limiter.entity_parser import Entity, EntityKind
from ..helpers import _classified, _make_entity, _plan


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


def test_generate_top_level_entity_imports_not_duplicated():
    # When a TOP_LEVEL entity source contains regular imports (e.g. `import os`)
    # those must NOT appear twice in the generated file: once from
    # _find_needed_imports and again from the entity source itself.
    source = "import os\n\n_CONST = os.sep\n\ndef foo():\n    return os.getcwd()\n"
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["os", "_CONST"])
    e_foo = _make_entity("foo", 5, 6)
    c = _classified(entities=[e_block, e_foo])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1", "foo"], target_file="utils.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert new_src.count("import os") == 1
