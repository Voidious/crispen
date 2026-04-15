from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    ImportInfo,
    _find_cross_file_imports,
    _find_cross_file_type_checking_imports,
    _find_needed_imports,
    _find_type_checking_needed_imports,
    _module_import_stmt,
    _relative_import_prefix,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .helpers import _classified, _plan


def test_find_needed_imports_referenced_name():
    # Entity references "os"; import for "os" should be included.
    entity_src_map = {"foo": "def foo():\n    os.getcwd()\n"}
    infos = [ImportInfo(names=["os"], source="import os", is_future=False)]
    result = _find_needed_imports(["foo"], entity_src_map, infos, {"foo"})
    assert "import os" in result


def test_find_needed_imports_unreferenced_name():
    # Entity doesn't reference "sys"; import should be excluded.
    entity_src_map = {"foo": "def foo():\n    pass\n"}
    infos = [ImportInfo(names=["sys"], source="import sys", is_future=False)]
    result = _find_needed_imports(["foo"], entity_src_map, infos, {"foo"})
    assert result == []


def test_find_needed_imports_future_always_included():
    # __future__ import is always included regardless of entity references.
    entity_src_map = {"foo": "def foo():\n    pass\n"}
    infos = [
        ImportInfo(
            names=["annotations"],
            source="from __future__ import annotations",
            is_future=True,
        )
    ]
    result = _find_needed_imports(["foo"], entity_src_map, infos, {"foo"})
    assert "from __future__ import annotations" in result


def test_find_needed_imports_deduplicates():
    # Two ImportInfo entries with the same source string → only one included.
    entity_src_map = {"foo": "def foo():\n    os.getcwd()\n"}
    infos = [
        ImportInfo(names=["os"], source="import os", is_future=False),
        ImportInfo(names=["os"], source="import os", is_future=False),  # duplicate
    ]
    result = _find_needed_imports(["foo"], entity_src_map, infos, {"foo"})
    assert result.count("import os") == 1


def test_find_needed_imports_entity_not_in_map():
    # Entity name not in entity_source_map → treated as empty source.
    infos = [ImportInfo(names=["os"], source="import os", is_future=False)]
    result = _find_needed_imports(["ghost"], {}, infos, set())
    assert result == []


def test_find_needed_imports_skips_type_checking():
    # is_type_checking imports must not appear as regular imports.
    entity_src_map = {"foo": 'def foo(x: "MyConfig") -> None:\n    pass\n'}
    infos = [
        ImportInfo(
            names=["MyConfig"],
            source="from .config import MyConfig",
            is_future=False,
            is_type_checking=True,
        )
    ]
    result = _find_needed_imports(["foo"], entity_src_map, infos, {"foo"})
    assert result == []


def test_find_type_checking_needed_imports_quoted_only():
    # "MyType" appears only in a quoted annotation, not a runtime load.
    entity_src_map = {"foo": 'def foo(x: Optional["MyType"]) -> None:\n    pass\n'}
    infos = [
        ImportInfo(
            names=["MyType"], source="from models import MyType", is_future=False
        )
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert "from models import MyType" in result


def test_find_type_checking_needed_imports_runtime_excluded():
    # When the name is used at runtime (not just in a quoted annotation),
    # it should NOT appear in the TYPE_CHECKING-only list.
    # annotation_only = quoted - runtime excludes runtime names directly.
    entity_src_map = {"foo": "def foo():\n    return MyType()\n"}
    infos = [
        ImportInfo(
            names=["MyType"], source="from models import MyType", is_future=False
        )
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert result == []


def test_find_type_checking_needed_imports_no_annotations():
    # No quoted annotations → result is empty.
    entity_src_map = {"foo": "def foo():\n    pass\n"}
    infos = [
        ImportInfo(
            names=["MyType"], source="from models import MyType", is_future=False
        )
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert result == []


def test_find_type_checking_needed_imports_future_excluded():
    # __future__ imports are never returned (they're always in regular imports).
    entity_src_map = {"foo": 'def foo(x: "MyType") -> None:\n    pass\n'}
    infos = [
        ImportInfo(
            names=["annotations"],
            source="from __future__ import annotations",
            is_future=True,
        ),
        ImportInfo(
            names=["MyType"], source="from models import MyType", is_future=False
        ),
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert "from __future__ import annotations" not in result
    assert "from models import MyType" in result


def test_find_type_checking_needed_imports_deduplicates():
    # Two ImportInfo entries with the same source → only one returned.
    entity_src_map = {"foo": 'def foo(x: "MyType") -> None:\n    pass\n'}
    infos = [
        ImportInfo(
            names=["MyType"], source="from models import MyType", is_future=False
        ),
        ImportInfo(
            names=["MyType"], source="from models import MyType", is_future=False
        ),
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert result.count("from models import MyType") == 1


def test_find_type_checking_needed_imports_import_names_no_match():
    # annotation_only has "MyType" but the ImportInfo names do not include it →
    # the tc_names check returns False → import is skipped.
    entity_src_map = {"foo": 'def foo(x: "MyType") -> None:\n    pass\n'}
    infos = [
        ImportInfo(
            names=["OtherType"], source="from models import OtherType", is_future=False
        )
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert result == []


def test_find_type_checking_needed_imports_partial_multi_name_import():
    # From a multi-name import, only the annotation-only name should appear in
    # the TYPE_CHECKING block; the other name (not referenced at all) must not.
    entity_src_map = {
        "foo": 'def foo(x: "MyResult") -> None:\n    pass\n',
    }
    infos = [
        ImportInfo(
            names=["MyResult", "run_thing"],
            source="from mymod import MyResult, run_thing",
            is_future=False,
        )
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert len(result) == 1
    assert "MyResult" in result[0]
    assert "run_thing" not in result[0]


def test_find_type_checking_needed_imports_narrowed_src_dedup():
    # When two ImportInfo entries produce the same narrowed source after
    # filtering, only one copy should appear in the result (line 535 branch).
    entity_src_map = {"foo": 'def foo(x: "MyResult") -> None:\n    pass\n'}
    infos = [
        ImportInfo(
            names=["MyResult", "run_thing"],
            source="from mymod import MyResult, run_thing",
            is_future=False,
        ),
        # A second entry with the same source (e.g. two entities requested it).
        ImportInfo(
            names=["MyResult", "run_thing"],
            source="from mymod import MyResult, run_thing",
            is_future=False,
        ),
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert result.count("from mymod import MyResult") == 1


def test_find_type_checking_needed_imports_shared_import_with_runtime_peer():
    # Regression: when an import line covers both a runtime name and an
    # annotation-only name, the annotation-only name must still get a
    # TYPE_CHECKING import even though the import source appears in the
    # regular imports (where _prune_unused_imports will later drop it).
    entity_src_map = {
        "foo": (
            'def foo(_acc: Optional["_LLMAccumulator"] = None) -> None:\n'
            "    call_with_tool(_PLACEMENT_TOOL)\n"
        )
    }
    infos = [
        ImportInfo(
            names=["_LLMAccumulator", "_PLACEMENT_TOOL"],
            source="from .llm_schemas import _LLMAccumulator, _PLACEMENT_TOOL",
            is_future=False,
        )
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    # _LLMAccumulator is only in a quoted annotation → must be in TC block
    assert any("_LLMAccumulator" in r for r in result)
    # _PLACEMENT_TOOL is a runtime reference → must NOT be in TC block
    assert not any("_PLACEMENT_TOOL" in r for r in result)


def test_find_type_checking_needed_imports_uses_is_type_checking_infos():
    # is_type_checking=True ImportInfo entries are used for TC distribution;
    # the function should return them for entities that use the name in a
    # quoted annotation.
    entity_src_map = {"foo": 'def foo(config: "MyConfig") -> None:\n    pass\n'}
    infos = [
        ImportInfo(
            names=["MyConfig"],
            source="from .config import MyConfig",
            is_future=False,
            is_type_checking=True,
        )
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert "from .config import MyConfig" in result


def test_find_cross_file_imports_basic():
    # fn_a references _MODEL which is defined in block_1.py
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "llm_extract.py"
    )
    assert from_imports == ["from .block_1 import _MODEL"]
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_same_file_excluded():
    # _MODEL goes to the same file as fn_a → no cross-file import needed
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "llm_extract.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "llm_extract.py"
    )
    assert from_imports == []
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_no_match():
    # Referenced name not in name_to_target_file → no cross-file import
    entity_source_map = {"fn_a": "def fn_a():\n    return os.getcwd()\n"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, {}, "utils.py"
    )
    assert from_imports == []
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_entity_not_in_map():
    # Entity name not in entity_source_map → treated as empty source, no imports
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["ghost"], {}, {"x": "other.py"}, "utils.py"
    )
    assert from_imports == []
    assert module_imports == []
    assert rewrites == {}


def test_relative_import_prefix_same_directory():
    # Both files at the root level → single dot.
    assert _relative_import_prefix("a.py", "b.py") == ".b"


def test_relative_import_prefix_sibling_subdir():
    # from_file is in sub/, to_file is in helpers/ → go up one, then down.
    assert _relative_import_prefix("sub/a.py", "helpers/b.py") == "..helpers.b"


def test_relative_import_prefix_same_subdir():
    # Both in the same subdirectory → single dot.
    assert _relative_import_prefix("sub/a.py", "sub/b.py") == ".b"


def test_relative_import_prefix_to_nested():
    # to_file is in a subdirectory of root while from_file is at root.
    assert _relative_import_prefix("a.py", "helpers/b.py") == ".helpers.b"


def test_relative_import_prefix_to_init_same_dir():
    # to_file is __init__.py in the same directory → "." (the package itself).
    assert _relative_import_prefix("a.py", "__init__.py") == "."


def test_relative_import_prefix_to_init_same_subdir():
    # Both in sub/, to_file is sub/__init__.py → "." (the package itself).
    assert _relative_import_prefix("sub/a.py", "sub/__init__.py") == "."


def test_find_cross_file_imports_cross_directory():
    # fn_a is in tests/test.py; helper is in helpers/entities.py.
    # Cross-directory import needs ".." to go up from tests/ to root.
    entity_source_map = {"fn_a": "def fn_a():\n    return _helper()\n"}
    name_to_target_file = {"_helper": "helpers/entities.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "tests/test.py"
    )
    assert from_imports == ["from ..helpers.entities import _helper"]
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_top_level_var_uses_module_import():
    # SAFE_MODE is a TOP_LEVEL variable in conversion.py; runtime.py references it.
    # Should produce a module-level import (from . import conversion) in
    # module_imports, not a direct name import, so that later mutations to the
    # variable propagate correctly.
    entity_source_map = {
        "create_lua_runtime": (
            "def create_lua_runtime(safe_mode=None):\n"
            "    if safe_mode is None:\n"
            "        safe_mode = SAFE_MODE\n"
        )
    }
    name_to_target_file = {"SAFE_MODE": "conversion.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["create_lua_runtime"],
        entity_source_map,
        name_to_target_file,
        "runtime.py",
        top_level_var_names={"SAFE_MODE"},
    )
    assert from_imports == []
    assert module_imports == ["from . import conversion"]
    assert rewrites == {"SAFE_MODE": "conversion.SAFE_MODE"}


def test_find_cross_file_imports_top_level_var_abs_pkg():
    # Same as above but with abs_pkg set (test-file context).
    # Uses "import pkg.module as local" syntax to avoid test-name misclassification.
    entity_source_map = {"fn_a": "def fn_a():\n    return SAFE_MODE\n"}
    name_to_target_file = {"SAFE_MODE": "conversion.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "test_fn.py",
        abs_pkg="mylib",
        top_level_var_names={"SAFE_MODE"},
    )
    assert from_imports == []
    assert module_imports == ["import mylib.conversion as conversion"]
    assert rewrites == {"SAFE_MODE": "conversion.SAFE_MODE"}


def test_find_cross_file_imports_top_level_var_abs_pkg_empty():
    # abs_pkg="" (root-level test) — no package prefix, plain "import conversion".
    entity_source_map = {"fn_a": "def fn_a():\n    return SAFE_MODE\n"}
    name_to_target_file = {"SAFE_MODE": "conversion.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "test_fn.py",
        abs_pkg="",
        top_level_var_names={"SAFE_MODE"},
    )
    assert from_imports == []
    assert module_imports == ["import conversion"]
    assert rewrites == {"SAFE_MODE": "conversion.SAFE_MODE"}


def test_find_cross_file_imports_top_level_var_cross_directory():
    # TOP_LEVEL var in sub/constants.py, referenced from runtime.py at root.
    entity_source_map = {"fn_a": "def fn_a():\n    return TIMEOUT\n"}
    name_to_target_file = {"TIMEOUT": "sub/constants.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "runtime.py",
        top_level_var_names={"TIMEOUT"},
    )
    assert from_imports == []
    assert module_imports == ["from .sub import constants"]
    assert rewrites == {"TIMEOUT": "constants.TIMEOUT"}


def test_find_cross_file_imports_mixed_top_level_and_function():
    # SAFE_MODE is a TOP_LEVEL var; _helper is a function — mixed case.
    entity_source_map = {
        "fn_a": ("def fn_a():\n" "    if SAFE_MODE:\n" "        return _helper()\n")
    }
    name_to_target_file = {"SAFE_MODE": "conversion.py", "_helper": "helpers.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "runtime.py",
        top_level_var_names={"SAFE_MODE"},
    )
    assert from_imports == ["from .helpers import _helper"]
    assert module_imports == ["from . import conversion"]
    assert rewrites == {"SAFE_MODE": "conversion.SAFE_MODE"}


def test_module_import_stmt_sibling_relative():
    stmt, local = _module_import_stmt("runtime.py", "conversion.py", abs_pkg=None)
    assert stmt == "from . import conversion"
    assert local == "conversion"


def test_module_import_stmt_cross_directory_relative():
    stmt, local = _module_import_stmt("runtime.py", "sub/constants.py", abs_pkg=None)
    assert stmt == "from .sub import constants"
    assert local == "constants"


def test_module_import_stmt_parent_directory_relative():
    # svc/test_fns.py importing from test_svc.py (parent dir)
    stmt, local = _module_import_stmt("svc/test_fns.py", "test_svc.py", abs_pkg=None)
    assert stmt == "from .. import test_svc"
    assert local == "test_svc"


def test_module_import_stmt_abs_pkg_with_prefix():
    # Uses "import pkg.module as local" to avoid test-name collision.
    stmt, local = _module_import_stmt("test_fn.py", "conversion.py", abs_pkg="mylib")
    assert stmt == "import mylib.conversion as conversion"
    assert local == "conversion"


def test_module_import_stmt_abs_pkg_empty():
    # No package prefix → plain "import conversion".
    stmt, local = _module_import_stmt("test_fn.py", "conversion.py", abs_pkg="")
    assert stmt == "import conversion"
    assert local == "conversion"


def test_module_import_stmt_abs_pkg_nested_module():
    # source_file has a nested path within the package
    stmt, local = _module_import_stmt("test_fn.py", "sub/constants.py", abs_pkg="mylib")
    assert stmt == "import mylib.sub.constants as constants"
    assert local == "constants"


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


def test_find_cross_file_type_checking_imports_basic():
    # _LLMAccumulator appears only in a quoted annotation in fn_a.
    # It lives in block_1.py — a TYPE_CHECKING import should be generated.
    entity_source_map = {
        "fn_a": 'def fn_a(x: Optional["_LLMAccumulator"]) -> None:\n    pass\n'
    }
    name_to_target_file = {"_LLMAccumulator": "block_1.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "placements.py"
    )
    assert result == ["from .block_1 import _LLMAccumulator"]


def test_find_cross_file_type_checking_imports_same_file_excluded():
    # _LLMAccumulator goes to the same target file — no import needed.
    entity_source_map = {
        "fn_a": 'def fn_a(x: Optional["_LLMAccumulator"]) -> None:\n    pass\n'
    }
    name_to_target_file = {"_LLMAccumulator": "placements.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "placements.py"
    )
    assert result == []


def test_find_cross_file_type_checking_imports_runtime_excluded():
    # _LLMAccumulator is used at runtime (not just annotation) — excluded.
    entity_source_map = {"fn_a": "def fn_a():\n    return _LLMAccumulator()\n"}
    name_to_target_file = {"_LLMAccumulator": "block_1.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "placements.py"
    )
    assert result == []


def test_find_cross_file_type_checking_imports_no_annotations():
    # No quoted annotations at all → empty result.
    entity_source_map = {"fn_a": "def fn_a():\n    pass\n"}
    name_to_target_file = {"_LLMAccumulator": "block_1.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "placements.py"
    )
    assert result == []


def test_find_cross_file_type_checking_imports_not_in_map():
    # Referenced quoted name not in name_to_target_file → no import.
    entity_source_map = {"fn_a": 'def fn_a(x: "UnknownType") -> None:\n    pass\n'}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, {}, "placements.py"
    )
    assert result == []


def test_find_cross_file_type_checking_imports_top_level_var_excluded():
    # A name in top_level_var_names is skipped (handled separately).
    entity_source_map = {
        "fn_a": 'def fn_a(x: Optional["SAFE_MODE"]) -> None:\n    pass\n'
    }
    name_to_target_file = {"SAFE_MODE": "constants.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "placements.py",
        top_level_var_names={"SAFE_MODE"},
    )
    assert result == []


def test_find_cross_file_type_checking_imports_abs_pkg():
    # With abs_pkg set, use absolute import style.
    entity_source_map = {
        "fn_a": 'def fn_a(x: Optional["_LLMAccumulator"]) -> None:\n    pass\n'
    }
    name_to_target_file = {"_LLMAccumulator": "block_1.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "test_fn.py",
        abs_pkg="tests",
    )
    assert result == ["from tests.block_1 import _LLMAccumulator"]


def test_find_cross_file_type_checking_imports_entity_not_in_map():
    # Entity not in entity_source_map → treated as empty, no imports.
    result = _find_cross_file_type_checking_imports(
        ["ghost"], {}, {"_X": "other.py"}, "placements.py"
    )
    assert result == []
