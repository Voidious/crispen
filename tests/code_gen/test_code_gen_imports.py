from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    ImportInfo,
    _add_re_exports,
    _bump_relative_imports,
    _collect_name_loads,
    _extract_import_info,
    _find_cross_file_imports,
    _find_needed_imports,
    _import_derived_names,
    _import_line_numbers,
    _prune_inline_redundant_imports,
    _prune_unused_imports,
    _relative_import_prefix,
    _remove_entity_lines,
    _strip_top_level_import_lines,
    _target_module_name,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .test_code_gen_entities import _classified, _make_entity, _plan


def test_collect_name_loads_basic():
    source = "x = foo + bar"
    names = _collect_name_loads(source)
    assert "foo" in names
    assert "bar" in names


def test_collect_name_loads_store_not_included():
    source = "x = 1"
    names = _collect_name_loads(source)
    # x is a Store, not a Load
    assert "x" not in names


def test_collect_name_loads_syntax_error():
    assert _collect_name_loads("def (invalid") == set()


def test_extract_import_info_syntax_error():
    assert _extract_import_info("def (invalid") == []


def test_extract_import_info_plain_import():
    infos = _extract_import_info("import os\n")
    assert len(infos) == 1
    assert "os" in infos[0].names
    assert infos[0].is_future is False


def test_extract_import_info_import_with_asname():
    infos = _extract_import_info("import os as operating_system\n")
    assert infos[0].names == ["operating_system"]


def test_extract_import_info_dotted_import():
    infos = _extract_import_info("import os.path\n")
    assert infos[0].names == ["os"]


def test_extract_import_info_from_import():
    infos = _extract_import_info("from pathlib import Path\n")
    assert "Path" in infos[0].names
    assert infos[0].is_future is False


def test_extract_import_info_from_import_with_asname():
    infos = _extract_import_info("from pathlib import Path as P\n")
    assert infos[0].names == ["P"]


def test_extract_import_info_future_import():
    infos = _extract_import_info("from __future__ import annotations\n")
    assert infos[0].is_future is True
    assert "annotations" in infos[0].names


def test_extract_import_info_skips_non_imports():
    infos = _extract_import_info("def foo():\n    pass\n")
    assert infos == []


def test_extract_import_info_multiple():
    source = "import os\nfrom pathlib import Path\n"
    infos = _extract_import_info(source)
    assert len(infos) == 2


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


def test_target_module_name_simple():
    assert _target_module_name("utils.py") == "utils"


def test_target_module_name_nested():
    assert _target_module_name("helpers/io.py") == "helpers.io"


def test_remove_entity_lines_removes_range():
    source = "line1\nline2\nline3\nline4\n"
    entity = _make_entity("foo", 2, 3)
    entity_map = {"foo": entity}
    result = _remove_entity_lines(source, {"foo"}, entity_map, {})
    assert "line1" in result
    assert "line2" not in result
    assert "line3" not in result
    assert "line4" in result


def test_remove_entity_lines_name_not_in_map():
    # Name not in entity_map → nothing removed.
    source = "line1\nline2\n"
    result = _remove_entity_lines(source, {"ghost"}, {}, {})
    assert result == source


def test_remove_entity_lines_top_level_preserves_import_lines():
    # When a TOP_LEVEL entity containing both imports and assignments is
    # migrated, the import lines must be kept in the original file so that
    # the remaining functions still have access to those names.
    source = "import os\n_CONST = 1\n\ndef foo():\n    return os.getcwd()\n"
    entity_src = "import os\n_CONST = 1\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 2, ["os", "_CONST"])
    entity_map = {"_block_1": entity}
    entity_source_map = {"_block_1": entity_src}
    result = _remove_entity_lines(source, {"_block_1"}, entity_map, entity_source_map)
    assert "import os" in result  # import line preserved
    assert "_CONST" not in result  # assignment line removed
    assert "def foo():" in result  # function untouched


def test_remove_entity_lines_top_level_no_source_map_removes_all():
    # Empty entity_source_map → no imports can be identified, all lines removed.
    source = "import os\n_CONST = 1\n\ndef foo():\n    pass\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 2, ["os", "_CONST"])
    entity_map = {"_block_1": entity}
    result = _remove_entity_lines(source, {"_block_1"}, entity_map, {})
    assert "import os" not in result
    assert "_CONST" not in result


def test_import_derived_names_plain_import():
    src = "import os\nimport sys\n"
    assert _import_derived_names(src) == {"os", "sys"}


def test_import_derived_names_from_import():
    src = "from typing import Dict, List\n"
    assert _import_derived_names(src) == {"Dict", "List"}


def test_import_derived_names_aliased():
    src = "import libcst as cst\nfrom dataclasses import dataclass\n"
    assert _import_derived_names(src) == {"cst", "dataclass"}


def test_import_derived_names_ignores_assignments():
    src = "_MODEL = 'x'\n_MIN = 3\n"
    assert _import_derived_names(src) == set()


def test_import_derived_names_syntax_error():
    assert _import_derived_names("def (\n") == set()


def test_import_line_numbers_basic():
    src = "import os\n_CONST = 1\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 5, 6, [])
    # Entity starts at line 5; "import os" is relative line 1 → absolute line 5.
    result = _import_line_numbers(entity, src)
    assert result == {5}


def test_import_line_numbers_no_imports():
    src = "_CONST = 1\n_OTHER = 2\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 2, [])
    assert _import_line_numbers(entity, src) == set()


def test_import_line_numbers_syntax_error():
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, [])
    assert _import_line_numbers(entity, "def (\n") == set()


def test_find_cross_file_imports_basic():
    # fn_a references _MODEL which is defined in block_1.py
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "llm_extract.py"
    )
    assert result == ["from .block_1 import _MODEL"]


def test_find_cross_file_imports_same_file_excluded():
    # _MODEL goes to the same file as fn_a → no cross-file import needed
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "llm_extract.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "llm_extract.py"
    )
    assert result == []


def test_find_cross_file_imports_no_match():
    # Referenced name not in name_to_target_file → no cross-file import
    entity_source_map = {"fn_a": "def fn_a():\n    return os.getcwd()\n"}
    result = _find_cross_file_imports(["fn_a"], entity_source_map, {}, "utils.py")
    assert result == []


def test_find_cross_file_imports_entity_not_in_map():
    # Entity name not in entity_source_map → treated as empty source, no imports
    result = _find_cross_file_imports(["ghost"], {}, {"x": "other.py"}, "utils.py")
    assert result == []


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


def test_find_cross_file_imports_cross_directory():
    # fn_a is in tests/test.py; helper is in helpers/entities.py.
    # Cross-directory import needs ".." to go up from tests/ to root.
    entity_source_map = {"fn_a": "def fn_a():\n    return _helper()\n"}
    name_to_target_file = {"_helper": "helpers/entities.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "tests/test.py"
    )
    assert result == ["from ..helpers.entities import _helper"]


def test_prune_unused_imports_syntax_error():
    # Unparseable source → returned unchanged.
    source = "def (invalid syntax"
    assert _prune_unused_imports(source) == source


def test_prune_unused_imports_no_replacements_needed():
    # All imports are fully used → fast-path returns source unchanged.
    source = "import os\n\ndef f():\n    os.getcwd()\n"
    assert _prune_unused_imports(source) == source


def test_prune_unused_imports_preserves_future_import():
    # __future__ imports are always kept, even when the name isn't referenced.
    source = "from __future__ import annotations\n\ndef f():\n    pass\n"
    result = _prune_unused_imports(source)
    assert "from __future__ import annotations" in result


def test_prune_unused_imports_preserves_star_import():
    # Star imports cannot be pruned — kept as-is.
    source = "from os.path import *\n\ndef f():\n    pass\n"
    result = _prune_unused_imports(source)
    assert "from os.path import *" in result


def test_prune_unused_imports_removes_fully_unused_plain_import():
    # import whose name is never referenced is dropped entirely.
    source = "import sys\n\ndef f():\n    pass\n"
    result = _prune_unused_imports(source)
    assert "import sys" not in result


def test_prune_unused_imports_removes_fully_unused_from_import():
    # from-import whose names are never referenced is dropped entirely.
    source = "from typing import Dict\n\ndef f():\n    return 1\n"
    result = _prune_unused_imports(source)
    assert "from typing import" not in result


def test_prune_unused_imports_narrows_partial_from_import():
    # Only List is used — import narrowed to just List.
    source = "from typing import Dict, List\n\ndef f(x: List):\n    return x\n"
    result = _prune_unused_imports(source)
    assert "from typing import List" in result
    assert "Dict" not in result


def test_prune_unused_imports_narrows_plain_import():
    # import x, y where only y is used → narrowed to import y.
    source = "import os, sys\n\ndef f():\n    sys.exit()\n"
    result = _prune_unused_imports(source)
    assert "import sys" in result
    assert "os" not in result


def test_prune_unused_imports_multiline_import_collapsed():
    # Multi-line parenthesised import is collapsed to a single line.
    source = textwrap.dedent(
        """\
        from typing import (
            Dict,
            List,
        )

        def f(x: List):
            return x
        """
    )
    result = _prune_unused_imports(source)
    assert "from typing import List" in result
    assert "Dict" not in result
    assert "(\n" not in result


def test_prune_unused_imports_relative_import_narrowed():
    # Relative from-import is reconstructed with dots preserved.
    source = "from .utils import foo, bar\n\ndef f():\n    return foo()\n"
    result = _prune_unused_imports(source)
    assert "from .utils import foo" in result
    assert "bar" not in result


def test_add_re_exports_private_in_external_loads():
    # Private name not referenced in remaining source but present in external_loads
    # → re-export proxy IS added so the external caller continues to work.
    source = "import os\n"
    entity = _make_entity("_helper", 1, 2)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(
        source, [placement], {"_helper": entity}, {}, external_loads={"_helper"}
    )
    assert "from .utils import _helper" in result


def test_add_re_exports_test_function_in_external_loads_not_re_exported():
    # test_ functions must never get a proxy even when listed in external_loads,
    # because pytest would discover and run them twice.
    source = "import os\n"
    entity = _make_entity("test_something", 1, 2)
    placement = GroupPlacement(group=["test_something"], target_file="helpers.py")
    result = _add_re_exports(
        source,
        [placement],
        {"test_something": entity},
        {},
        external_loads={"test_something"},
    )
    assert result == source


def test_generate_private_entity_reexported_when_external_caller(tmp_path):
    # Private entity is re-exported when an external file imports it.
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    mod = pkg / "big.py"
    mod.write_text("def _helper():\n    pass\n")
    caller = tmp_path / "tests" / "test_big.py"
    caller.parent.mkdir()
    caller.write_text("from mypkg.big import _helper\n")

    source = "def _helper():\n    pass\n"
    entity = _make_entity("_helper", 1, 2)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["_helper"], target_file="private.py")])

    result = generate_file_splits(c, plan, source, str(mod))

    assert "from .private import _helper" in result.original_source


def test_add_re_exports_private_external_only_gets_noqa():
    # Private name in external_loads but NOT in remaining source → fmt: skip # noqa comment.
    source = "import os\n"
    entity = _make_entity("_helper", 1, 2)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(
        source, [placement], {"_helper": entity}, {}, external_loads={"_helper"}
    )
    assert "from .utils import _helper  # fmt: skip # noqa: F401, E501" in result


def test_add_re_exports_private_in_still_loaded_no_noqa():
    # Private name referenced in remaining source → re-export without noqa.
    source = "import os\n\n_helper()\n"
    entity = _make_entity("_helper", 3, 3)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"_helper": entity}, {})
    assert "from .utils import _helper\n" in result
    assert "# noqa" not in result


def test_add_re_exports_public_not_in_still_loaded_gets_noqa():
    # Public name migrated but not referenced in remaining source → fmt: skip # noqa.
    source = "import os\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {})
    assert "from .utils import foo  # fmt: skip # noqa: F401, E501" in result


def test_add_re_exports_public_in_still_loaded_no_noqa():
    # Public name still referenced in remaining source → re-export without noqa.
    source = "import os\n\nfoo()\n"
    entity = _make_entity("foo", 3, 3)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {})
    assert "from .utils import foo\n" in result
    assert "# noqa" not in result


def test_add_re_exports_multiple_noqa_each_on_own_line():
    # Two names both need noqa → one import line each so Black can't break the comment.
    source = "import os\n"
    entity = _make_entity("_block", 3, 4, ["_a", "_b"])
    placement = GroupPlacement(group=["_block"], target_file="utils.py")
    result = _add_re_exports(
        source,
        [placement],
        {"_block": entity},
        {},
        external_loads={"_a", "_b"},
    )
    lines = result.splitlines()
    noqa_lines = [line for line in lines if "# fmt: skip # noqa: F401, E501" in line]
    assert len(noqa_lines) == 2
    names = {line.split("import")[1].split("#")[0].strip() for line in noqa_lines}
    assert names == {"_a", "_b"}


def test_add_re_exports_mixed_splits_into_two_lines():
    # One entity defines two names: one in still_loaded, one purely re-exported.
    # They must appear on separate lines so noqa doesn't suppress used-name warnings.
    source = "import os\n\n_used()\n"
    entity = _make_entity("_block", 3, 4, ["_used", "_reexport"])
    placement = GroupPlacement(group=["_block"], target_file="utils.py")
    result = _add_re_exports(
        source,
        [placement],
        {"_block": entity},
        {},
        external_loads={"_used", "_reexport"},
    )
    lines = result.splitlines()
    noqa_lines = [line for line in lines if "# fmt: skip # noqa: F401, E501" in line]
    plain_lines = [
        line
        for line in lines
        if "from .utils import" in line and "# fmt: skip" not in line
    ]
    assert len(noqa_lines) == 1
    assert "_reexport" in noqa_lines[0]
    assert "_used" not in noqa_lines[0]
    assert len(plain_lines) == 1
    assert "_used" in plain_lines[0]


def test_prune_inline_syntax_error():
    # Unparseable source → returned unchanged.
    source = "def (invalid syntax"
    assert _prune_inline_redundant_imports(source) == source


def test_prune_inline_no_top_level_imports():
    # No module-level imports → nothing can be redundant, return unchanged.
    source = "def f():\n    from os import path\n    path.join('a', 'b')\n"
    assert _prune_inline_redundant_imports(source) == source


def test_prune_inline_no_inner_imports():
    # Only top-level imports, no function-body imports → unchanged.
    source = "import os\n\ndef f():\n    return os.getcwd()\n"
    assert _prune_inline_redundant_imports(source) == source


def test_prune_inline_no_redundancy():
    # Inner import brings in a different name than the top-level import.
    source = "import os\n\ndef f():\n    from sys import argv\n    return argv\n"
    assert _prune_inline_redundant_imports(source) == source


def test_prune_inline_removes_fully_redundant_from_import():
    # Top-level import covers all names in the inner from-import → remove it.
    source = textwrap.dedent(
        """\
        from unittest.mock import patch
        from mymod import Foo

        def test_thing():
            from mymod import Foo
            assert Foo()
        """
    )
    result = _prune_inline_redundant_imports(source)
    assert result.count("from mymod import Foo") == 1
    assert "assert Foo()" in result


def test_prune_inline_narrows_partially_redundant_from_import():
    # Only one of two inner names is already at top level → narrow the inner import.
    source = textwrap.dedent(
        """\
        from mymod import Foo

        def test_thing():
            from mymod import Foo, Bar
            assert Foo() and Bar()
        """
    )
    result = _prune_inline_redundant_imports(source)
    lines = result.splitlines()
    inner = [ln for ln in lines if "from mymod import" in ln and ln.startswith("    ")]
    assert len(inner) == 1
    assert "Bar" in inner[0]
    assert "Foo" not in inner[0]


def test_prune_inline_removes_fully_redundant_plain_import():
    # Inner ``import x`` where x is already available at top level → removed.
    source = textwrap.dedent(
        """\
        import os

        def f():
            import os
            return os.getcwd()
        """
    )
    result = _prune_inline_redundant_imports(source)
    assert result.count("import os") == 1


def test_prune_inline_narrows_partially_redundant_plain_import():
    # ``import os, sys`` inside function where os is already top-level → narrows to sys.
    source = textwrap.dedent(
        """\
        import os

        def f():
            import os, sys
            return sys.argv
        """
    )
    result = _prune_inline_redundant_imports(source)
    inner = [
        ln
        for ln in result.splitlines()
        if ln.strip().startswith("import") and ln.startswith("    ")
    ]
    assert len(inner) == 1
    assert "sys" in inner[0]
    assert "os" not in inner[0]


def test_prune_inline_preserves_indentation():
    # The narrowed replacement line must preserve the original indentation.
    source = textwrap.dedent(
        """\
        from mymod import Foo

        def test_thing():
            if True:
                from mymod import Foo, Bar
                assert Bar()
        """
    )
    result = _prune_inline_redundant_imports(source)
    inner = [
        ln
        for ln in result.splitlines()
        if "from mymod import" in ln and ln.startswith("        ")
    ]
    assert len(inner) == 1
    assert inner[0].startswith("        from mymod import Bar")


def test_add_re_exports_relative_from_uses_relative_prefix():
    # When relative_from is set, imports are computed via _relative_import_prefix
    # rather than _target_module_name, so "service/__init__.py" → ".utils"
    # (not ".service.utils").
    source = "# stayed\n"
    entity = _make_entity("Foo", 1, 1)
    placements = [GroupPlacement(group=["Foo"], target_file="service/utils.py")]
    entity_map = {"Foo": entity}
    entity_source_map = {"Foo": "class Foo: pass"}

    result = _add_re_exports(
        source,
        placements,
        entity_map,
        entity_source_map,
        relative_from="service/__init__.py",
    )

    assert "from .utils import Foo" in result
    # Must NOT use the fully-qualified form that would be wrong from __init__.py.
    assert "from .service.utils" not in result


def test_bump_relative_imports_single_dot():
    assert _bump_relative_imports("from .foo import bar") == "from ..foo import bar"


def test_bump_relative_imports_two_dots():
    assert _bump_relative_imports("from .. import baz") == "from ... import baz"


def test_bump_relative_imports_leaves_absolute():
    src = "import os\nfrom typing import List"
    assert _bump_relative_imports(src) == src


def test_bump_relative_imports_multiline():
    src = "from .a import x\nimport sys\nfrom ..b import y\n"
    result = _bump_relative_imports(src)
    assert "from ..a import x" in result
    assert "from ...b import y" in result
    assert "import sys" in result


def test_strip_top_level_import_lines_removes_imports():
    src = "import os\nfrom typing import List\n\n_CONST = 1\n"
    result = _strip_top_level_import_lines(src)
    assert "import os" not in result
    assert "from typing import List" not in result
    assert "_CONST = 1" in result


def test_strip_top_level_import_lines_no_imports():
    src = "_CONST = 1\n"
    assert _strip_top_level_import_lines(src) == src


def test_strip_top_level_import_lines_syntax_error():
    src = "def (\n"
    assert _strip_top_level_import_lines(src) == src
