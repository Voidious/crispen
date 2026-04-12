from __future__ import annotations
from crispen.file_limiter.code_gen import (
    ImportInfo,
    _extract_import_info,
    _find_needed_imports,
    _find_type_checking_needed_imports,
    _import_derived_names,
    _import_line_numbers,
    _narrow_import_source,
    _strip_top_level_import_lines,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind


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


def test_extract_import_info_multiline_parens_normalized():
    # Multi-line parenthesized from-import must be normalized to a single line
    # so that _merge_from_imports can process it without producing malformed output.
    source = "from pathlib import (\n    Path,\n    PurePath,\n)\n"
    infos = _extract_import_info(source)
    assert len(infos) == 1
    assert infos[0].source == "from pathlib import Path, PurePath"
    assert "\n" not in infos[0].source
    assert "Path" in infos[0].names
    assert "PurePath" in infos[0].names


def test_extract_import_info_type_checking_from_import():
    # Imports inside `if TYPE_CHECKING:` are extracted with is_type_checking=True.
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .config import MyConfig\n"
    )
    infos = _extract_import_info(source)
    tc = [i for i in infos if i.is_type_checking]
    assert len(tc) == 1
    assert "MyConfig" in tc[0].names
    assert tc[0].source == "from .config import MyConfig"
    assert tc[0].is_future is False


def test_extract_import_info_type_checking_plain_import():
    # Plain `import` inside `if TYPE_CHECKING:` is also captured.
    source = "if TYPE_CHECKING:\n    import sys\n"
    infos = _extract_import_info(source)
    tc = [i for i in infos if i.is_type_checking]
    assert len(tc) == 1
    assert "sys" in tc[0].names
    assert tc[0].is_type_checking is True


def test_extract_import_info_type_checking_not_is_future():
    # TYPE_CHECKING block imports must not be marked as is_future.
    source = "if TYPE_CHECKING:\n    from .foo import Bar\n"
    infos = _extract_import_info(source)
    tc = [i for i in infos if i.is_type_checking]
    assert all(not i.is_future for i in tc)


def test_extract_import_info_type_checking_skips_non_import_children():
    # Non-import statements inside a TYPE_CHECKING block (rare but valid)
    # must not cause errors and must be silently skipped.
    source = "if TYPE_CHECKING:\n    from .foo import Bar\n    x = 1\n"
    infos = _extract_import_info(source)
    tc = [i for i in infos if i.is_type_checking]
    assert len(tc) == 1
    assert "Bar" in tc[0].names


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


def test_narrow_import_source_syntax_error():
    # Invalid Python → original string returned unchanged.
    bad = "from ??? import Foo"
    assert _narrow_import_source(bad, {"Foo"}) == bad


def test_narrow_import_source_plain_import():
    # Non-ImportFrom statement (bare `import X`) → returned unchanged.
    src = "import os"
    assert _narrow_import_source(src, {"os"}) == src


def test_narrow_import_source_empty_keep():
    # keep_names matches nothing → alias_strs is empty → return original.
    src = "from mymod import A, B"
    assert _narrow_import_source(src, {"C"}) == src


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


def test_strip_top_level_import_lines_strips_type_checking_block():
    # `if TYPE_CHECKING:` blocks must be stripped so that their imports are
    # not emitted verbatim in sub-files (wrong path, wrong file).
    src = "if TYPE_CHECKING:\n" "    from .config import MyConfig\n" "\n" "_CONST = 1\n"
    result = _strip_top_level_import_lines(src)
    assert "TYPE_CHECKING" not in result
    assert "MyConfig" not in result
    assert "_CONST = 1" in result
