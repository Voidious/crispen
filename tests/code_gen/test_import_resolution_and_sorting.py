from __future__ import annotations
import textwrap
from crispen.import_sort import _sort_imports_pep8
from crispen.file_limiter.code_gen import (
    ImportInfo,
    _find_needed_imports,
    _find_type_checking_needed_imports,
    _merge_from_imports,
    _narrow_import_source,
    _prune_unused_imports,
)


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


def test_merge_from_imports_no_overlap():
    imports = ["from .a import x", "from .b import y"]
    assert _merge_from_imports(imports) == ["from .a import x", "from .b import y"]


def test_merge_from_imports_overlapping():
    imports = ["from .conv import A, C", "from .conv import B, C"]
    result = _merge_from_imports(imports)
    assert result == ["from .conv import A, B, C"]


def test_merge_from_imports_deduplicates_names():
    imports = ["from .m import foo, bar", "from .m import bar, baz"]
    result = _merge_from_imports(imports)
    assert result == ["from .m import bar, baz, foo"]


def test_merge_from_imports_preserves_plain_imports():
    imports = ["import os", "from .m import x", "import sys"]
    result = _merge_from_imports(imports)
    assert result == ["from .m import x", "import os", "import sys"]


def test_merge_from_imports_empty():
    assert _merge_from_imports([]) == []


def test_sort_imports_pep8_basic_ordering():
    # Third-party plain import after relative from-import → should be reordered.
    imports = [
        "from typing import Any",
        "from .conversion import foo",
        "import lupa",
    ]
    result = _sort_imports_pep8(imports)
    assert result == [
        "from typing import Any",
        "import lupa",
        "from .conversion import foo",
    ]


def test_sort_imports_pep8_future_first():
    imports = ["import os", "from __future__ import annotations", "from .x import y"]
    result = _sort_imports_pep8(imports)
    assert result[0] == "from __future__ import annotations"


def test_sort_imports_pep8_preserves_within_group_order():
    imports = ["from .b import y", "from .a import x"]
    result = _sort_imports_pep8(imports)
    # Both are local; original order preserved
    assert result == ["from .b import y", "from .a import x"]


def test_sort_imports_pep8_empty():
    assert _sort_imports_pep8([]) == []


def test_sort_imports_pep8_all_stdlib():
    imports = ["import os", "import sys", "from pathlib import Path"]
    result = _sort_imports_pep8(imports)
    assert result == imports  # already ordered, stable sort keeps original order


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


def test_prune_unused_imports_preserves_noqa_f401():
    # Imports marked "# noqa: F401" are intentional re-export stubs and must
    # never be pruned, even when the name is unused in the file body.
    source = (
        "from .utils import _helper  # fmt: skip # noqa: F401, E501\n"
        "\n"
        "def f():\n"
        "    pass\n"
    )
    result = _prune_unused_imports(source)
    assert "from .utils import _helper" in result


def test_prune_unused_imports_prunes_unused_without_noqa():
    # Without noqa, unused imports are still removed.
    source = "from .utils import _helper\n\ndef f():\n    pass\n"
    result = _prune_unused_imports(source)
    assert "from .utils import _helper" not in result
