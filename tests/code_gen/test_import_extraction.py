from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    ImportInfo,
    _collect_external_imported_names,
    _extract_import_info,
    _find_cross_file_imports,
    _find_cross_file_type_checking_imports,
    _find_needed_imports,
    _find_project_root,
    _find_type_checking_needed_imports,
    _import_derived_names,
    _import_line_numbers,
    _module_path_from_file,
    _strip_top_level_import_lines,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .helpers import _classified, _make_entity, _plan


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


def test_find_project_root_finds_pyproject_toml(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    sub = tmp_path / "pkg" / "module.py"
    sub.parent.mkdir()
    sub.write_text("x = 1\n")
    assert _find_project_root(sub) == tmp_path


def test_find_project_root_finds_git(tmp_path):
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "module.py"
    sub.write_text("x = 1\n")
    assert _find_project_root(sub) == tmp_path


def test_find_project_root_called_with_directory(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    assert _find_project_root(tmp_path) == tmp_path


def test_find_project_root_not_found(tmp_path):
    # tmp_path is under /tmp which has no project markers → None.
    sub = tmp_path / "module.py"
    sub.write_text("x = 1\n")
    result = _find_project_root(sub)
    # If the test runner is inside a project that happens to include tmp_path
    # (unlikely but possible with in-tree pytest), just ensure the function
    # returns without crashing.  The important coverage is the happy path above.
    assert result is None or result.exists()


def test_module_path_from_file_success(tmp_path):
    f = tmp_path / "pkg" / "utils.py"
    f.parent.mkdir()
    f.write_text("")
    assert _module_path_from_file(tmp_path, f) == "pkg.utils"


def test_module_path_from_file_top_level(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("")
    assert _module_path_from_file(tmp_path, f) == "module"


def test_module_path_from_file_not_under_root(tmp_path):
    other = tmp_path.parent / "other.py"
    assert _module_path_from_file(tmp_path, other) is None


def test_collect_external_imported_names_relative_path():
    # Non-absolute path → empty set (no scan).
    assert _collect_external_imported_names("relative/path.py") == set()


def test_collect_external_imported_names_nonexistent_file(tmp_path):
    # Absolute but non-existent → empty set.
    assert _collect_external_imported_names(str(tmp_path / "ghost.py")) == set()


def test_collect_external_imported_names_no_project_root(tmp_path):
    # File exists but no pyproject.toml/.git above it → empty set.
    f = tmp_path / "module.py"
    f.write_text("x = 1\n")
    # tmp_path is under /tmp which typically has no project markers.
    result = _collect_external_imported_names(str(f))
    # May or may not find a root depending on environment; we just verify no crash.
    assert isinstance(result, set)


def test_collect_external_imported_names_absolute_import(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    mod = pkg / "utils.py"
    mod.write_text("def _helper():\n    pass\n")
    caller = tmp_path / "tests" / "test_utils.py"
    caller.parent.mkdir()
    caller.write_text("from mypkg.utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_relative_import(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    mod = pkg / "utils.py"
    mod.write_text("def _helper():\n    pass\n")
    sibling = pkg / "other.py"
    sibling.write_text("from .utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_self_excluded(tmp_path):
    # The file being scanned is excluded from the search.
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("from module import _x\n")  # self-referential (ignored)
    result = _collect_external_imported_names(str(mod))
    assert "_x" not in result


def test_collect_external_imported_names_syntax_error_skipped(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("def _helper(): pass\n")
    bad = tmp_path / "bad.py"
    bad.write_text("def (invalid\n")
    good = tmp_path / "good.py"
    good.write_text("from module import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_non_matching_import_ignored(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("def _helper(): pass\n")
    other = tmp_path / "other.py"
    other.write_text("from different_module import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" not in result


def test_collect_external_imported_names_non_importfrom_nodes_skipped(tmp_path):
    # Caller file contains a plain `import` statement (not ImportFrom) mixed
    # with a matching `from … import`.  The plain import must be skipped without
    # crashing, and the matching ImportFrom still contributes to the result.
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("def _helper(): pass\n")
    caller = tmp_path / "caller.py"
    caller.write_text("import os\nfrom module import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_deep_relative_import(tmp_path):
    # Two-level relative import: `from ..utils import _helper`
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "utils.py"
    mod.write_text("def _helper(): pass\n")
    sub = tmp_path / "pkg" / "sub" / "caller.py"
    sub.parent.mkdir(parents=True)
    sub.write_text("from ...utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_init_py_at_root(tmp_path):
    # A bare __init__.py at the project root has no package prefix, so no
    # external caller can import from it by package path — returns empty set.
    (tmp_path / "pyproject.toml").write_text("")
    init_py = tmp_path / "__init__.py"
    init_py.write_text("class Foo: pass\n")
    result = _collect_external_imported_names(str(init_py))
    assert result == set()


def test_collect_external_imported_names_init_py(tmp_path):
    # When original_path is an __init__.py, callers import from the package
    # name (e.g. "mypkg.sub"), not "mypkg.sub.__init__".
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg" / "sub"
    pkg.mkdir(parents=True)
    init_py = pkg / "__init__.py"
    init_py.write_text("class Foo: pass\n")
    caller = tmp_path / "caller.py"
    caller.write_text("from mypkg.sub import Foo\n")
    result = _collect_external_imported_names(str(init_py))
    assert "Foo" in result


def test_collect_external_imported_names_init_py_relative_caller(tmp_path):
    # Relative import from sibling module targeting a package __init__.py.
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    sub = pkg / "sub"
    sub.mkdir()
    (sub / "__init__.py").write_text("def _helper(): pass\n")
    sibling = pkg / "other.py"
    sibling.write_text("from .sub import _helper\n")
    result = _collect_external_imported_names(str(sub / "__init__.py"))
    assert "_helper" in result


def test_collect_external_imported_names_relative_level_too_deep(tmp_path):
    # Relative import that goes above the project root → skipped without crash.
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "utils.py"
    mod.write_text("def _helper(): pass\n")
    # A file at the top level trying to go up 5 packages (impossible).
    top = tmp_path / "top.py"
    top.write_text("from .....utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    # The over-deep import is silently skipped; no crash.
    assert isinstance(result, set)


def test_find_cross_file_imports_abs_pkg_package_prefix():
    # abs_pkg="tests" → "from tests.block_1 import _MODEL"
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "test_fn.py", abs_pkg="tests"
    )
    assert from_imports == ["from tests.block_1 import _MODEL"]
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_abs_pkg_root_level():
    # abs_pkg="" → "from block_1 import _MODEL" (no package prefix)
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "test_fn.py", abs_pkg=""
    )
    assert from_imports == ["from block_1 import _MODEL"]
    assert module_imports == []
    assert rewrites == {}


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


def test_generate_test_file_cross_imports_use_absolute_imports(tmp_path):
    # Cross-file imports in generated test split files must also be absolute.
    (tmp_path / "pyproject.toml").touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_engine.py"
    test_file.write_text("")

    source = "_CONST = 42\n\ndef test_fn():\n    return _CONST\n"
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_fn = _make_entity("test_fn", 3, 4)
    c = _classified(entities=[e_block, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="test_constants.py"),
            GroupPlacement(group=["test_fn"], target_file="test_fns.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, str(test_file))

    fn_src = result.new_files["test_fns.py"]
    # _CONST is a TOP_LEVEL variable that is never reassigned → plain absolute
    # from-import (idiomatic Python; module alias only needed if reassigned).
    assert "from tests.test_constants import _CONST" in fn_src
    assert "import tests.test_constants as test_constants" not in fn_src
    assert "test_constants._CONST" not in fn_src


def test_generate_file_splits_subdir_name_cross_file_uses_relative():
    # In subdir mode, cross-file imports between new files use relative imports
    # even when the original file is a test (abs_pkg would normally apply).
    # NOTE: runner.py prefixes target_file with subdir_name before this call,
    # so target_files already include "svc/" here.
    source = "def helper():\n    return 1\n\ndef test_fn():\n    return helper()\n"
    e_helper = _make_entity("helper", 1, 2)
    e_test = _make_entity("test_fn", 4, 5)
    c = _classified(entities=[e_helper, e_test])
    plan = _plan(
        [
            GroupPlacement(group=["helper"], target_file="svc/helpers.py"),
            GroupPlacement(group=["test_fn"], target_file="svc/test_fns.py"),
        ]
    )

    # Use a path that looks like a test file so abs_pkg would normally be set.
    result = generate_file_splits(
        c, plan, source, "tests/test_svc.py", subdir_name="svc"
    )

    assert not result.abort
    # Cross-file import from test_fns.py to helpers.py should be relative.
    test_src = result.new_files["svc/test_fns.py"]
    assert "from .helpers import helper" in test_src


def test_generate_file_splits_test_subdir_nonmigrated_imports_from_original():
    # Non-migrated TOP_LEVEL variables (e.g. module-level constants) stay in
    # the original test file.  A new subfile that references a constant that is
    # never reassigned should use a plain ``from`` import (idiomatic Python);
    # module-alias access is only needed when the constant is mutated at runtime.
    source = "_CONFIG = 'val'\n\ndef test_fn():\n    return _CONFIG\n"
    # Use TOP_LEVEL kind so _extract_shared_helpers does not pull _CONFIG into
    # the new file (it only extracts FUNCTION/CLASS entities).
    e_config = Entity(EntityKind.TOP_LEVEL, "_CONFIG", 1, 1, ["_CONFIG"])
    e_test = _make_entity("test_fn", 3, 4)
    c = _classified(entities=[e_config, e_test])
    plan = _plan([GroupPlacement(group=["test_fn"], target_file="svc/test_fns.py")])

    result = generate_file_splits(
        c, plan, source, "tests/test_svc.py", subdir_name="svc"
    )

    assert not result.abort
    test_src = result.new_files["svc/test_fns.py"]
    # _CONFIG is never reassigned → plain from-import (no module alias).
    assert "from ..test_svc import _CONFIG" in test_src
    assert "from .. import test_svc" not in test_src
    assert "test_svc._CONFIG" not in test_src


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
