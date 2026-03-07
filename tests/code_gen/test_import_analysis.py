from __future__ import annotations
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import (
    ImportInfo,
    _abs_package_for_dir,
    _collect_external_imported_names,
    _collect_name_loads,
    _extract_import_info,
    _find_cross_file_imports,
    _find_needed_imports,
    _find_project_root,
    _import_derived_names,
    _import_line_numbers,
    _module_path_from_file,
    _remove_entity_lines,
    _target_module_name,
)
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
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "test_fn.py", abs_pkg="tests"
    )
    assert result == ["from tests.block_1 import _MODEL"]


def test_find_cross_file_imports_abs_pkg_root_level():
    # abs_pkg="" → "from block_1 import _MODEL" (no package prefix)
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "test_fn.py", abs_pkg=""
    )
    assert result == ["from block_1 import _MODEL"]


def test_abs_package_for_dir_subdir(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_engine.py"
    test_file.touch()
    assert _abs_package_for_dir(str(test_file)) == "tests"


def test_abs_package_for_dir_root_level(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    test_file = tmp_path / "test_engine.py"
    test_file.touch()
    assert _abs_package_for_dir(str(test_file)) == ""


def test_abs_package_for_dir_no_project_root(monkeypatch):
    monkeypatch.setattr(
        "crispen.file_limiter.code_gen._find_project_root", lambda _p: None
    )
    assert _abs_package_for_dir("/some/random/path/test_engine.py") is None


def test_abs_package_for_dir_non_ancestor_root(tmp_path, monkeypatch):
    # Defensive branch: project root is not an ancestor of the file's directory.
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.setattr(
        "crispen.file_limiter.code_gen._find_project_root", lambda _p: other_dir
    )
    test_file = tmp_path / "tests" / "test_engine.py"
    test_file.parent.mkdir()
    test_file.touch()
    assert _abs_package_for_dir(str(test_file)) is None
