from __future__ import annotations
from crispen.file_limiter.code_gen import (
    ImportInfo,
    _collect_external_imported_names,
    _collect_name_loads,
    _extract_import_info,
    _find_needed_imports,
    _import_derived_names,
    _import_line_numbers,
    _target_module_name,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind


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


def test_target_module_name_init():
    # __init__.py represents the package, not a "__init__" submodule.
    assert _target_module_name("pkg/__init__.py") == "pkg"


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
