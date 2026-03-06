from __future__ import annotations
from crispen.file_limiter.code_gen import (
    _extract_import_info,
    _import_derived_names,
    _import_line_numbers,
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
