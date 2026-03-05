from __future__ import annotations
from crispen.file_limiter.code_gen import ImportInfo, _find_needed_imports


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
