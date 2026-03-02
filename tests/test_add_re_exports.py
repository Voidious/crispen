from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import _add_re_exports
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .helpers import _make_entity


def test_add_re_exports_all_private_no_change():
    # Private name not called anywhere in remaining source → no import added.
    source = "import os\n\ndef _helper():\n    pass\n"
    entity = _make_entity("_helper", 3, 4)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"_helper": entity})
    assert result == source


def test_add_re_exports_private_referenced_in_source():
    # Private name still called in remaining source → import is added.
    source = "import os\n\n_helper()\n"
    entity = _make_entity("_helper", 3, 3)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"_helper": entity})
    assert "from .utils import _helper" in result


def test_add_re_exports_public_inserted_after_imports():
    source = "import os\n\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 3, 4)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity})
    assert "from .utils import foo" in result
    # Re-export line should come after "import os"
    lines = result.splitlines()
    import_idx = next(i for i, l in enumerate(lines) if "import os" in l)
    reexport_idx = next(i for i, l in enumerate(lines) if "from .utils import foo" in l)
    assert reexport_idx > import_idx


def test_add_re_exports_no_import_in_source():
    # No imports in source → re-export inserted at beginning.
    source = "\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 2, 3)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity})
    assert "from .utils import foo" in result


def test_add_re_exports_from_import_line():
    # "from pathlib import Path" should be detected as an import line.
    source = "from pathlib import Path\n\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 3, 4)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity})
    lines = result.splitlines()
    from_import_idx = next(
        i for i, l in enumerate(lines) if "from pathlib import Path" in l
    )
    reexport_idx = next(i for i, l in enumerate(lines) if "from .utils import foo" in l)
    assert reexport_idx > from_import_idx


def test_add_re_exports_multiple_targets_sorted():
    source = "import os\n"
    e1 = _make_entity("foo", 1, 2)
    e2 = _make_entity("bar", 3, 4)
    placements = [
        GroupPlacement(group=["foo"], target_file="b_module.py"),
        GroupPlacement(group=["bar"], target_file="a_module.py"),
    ]
    result = _add_re_exports(source, placements, {"foo": e1, "bar": e2})
    # a_module comes before b_module (sorted)
    a_idx = result.index("a_module")
    b_idx = result.index("b_module")
    assert a_idx < b_idx


def test_add_re_exports_mixed_public_private():
    source = "import os\n"
    entity_map = {
        "pub": _make_entity("pub", 1, 2),
        "_priv": _make_entity("_priv", 3, 4),
    }
    placement = GroupPlacement(group=["pub", "_priv"], target_file="utils.py")
    result = _add_re_exports(source, [placement], entity_map)
    # Only "pub" in re-export, not "_priv"
    assert "pub" in result
    assert "_priv" not in result


def test_add_re_exports_test_function_not_re_exported():
    # test_ functions must never get a proxy import — pytest would discover and
    # run them twice (once from the original file, once from the new file).
    source = "import os\n"
    entity = _make_entity("test_something", 1, 3)
    placement = GroupPlacement(group=["test_something"], target_file="tests/helpers.py")
    result = _add_re_exports(source, [placement], {"test_something": entity})
    assert result == source


def test_add_re_exports_test_function_re_exported_when_referenced():
    # If something in the remaining source actually calls test_something (unusual
    # but possible), the proxy import is still added.
    source = "import os\n\ntest_something()\n"
    entity = _make_entity("test_something", 1, 3)
    placement = GroupPlacement(group=["test_something"], target_file="tests/helpers.py")
    result = _add_re_exports(source, [placement], {"test_something": entity})
    assert "from .tests.helpers import test_something" in result


def test_add_re_exports_top_level_block_private_names_referenced():
    # TOP_LEVEL block entity name (_block_1) differs from its defined names.
    # Both defined names are still loaded in remaining source → re-imported.
    source = "import os\n\n_DUP_SOURCE\n_DUP_RANGES\n"
    entity = Entity(
        EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_DUP_SOURCE", "_DUP_RANGES"]
    )
    placement = GroupPlacement(group=["_block_1"], target_file="test_helpers.py")
    result = _add_re_exports(source, [placement], {"_block_1": entity})
    assert "from .test_helpers import _DUP_RANGES, _DUP_SOURCE" in result


def test_add_re_exports_entity_not_in_map_falls_back_to_entity_name():
    # Entity name in group is missing from entity_map → falls back to entity name.
    source = "import os\n\nghost()\n"  # 'ghost' is still referenced
    placement = GroupPlacement(group=["ghost"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {})
    assert "from .utils import ghost" in result


def test_add_re_exports_top_level_block_private_names_not_referenced():
    # TOP_LEVEL block entity whose defined name is private and not used → no import.
    source = "import os\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    placement = GroupPlacement(group=["_block_1"], target_file="constants.py")
    result = _add_re_exports(source, [placement], {"_block_1": entity})
    assert result == source
