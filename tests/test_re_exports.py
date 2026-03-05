from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import _add_re_exports
from crispen.file_limiter.entity_parser import Entity, EntityKind
from tests.test_helpers import _make_entity


def test_add_re_exports_top_level_import_derived_names_not_re_exported():
    # A TOP_LEVEL entity that includes import statements: the names introduced
    # by those imports must NOT appear in re-exports because they are preserved
    # in the original file by _remove_entity_lines, not moved to the new file.
    source = "import os\n\nMY_CONST\n"  # MY_CONST still loaded
    entity_src = "from typing import Dict\n\nMY_CONST = 42\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["Dict", "MY_CONST"])
    placement = GroupPlacement(group=["_block_1"], target_file="constants.py")
    result = _add_re_exports(
        source, [placement], {"_block_1": entity}, {"_block_1": entity_src}
    )
    assert "MY_CONST" in result  # assignment-defined name re-exported
    assert "Dict" not in result  # import-derived name suppressed


def test_add_re_exports_all_private_no_change():
    # Private name not called anywhere in remaining source → no import added.
    source = "import os\n\ndef _helper():\n    pass\n"
    entity = _make_entity("_helper", 3, 4)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"_helper": entity}, {})
    assert result == source


def test_add_re_exports_private_referenced_in_source():
    # Private name still called in remaining source → import is added.
    source = "import os\n\n_helper()\n"
    entity = _make_entity("_helper", 3, 3)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"_helper": entity}, {})
    assert "from .utils import _helper" in result


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
    noqa_lines = [line for line in lines if "# noqa F401" in line]
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
    noqa_lines = [l for l in lines if "# noqa F401" in l]
    plain_lines = [l for l in lines if "from .utils import" in l and "# noqa" not in l]
    assert len(noqa_lines) == 1
    assert "_reexport" in noqa_lines[0]
    assert "_used" not in noqa_lines[0]
    assert len(plain_lines) == 1
    assert "_used" in plain_lines[0]
