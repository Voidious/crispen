from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import _add_re_exports
from crispen.file_limiter.entity_parser import Entity, EntityKind
from tests.codegen_test_fixtures import _make_entity


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


def test_add_re_exports_from_import_line():
    # "from pathlib import Path" should be detected as an import line.
    source = "from pathlib import Path\n\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 3, 4)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {})
    lines = result.splitlines()
    from_import_idx = next(
        i for i, l in enumerate(lines) if "from pathlib import Path" in l
    )
    reexport_idx = next(i for i, l in enumerate(lines) if "from .utils import foo" in l)
    assert reexport_idx > from_import_idx


def test_add_re_exports_indented_local_import_not_treated_as_last_import():
    # Functions with local (indented) imports must not cause re-exports to be
    # inserted inside the function body.  The re-export should appear after the
    # top-level "import os" line, not after the indented "from x import y".
    source = (
        "import os\n"
        "\n"
        "def foo():\n"
        "    from unittest.mock import MagicMock\n"
        "    MagicMock()\n"
        "\n"
        "def bar():\n"
        "    pass\n"
    )
    entity = _make_entity("baz", 7, 8)
    placement = GroupPlacement(group=["baz"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"baz": entity}, {})
    # Re-export must appear immediately after "import os", not inside foo().
    lines = result.splitlines()
    os_idx = next(i for i, l in enumerate(lines) if l == "import os")
    reexport_idx = next(i for i, l in enumerate(lines) if "from .utils import baz" in l)
    assert reexport_idx == os_idx + 1
    # The function body must remain intact (local import line must still be there).
    assert "    from unittest.mock import MagicMock" in result
