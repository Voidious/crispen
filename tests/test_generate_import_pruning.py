from __future__ import annotations
from tests.codegen_entity_src_helpers import _gen_utils_new_src_for_single_entity
from tests.codegen_migration_split_setup import _setup_generate_splits_for_foo_migration


def test_generate_prunes_unused_names_from_multiname_import():
    # foo uses only List, not Dict; the new file's import should be narrowed.
    source = "from typing import Dict, List\n\ndef foo(x: List):\n    return x\n"
    result, new_src = _gen_utils_new_src_for_single_entity(
        source=source, original_filename="big.py"
    )
    assert "from typing import List" in new_src
    assert "Dict" not in new_src


def test_generate_prunes_fully_unused_import_from_original():
    # import os is only used by foo; after foo migrates the original no longer
    # needs os, so the import should be removed.
    source = "import os\n\ndef foo():\n    os.getcwd()\n\ndef bar():\n    return 1\n"
    result = _setup_generate_splits_for_foo_migration(source)

    assert "from .utils import foo" in result.original_source
    assert "import os" not in result.original_source
    assert "def bar():" in result.original_source


def test_generate_narrows_partial_unused_import_in_original():
    # foo uses Dict; bar uses List.  After foo migrates, Dict should be
    # stripped from the original's import while List is kept.
    source = (
        "from typing import Dict, List\n\n"
        "def foo(x: Dict):\n    return x\n\n"
        "def bar(x: List):\n    return x\n"
    )
    result = _setup_generate_splits_for_foo_migration(source)

    assert "from typing import List" in result.original_source
    assert "Dict" not in result.original_source
