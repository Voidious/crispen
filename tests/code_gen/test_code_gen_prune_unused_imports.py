from __future__ import annotations
from crispen.file_limiter.code_gen import _prune_unused_imports


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
