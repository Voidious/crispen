from __future__ import annotations
import textwrap
from crispen.file_limiter.code_gen import _prune_unused_imports


def test_prune_unused_imports_narrows_partial_from_import():
    # Only List is used — import narrowed to just List.
    source = "from typing import Dict, List\n\ndef f(x: List):\n    return x\n"
    result = _prune_unused_imports(source)
    assert "from typing import List" in result
    assert "Dict" not in result


def test_prune_unused_imports_narrows_plain_import():
    # import x, y where only y is used → narrowed to import y.
    source = "import os, sys\n\ndef f():\n    sys.exit()\n"
    result = _prune_unused_imports(source)
    assert "import sys" in result
    assert "os" not in result


def test_prune_unused_imports_multiline_import_collapsed():
    # Multi-line parenthesised import is collapsed to a single line.
    source = textwrap.dedent(
        """\
        from typing import (
            Dict,
            List,
        )

        def f(x: List):
            return x
        """
    )
    result = _prune_unused_imports(source)
    assert "from typing import List" in result
    assert "Dict" not in result
    assert "(\n" not in result


def test_prune_unused_imports_relative_import_narrowed():
    # Relative from-import is reconstructed with dots preserved.
    source = "from .utils import foo, bar\n\ndef f():\n    return foo()\n"
    result = _prune_unused_imports(source)
    assert "from .utils import foo" in result
    assert "bar" not in result
