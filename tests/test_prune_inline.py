from __future__ import annotations
from crispen.file_limiter.code_gen import _prune_inline_redundant_imports


def test_prune_inline_syntax_error():
    # Unparseable source → returned unchanged.
    source = "def (invalid syntax"
    assert _prune_inline_redundant_imports(source) == source
