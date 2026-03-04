from __future__ import annotations
import textwrap
from crispen.file_limiter.code_gen import _prune_inline_redundant_imports


def test_prune_inline_no_top_level_imports():
    # No module-level imports → nothing can be redundant, return unchanged.
    source = "def f():\n    from os import path\n    path.join('a', 'b')\n"
    assert _prune_inline_redundant_imports(source) == source


def test_prune_inline_no_inner_imports():
    # Only top-level imports, no function-body imports → unchanged.
    source = "import os\n\ndef f():\n    return os.getcwd()\n"
    assert _prune_inline_redundant_imports(source) == source


def test_prune_inline_no_redundancy():
    # Inner import brings in a different name than the top-level import.
    source = "import os\n\ndef f():\n    from sys import argv\n    return argv\n"
    assert _prune_inline_redundant_imports(source) == source


def test_prune_inline_removes_fully_redundant_from_import():
    # Top-level import covers all names in the inner from-import → remove it.
    source = textwrap.dedent(
        """\
        from unittest.mock import patch
        from mymod import Foo

        def test_thing():
            from mymod import Foo
            assert Foo()
        """
    )
    result = _prune_inline_redundant_imports(source)
    assert result.count("from mymod import Foo") == 1
    assert "assert Foo()" in result


def test_prune_inline_narrows_partially_redundant_from_import():
    # Only one of two inner names is already at top level → narrow the inner import.
    source = textwrap.dedent(
        """\
        from mymod import Foo

        def test_thing():
            from mymod import Foo, Bar
            assert Foo() and Bar()
        """
    )
    result = _prune_inline_redundant_imports(source)
    lines = result.splitlines()
    inner = [ln for ln in lines if "from mymod import" in ln and ln.startswith("    ")]
    assert len(inner) == 1
    assert "Bar" in inner[0]
    assert "Foo" not in inner[0]


def test_prune_inline_removes_fully_redundant_plain_import():
    # Inner ``import x`` where x is already available at top level → removed.
    source = textwrap.dedent(
        """\
        import os

        def f():
            import os
            return os.getcwd()
        """
    )
    result = _prune_inline_redundant_imports(source)
    assert result.count("import os") == 1


def test_prune_inline_narrows_partially_redundant_plain_import():
    # ``import os, sys`` inside function where os is already top-level → narrows to sys.
    source = textwrap.dedent(
        """\
        import os

        def f():
            import os, sys
            return sys.argv
        """
    )
    result = _prune_inline_redundant_imports(source)
    inner = [
        ln
        for ln in result.splitlines()
        if ln.strip().startswith("import") and ln.startswith("    ")
    ]
    assert len(inner) == 1
    assert "sys" in inner[0]
    assert "os" not in inner[0]


def test_prune_inline_preserves_indentation():
    # The narrowed replacement line must preserve the original indentation.
    source = textwrap.dedent(
        """\
        from mymod import Foo

        def test_thing():
            if True:
                from mymod import Foo, Bar
                assert Bar()
        """
    )
    result = _prune_inline_redundant_imports(source)
    inner = [
        ln
        for ln in result.splitlines()
        if "from mymod import" in ln and ln.startswith("        ")
    ]
    assert len(inner) == 1
    assert inner[0].startswith("        from mymod import Bar")
