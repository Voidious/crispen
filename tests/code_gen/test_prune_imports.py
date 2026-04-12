from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    _prune_inline_redundant_imports,
    _prune_unused_imports,
    generate_file_splits,
)
from .shared_helpers import _classified, _make_entity, _plan


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


def test_prune_unused_imports_preserves_noqa_f401():
    # Imports marked "# noqa: F401" are intentional re-export stubs and must
    # never be pruned, even when the name is unused in the file body.
    source = (
        "from .utils import _helper  # fmt: skip # noqa: F401, E501\n"
        "\n"
        "def f():\n"
        "    pass\n"
    )
    result = _prune_unused_imports(source)
    assert "from .utils import _helper" in result


def test_prune_unused_imports_prunes_unused_without_noqa():
    # Without noqa, unused imports are still removed.
    source = "from .utils import _helper\n\ndef f():\n    pass\n"
    result = _prune_unused_imports(source)
    assert "from .utils import _helper" not in result


def test_prune_inline_syntax_error():
    # Unparseable source → returned unchanged.
    source = "def (invalid syntax"
    assert _prune_inline_redundant_imports(source) == source


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


def test_prune_inline_preserves_type_checking_block():
    # Imports inside 'if TYPE_CHECKING:' must never be stripped even when the
    # same name is already imported at module level — removing them would leave
    # an empty (and syntactically invalid) if-block.
    source = textwrap.dedent(
        """\
        from typing import TYPE_CHECKING
        from mymod import Foo

        if TYPE_CHECKING:
            from mymod import Foo
        """
    )
    result = _prune_inline_redundant_imports(source)
    assert result == source


def test_generate_file_splits_removes_inline_redundant_imports():
    # When a split new file has both a top-level import and an inline re-import
    # of the same name, the inline one should be removed.
    source = textwrap.dedent(
        """\
        from mymod import Helper

        def test_uses_helper():
            from mymod import Helper
            assert Helper()
        """
    )
    entity = _make_entity("test_uses_helper", 3, 5)
    c = _classified(entities=[entity])
    plan = _plan(
        [GroupPlacement(group=["test_uses_helper"], target_file="test_split.py")]
    )
    result = generate_file_splits(c, plan, source, "big.py")
    new_src = result.new_files["test_split.py"]
    # The inline re-import should be removed; the module-level one covers it.
    assert new_src.count("from mymod import Helper") == 1
