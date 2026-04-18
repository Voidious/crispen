from __future__ import annotations
import textwrap
from crispen.file_limiter.code_gen import (
    _inject_inline_imports,
    _inject_inline_test_imports_original,
    _inject_module_level_imports,
    _inject_type_checking_imports,
)


def test_inject_module_level_imports_docstring_only():
    # Source with only a docstring and no imports — insert after the docstring.
    src = '"""Module doc."""\n\nx = 1\n'
    result = _inject_module_level_imports(src, ["from . import converters"])
    assert '"""Module doc."""' in result
    assert "from . import converters" in result
    doc_pos = result.index('"""Module doc."""')
    imp_pos = result.index("from . import converters")
    assert doc_pos < imp_pos


def test_inject_module_level_imports_empty_list():
    src = "x = 1\n"
    assert _inject_module_level_imports(src, []) == src


def test_inject_module_level_imports_after_imports():
    src = "import os\n\nx = 1\n"
    result = _inject_module_level_imports(src, ["from . import converters"])
    assert result == "import os\nfrom . import converters\n\nx = 1\n"


def test_inject_module_level_imports_no_existing_imports():
    src = "x = 1\n"
    result = _inject_module_level_imports(src, ["from . import converters"])
    # Prepended before non-import content
    assert "from . import converters" in result
    assert result.index("from . import converters") < result.index("x = 1")


def test_inject_module_level_imports_sorted():
    src = "import os\n\nx = 1\n"
    result = _inject_module_level_imports(
        src, ["from . import z_mod", "from . import a_mod"]
    )
    lines = result.splitlines()
    import_lines = [ln for ln in lines if "import" in ln]
    assert import_lines.index("from . import a_mod") < import_lines.index(
        "from . import z_mod"
    )


def test_inject_module_level_imports_syntax_error_prepends():
    src = "def (broken:\n"
    result = _inject_module_level_imports(src, ["import os"])
    assert result.startswith("import os\n")


def test_inject_type_checking_imports_empty_list():
    src = "import os\n"
    assert _inject_type_checking_imports(src, []) == src


def test_inject_type_checking_imports_syntax_error():
    src = "def (broken:\n"
    assert _inject_type_checking_imports(src, ["from .config import Cfg"]) == src


def test_inject_type_checking_imports_all_already_present():
    # If every requested import is already in an existing TC block, no change.
    src = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .config import Cfg\n"
        "\n"
        "x = 1\n"
    )
    result = _inject_type_checking_imports(src, ["from .config import Cfg"])
    assert result == src


def test_inject_type_checking_imports_appends_to_existing_block():
    # New import should be appended inside the existing TYPE_CHECKING block.
    src = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .config import Cfg\n"
        "\n"
        "x = 1\n"
    )
    result = _inject_type_checking_imports(src, ["from .models import MyModel"])
    assert "from .models import MyModel" in result
    tc_start = result.index("if TYPE_CHECKING:")
    assert result.index("from .models import MyModel") > tc_start
    assert "x = 1" in result


def test_inject_type_checking_imports_creates_block_with_typing_import():
    # No existing TC block and TYPE_CHECKING not imported → add both.
    src = "from typing import List\n\ndef foo(x: 'Cfg') -> None:\n    pass\n"
    result = _inject_type_checking_imports(src, ["from .config import Cfg"])
    assert "from typing import TYPE_CHECKING" in result
    assert "if TYPE_CHECKING:" in result
    assert "    from .config import Cfg" in result


def test_inject_type_checking_imports_creates_block_type_checking_already_imported():
    # TYPE_CHECKING already in typing import → don't add it again.
    src = (
        "from typing import List, TYPE_CHECKING\n"
        "\n"
        "def foo(x: 'Cfg') -> None:\n"
        "    pass\n"
    )
    result = _inject_type_checking_imports(src, ["from .config import Cfg"])
    assert result.count("TYPE_CHECKING") == 2  # one in import, one in if-block
    assert "if TYPE_CHECKING:" in result
    assert "    from .config import Cfg" in result


def test_inject_type_checking_imports_block_after_last_import():
    # The new block should appear after the last import, before other code.
    src = "import os\nimport sys\n\nx = 1\n"
    result = _inject_type_checking_imports(src, ["from .config import Cfg"])
    lines = result.splitlines()
    sys_line = next(i for i, l in enumerate(lines) if "import sys" in l)
    if_line = next(i for i, l in enumerate(lines) if "if TYPE_CHECKING" in l)
    x_line = next(i for i, l in enumerate(lines) if "x = 1" in l)
    assert sys_line < if_line < x_line


def test_inject_inline_imports_into_function():
    src = "def foo():\n    return 1\n"
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert result == "def foo():\n    from .bar import Baz\n    return 1\n"


def test_inject_inline_imports_skips_docstring():
    src = 'def foo():\n    """Doc."""\n    return 1\n'
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert (
        result == 'def foo():\n    """Doc."""\n    from .bar import Baz\n    return 1\n'
    )


def test_inject_inline_imports_into_class():
    src = "class Foo:\n    x = 1\n"
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert result == "class Foo:\n    from .bar import Baz\n    x = 1\n"


def test_inject_inline_imports_toplevel_noop():
    # TOP_LEVEL entity (bare if-statement): no body scope, returns unchanged.
    src = "if True:\n    pass\n"
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert result == src


def test_inject_inline_imports_empty_list_noop():
    src = "def foo():\n    pass\n"
    assert _inject_inline_imports(src, []) == src


def test_inject_inline_imports_syntax_error_noop():
    src = "def (invalid"
    assert _inject_inline_imports(src, ["from .x import Y"]) == src


def test_inject_inline_imports_empty_source_noop():
    # Empty source parses to empty tree.body — returns unchanged.
    assert _inject_inline_imports("", ["from .x import Y"]) == ""


def test_inject_inline_imports_only_docstring_injects_after():
    # Function with only a docstring — inserts after docstring (at body[0] line)
    # since len(body) == 1.
    src = 'def foo():\n    """Only doc."""\n'
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert result == 'def foo():\n    from .bar import Baz\n    """Only doc."""\n'


def test_inject_inline_test_imports_original_basic():
    source = textwrap.dedent(
        """\
        def runner():
            TestFoo()
        """
    )
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg="pkg.tests", original_basename="test_orig.py"
    )
    assert "from pkg.tests.sub.test_foo import TestFoo" in result
    # Import appears inside the function body, not before the def line.
    lines = result.splitlines()
    def_idx = next(i for i, l in enumerate(lines) if l.startswith("def runner"))
    import_idx = next(i for i, l in enumerate(lines) if "import TestFoo" in l)
    assert import_idx > def_idx


def test_inject_inline_test_imports_original_skips_docstring():
    source = textwrap.dedent(
        """\
        def runner():
            \"\"\"Run tests.\"\"\"
            TestFoo()
        """
    )
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg="tests", original_basename="test_orig.py"
    )
    lines = result.splitlines()
    doc_idx = next(i for i, l in enumerate(lines) if '"""Run tests."""' in l)
    import_idx = next(i for i, l in enumerate(lines) if "import TestFoo" in l)
    assert import_idx > doc_idx


def test_inject_inline_test_imports_original_no_reference():
    source = "def runner():\n    pass\n"
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg="tests", original_basename="test_orig.py"
    )
    assert result == source


def test_inject_inline_test_imports_original_empty_map():
    source = "def runner():\n    TestFoo()\n"
    result = _inject_inline_test_imports_original(
        source, {}, abs_pkg="tests", original_basename="test_orig.py"
    )
    assert result == source


def test_inject_inline_test_imports_original_syntax_error():
    result = _inject_inline_test_imports_original(
        "def (invalid",
        {"TestFoo": "sub/test_foo.py"},
        abs_pkg="tests",
        original_basename="test_orig.py",
    )
    assert result == "def (invalid"


def test_inject_inline_test_imports_original_relative_import():
    source = "def runner():\n    TestFoo()\n"
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg=None, original_basename="test_orig.py"
    )
    assert "from .sub.test_foo import TestFoo" in result


def test_inject_inline_test_imports_original_unreferenced_symbol_skipped():
    # Function references `helper` (not test-named) and `other_func`, neither
    # of which is in migrated_test_symbols — the false branch of `if tfile:`.
    source = "def runner():\n    helper()\n    other_func()\n"
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg="tests", original_basename="test_orig.py"
    )
    assert result == source
