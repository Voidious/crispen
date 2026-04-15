from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.import_sort import _sort_imports_pep8
from crispen.file_limiter.code_gen import (
    _extract_import_info,
    _import_derived_names,
    _import_line_numbers,
    _merge_from_imports,
    _prune_inline_redundant_imports,
    _prune_unused_imports,
    _strip_top_level_import_lines,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .helpers import _classified, _make_entity, _plan


def test_extract_import_info_syntax_error():
    assert _extract_import_info("def (invalid") == []


def test_extract_import_info_plain_import():
    infos = _extract_import_info("import os\n")
    assert len(infos) == 1
    assert "os" in infos[0].names
    assert infos[0].is_future is False


def test_extract_import_info_import_with_asname():
    infos = _extract_import_info("import os as operating_system\n")
    assert infos[0].names == ["operating_system"]


def test_extract_import_info_dotted_import():
    infos = _extract_import_info("import os.path\n")
    assert infos[0].names == ["os"]


def test_extract_import_info_from_import():
    infos = _extract_import_info("from pathlib import Path\n")
    assert "Path" in infos[0].names
    assert infos[0].is_future is False


def test_extract_import_info_from_import_with_asname():
    infos = _extract_import_info("from pathlib import Path as P\n")
    assert infos[0].names == ["P"]


def test_extract_import_info_future_import():
    infos = _extract_import_info("from __future__ import annotations\n")
    assert infos[0].is_future is True
    assert "annotations" in infos[0].names


def test_extract_import_info_skips_non_imports():
    infos = _extract_import_info("def foo():\n    pass\n")
    assert infos == []


def test_extract_import_info_multiple():
    source = "import os\nfrom pathlib import Path\n"
    infos = _extract_import_info(source)
    assert len(infos) == 2


def test_extract_import_info_multiline_parens_normalized():
    # Multi-line parenthesized from-import must be normalized to a single line
    # so that _merge_from_imports can process it without producing malformed output.
    source = "from pathlib import (\n    Path,\n    PurePath,\n)\n"
    infos = _extract_import_info(source)
    assert len(infos) == 1
    assert infos[0].source == "from pathlib import Path, PurePath"
    assert "\n" not in infos[0].source
    assert "Path" in infos[0].names
    assert "PurePath" in infos[0].names


def test_extract_import_info_type_checking_from_import():
    # Imports inside `if TYPE_CHECKING:` are extracted with is_type_checking=True.
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .config import MyConfig\n"
    )
    infos = _extract_import_info(source)
    tc = [i for i in infos if i.is_type_checking]
    assert len(tc) == 1
    assert "MyConfig" in tc[0].names
    assert tc[0].source == "from .config import MyConfig"
    assert tc[0].is_future is False


def test_extract_import_info_type_checking_plain_import():
    # Plain `import` inside `if TYPE_CHECKING:` is also captured.
    source = "if TYPE_CHECKING:\n    import sys\n"
    infos = _extract_import_info(source)
    tc = [i for i in infos if i.is_type_checking]
    assert len(tc) == 1
    assert "sys" in tc[0].names
    assert tc[0].is_type_checking is True


def test_extract_import_info_type_checking_not_is_future():
    # TYPE_CHECKING block imports must not be marked as is_future.
    source = "if TYPE_CHECKING:\n    from .foo import Bar\n"
    infos = _extract_import_info(source)
    tc = [i for i in infos if i.is_type_checking]
    assert all(not i.is_future for i in tc)


def test_extract_import_info_type_checking_skips_non_import_children():
    # Non-import statements inside a TYPE_CHECKING block (rare but valid)
    # must not cause errors and must be silently skipped.
    source = "if TYPE_CHECKING:\n    from .foo import Bar\n    x = 1\n"
    infos = _extract_import_info(source)
    tc = [i for i in infos if i.is_type_checking]
    assert len(tc) == 1
    assert "Bar" in tc[0].names


def test_import_derived_names_plain_import():
    src = "import os\nimport sys\n"
    assert _import_derived_names(src) == {"os", "sys"}


def test_import_derived_names_from_import():
    src = "from typing import Dict, List\n"
    assert _import_derived_names(src) == {"Dict", "List"}


def test_import_derived_names_aliased():
    src = "import libcst as cst\nfrom dataclasses import dataclass\n"
    assert _import_derived_names(src) == {"cst", "dataclass"}


def test_import_derived_names_ignores_assignments():
    src = "_MODEL = 'x'\n_MIN = 3\n"
    assert _import_derived_names(src) == set()


def test_import_derived_names_syntax_error():
    assert _import_derived_names("def (\n") == set()


def test_import_line_numbers_basic():
    src = "import os\n_CONST = 1\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 5, 6, [])
    # Entity starts at line 5; "import os" is relative line 1 → absolute line 5.
    result = _import_line_numbers(entity, src)
    assert result == {5}


def test_import_line_numbers_no_imports():
    src = "_CONST = 1\n_OTHER = 2\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 2, [])
    assert _import_line_numbers(entity, src) == set()


def test_import_line_numbers_syntax_error():
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, [])
    assert _import_line_numbers(entity, "def (\n") == set()


def test_merge_from_imports_no_overlap():
    imports = ["from .a import x", "from .b import y"]
    assert _merge_from_imports(imports) == ["from .a import x", "from .b import y"]


def test_merge_from_imports_overlapping():
    imports = ["from .conv import A, C", "from .conv import B, C"]
    result = _merge_from_imports(imports)
    assert result == ["from .conv import A, B, C"]


def test_merge_from_imports_deduplicates_names():
    imports = ["from .m import foo, bar", "from .m import bar, baz"]
    result = _merge_from_imports(imports)
    assert result == ["from .m import bar, baz, foo"]


def test_merge_from_imports_preserves_plain_imports():
    imports = ["import os", "from .m import x", "import sys"]
    result = _merge_from_imports(imports)
    assert result == ["from .m import x", "import os", "import sys"]


def test_merge_from_imports_empty():
    assert _merge_from_imports([]) == []


def test_sort_imports_pep8_basic_ordering():
    # Third-party plain import after relative from-import → should be reordered.
    imports = [
        "from typing import Any",
        "from .conversion import foo",
        "import lupa",
    ]
    result = _sort_imports_pep8(imports)
    assert result == [
        "from typing import Any",
        "import lupa",
        "from .conversion import foo",
    ]


def test_sort_imports_pep8_future_first():
    imports = ["import os", "from __future__ import annotations", "from .x import y"]
    result = _sort_imports_pep8(imports)
    assert result[0] == "from __future__ import annotations"


def test_sort_imports_pep8_preserves_within_group_order():
    imports = ["from .b import y", "from .a import x"]
    result = _sort_imports_pep8(imports)
    # Both are local; original order preserved
    assert result == ["from .b import y", "from .a import x"]


def test_sort_imports_pep8_empty():
    assert _sort_imports_pep8([]) == []


def test_sort_imports_pep8_all_stdlib():
    imports = ["import os", "import sys", "from pathlib import Path"]
    result = _sort_imports_pep8(imports)
    assert result == imports  # already ordered, stable sort keeps original order


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


def test_generate_prunes_unused_names_from_multiname_import():
    # foo uses only List, not Dict; the new file's import should be narrowed.
    source = "from typing import Dict, List\n\ndef foo(x: List):\n    return x\n"
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert "from typing import List" in new_src
    assert "Dict" not in new_src


def test_generate_prunes_fully_unused_import_from_original():
    # import os is only used by foo; after foo migrates the original no longer
    # needs os, so the import should be removed.
    source = "import os\n\ndef foo():\n    os.getcwd()\n\ndef bar():\n    return 1\n"
    e_foo = _make_entity("foo", 3, 4)
    e_bar = _make_entity("bar", 6, 7)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

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
    e_foo = _make_entity("foo", 3, 4)
    e_bar = _make_entity("bar", 6, 7)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "from typing import List" in result.original_source
    assert "Dict" not in result.original_source


def test_generate_migrated_top_level_import_names_not_in_cross_file_imports():
    # Regression: when a TOP_LEVEL entity containing "from dataclasses import
    # dataclass" is migrated, the name "dataclass" must NOT be added to the
    # name→target-file map.  A FUNCTION entity in a separate new file that also
    # uses dataclass should get "from dataclasses import dataclass" (via
    # _find_needed_imports) rather than "from .constants import dataclass" (a
    # spurious cross-file import that would fail at runtime because constants.py
    # never exports dataclass).
    source = (
        "from dataclasses import dataclass\n\n"
        "_CONST = 42\n\n"
        "def make():\n    return dataclass\n"
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["dataclass", "_CONST"])
    e_make = _make_entity("make", 5, 6)
    c = _classified(entities=[e_block, e_make])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["make"], target_file="utils.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    utils_src = result.new_files["utils.py"]
    # Must import dataclass from the stdlib, not from constants.py
    assert "from dataclasses import dataclass" in utils_src
    assert "from .constants import dataclass" not in utils_src


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


def test_strip_top_level_import_lines_removes_imports():
    src = "import os\nfrom typing import List\n\n_CONST = 1\n"
    result = _strip_top_level_import_lines(src)
    assert "import os" not in result
    assert "from typing import List" not in result
    assert "_CONST = 1" in result


def test_strip_top_level_import_lines_no_imports():
    src = "_CONST = 1\n"
    assert _strip_top_level_import_lines(src) == src


def test_strip_top_level_import_lines_syntax_error():
    src = "def (\n"
    assert _strip_top_level_import_lines(src) == src


def test_strip_top_level_import_lines_strips_type_checking_block():
    # `if TYPE_CHECKING:` blocks must be stripped so that their imports are
    # not emitted verbatim in sub-files (wrong path, wrong file).
    src = "if TYPE_CHECKING:\n" "    from .config import MyConfig\n" "\n" "_CONST = 1\n"
    result = _strip_top_level_import_lines(src)
    assert "TYPE_CHECKING" not in result
    assert "MyConfig" not in result
    assert "_CONST = 1" in result
