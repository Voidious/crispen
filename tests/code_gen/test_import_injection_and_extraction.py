from __future__ import annotations
from crispen.file_limiter.code_gen import (
    _extract_import_info,
    _import_derived_names,
    _import_line_numbers,
    _inject_module_level_imports,
    _inject_type_checking_imports,
    _test_names_in_decorators,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind


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


def test_test_names_in_decorators_finds_name_in_decorator():
    src = (
        "@pytest.mark.parametrize('x', TestFixture.PARAMS)\ndef test_fn(x):\n    pass\n"
    )
    assert _test_names_in_decorators(src, {"TestFixture"}) == {"TestFixture"}


def test_test_names_in_decorators_name_only_in_body_not_found():
    src = "def test_fn():\n    TestFixture.setup()\n"
    assert _test_names_in_decorators(src, {"TestFixture"}) == set()


def test_test_names_in_decorators_syntax_error_returns_empty():
    assert _test_names_in_decorators("def (invalid", {"TestFixture"}) == set()


def test_test_names_in_decorators_class_decorator():
    src = "@TestFixture.mark\nclass TestSomething:\n    pass\n"
    assert _test_names_in_decorators(src, {"TestFixture"}) == {"TestFixture"}


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
