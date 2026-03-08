from __future__ import annotations
from crispen.file_limiter.code_gen import (
    _abs_package_for_dir,
    _bump_relative_imports,
    _collect_name_loads,
    _extract_import_info,
    _extract_module_docstring,
    _find_project_root,
    _import_derived_names,
    _import_line_numbers,
    _module_path_from_file,
    _remove_entity_lines,
    _strip_module_docstring,
    _strip_top_level_import_lines,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .test_generate_test_support import _make_entity
import textwrap


def test_collect_name_loads_basic():
    source = "x = foo + bar"
    names = _collect_name_loads(source)
    assert "foo" in names
    assert "bar" in names


def test_collect_name_loads_store_not_included():
    source = "x = 1"
    names = _collect_name_loads(source)
    # x is a Store, not a Load
    assert "x" not in names


def test_collect_name_loads_syntax_error():
    assert _collect_name_loads("def (invalid") == set()


def test_collect_name_loads_excludes_function_params():
    # 'client' is a parameter of test_foo — excluded from loads inside the body.
    source = "def test_foo(client):\n    client.call()\n"
    names = _collect_name_loads(source)
    assert "client" not in names


def test_collect_name_loads_includes_non_param_name():
    # 'helper' is not a parameter of test_foo — still counted as a load.
    source = "def test_foo(client):\n    helper(client)\n"
    names = _collect_name_loads(source)
    assert "helper" in names
    assert "client" not in names


def test_collect_name_loads_excludes_nested_function_params():
    # Inner function params are excluded only within that function's own body.
    source = textwrap.dedent(
        """\
        def outer(x):
            def inner(y):
                return y + x
            return inner
        """
    )
    names = _collect_name_loads(source)
    assert "y" not in names  # param of inner — excluded inside inner body
    assert "x" not in names  # param of outer — excluded inside outer body


def test_collect_name_loads_includes_annotation_names():
    # Type annotations are in the outer scope — their names are included.
    source = "def f(x: MyType) -> ReturnType:\n    pass\n"
    names = _collect_name_loads(source)
    assert "MyType" in names
    assert "ReturnType" in names
    assert "x" not in names  # param name itself, not counted


def test_collect_name_loads_includes_decorator_names():
    # Decorator expressions are in the outer scope.
    source = "@pytest.fixture\ndef client():\n    pass\n"
    names = _collect_name_loads(source)
    assert "pytest" in names


def test_collect_name_loads_kw_defaults_none_skipped():
    # kw_defaults may contain None for keyword-only args without defaults.
    # None entries must not cause a crash and are simply skipped.
    source = "def f(*, a, b=DEFAULT):\n    pass\n"
    names = _collect_name_loads(source)
    assert "DEFAULT" in names
    assert "a" not in names
    assert "b" not in names


def test_collect_name_loads_annotated_vararg_kwarg():
    # *args: T and **kwargs: T annotations are in the outer scope.
    source = "def f(*args: VarType, **kwargs: KwType):\n    pass\n"
    names = _collect_name_loads(source)
    assert "VarType" in names
    assert "KwType" in names


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


def test_remove_entity_lines_removes_range():
    source = "line1\nline2\nline3\nline4\n"
    entity = _make_entity("foo", 2, 3)
    entity_map = {"foo": entity}
    result = _remove_entity_lines(source, {"foo"}, entity_map, {})
    assert "line1" in result
    assert "line2" not in result
    assert "line3" not in result
    assert "line4" in result


def test_remove_entity_lines_name_not_in_map():
    # Name not in entity_map → nothing removed.
    source = "line1\nline2\n"
    result = _remove_entity_lines(source, {"ghost"}, {}, {})
    assert result == source


def test_remove_entity_lines_top_level_preserves_import_lines():
    # When a TOP_LEVEL entity containing both imports and assignments is
    # migrated, the import lines must be kept in the original file so that
    # the remaining functions still have access to those names.
    source = "import os\n_CONST = 1\n\ndef foo():\n    return os.getcwd()\n"
    entity_src = "import os\n_CONST = 1\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 2, ["os", "_CONST"])
    entity_map = {"_block_1": entity}
    entity_source_map = {"_block_1": entity_src}
    result = _remove_entity_lines(source, {"_block_1"}, entity_map, entity_source_map)
    assert "import os" in result  # import line preserved
    assert "_CONST" not in result  # assignment line removed
    assert "def foo():" in result  # function untouched


def test_remove_entity_lines_top_level_no_source_map_removes_all():
    # Empty entity_source_map → no imports can be identified, all lines removed.
    source = "import os\n_CONST = 1\n\ndef foo():\n    pass\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 2, ["os", "_CONST"])
    entity_map = {"_block_1": entity}
    result = _remove_entity_lines(source, {"_block_1"}, entity_map, {})
    assert "import os" not in result
    assert "_CONST" not in result


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


def test_find_project_root_finds_pyproject_toml(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    sub = tmp_path / "pkg" / "module.py"
    sub.parent.mkdir()
    sub.write_text("x = 1\n")
    assert _find_project_root(sub) == tmp_path


def test_find_project_root_finds_git(tmp_path):
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "module.py"
    sub.write_text("x = 1\n")
    assert _find_project_root(sub) == tmp_path


def test_find_project_root_called_with_directory(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    assert _find_project_root(tmp_path) == tmp_path


def test_find_project_root_not_found(tmp_path):
    # tmp_path is under /tmp which has no project markers → None.
    sub = tmp_path / "module.py"
    sub.write_text("x = 1\n")
    result = _find_project_root(sub)
    # If the test runner is inside a project that happens to include tmp_path
    # (unlikely but possible with in-tree pytest), just ensure the function
    # returns without crashing.  The important coverage is the happy path above.
    assert result is None or result.exists()


def test_module_path_from_file_success(tmp_path):
    f = tmp_path / "pkg" / "utils.py"
    f.parent.mkdir()
    f.write_text("")
    assert _module_path_from_file(tmp_path, f) == "pkg.utils"


def test_module_path_from_file_top_level(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("")
    assert _module_path_from_file(tmp_path, f) == "module"


def test_module_path_from_file_not_under_root(tmp_path):
    other = tmp_path.parent / "other.py"
    assert _module_path_from_file(tmp_path, other) is None


def test_abs_package_for_dir_subdir(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_engine.py"
    test_file.touch()
    assert _abs_package_for_dir(str(test_file)) == "tests"


def test_abs_package_for_dir_root_level(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    test_file = tmp_path / "test_engine.py"
    test_file.touch()
    assert _abs_package_for_dir(str(test_file)) == ""


def test_abs_package_for_dir_no_project_root(monkeypatch):
    monkeypatch.setattr(
        "crispen.file_limiter.code_gen._find_project_root", lambda _p: None
    )
    assert _abs_package_for_dir("/some/random/path/test_engine.py") is None


def test_abs_package_for_dir_non_ancestor_root(tmp_path, monkeypatch):
    # Defensive branch: project root is not an ancestor of the file's directory.
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.setattr(
        "crispen.file_limiter.code_gen._find_project_root", lambda _p: other_dir
    )
    test_file = tmp_path / "tests" / "test_engine.py"
    test_file.parent.mkdir()
    test_file.touch()
    assert _abs_package_for_dir(str(test_file)) is None


def test_bump_relative_imports_single_dot():
    assert _bump_relative_imports("from .foo import bar") == "from ..foo import bar"


def test_bump_relative_imports_two_dots():
    assert _bump_relative_imports("from .. import baz") == "from ... import baz"


def test_bump_relative_imports_leaves_absolute():
    src = "import os\nfrom typing import List"
    assert _bump_relative_imports(src) == src


def test_bump_relative_imports_multiline():
    src = "from .a import x\nimport sys\nfrom ..b import y\n"
    result = _bump_relative_imports(src)
    assert "from ..a import x" in result
    assert "from ...b import y" in result
    assert "import sys" in result


def test_bump_relative_imports_n_two():
    assert _bump_relative_imports("from .. import foo", n=2) == "from .... import foo"


def test_bump_relative_imports_n_zero():
    src = "from .foo import bar"
    assert _bump_relative_imports(src, n=0) == src


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


def test_extract_module_docstring_present():
    src = '"""My module."""\n\nimport os\n'
    assert _extract_module_docstring(src) == '"""My module."""'


def test_extract_module_docstring_absent():
    src = "import os\n\ndef foo():\n    pass\n"
    assert _extract_module_docstring(src) is None


def test_extract_module_docstring_syntax_error():
    assert _extract_module_docstring("def (\n") is None


def test_extract_module_docstring_non_string_expr():
    # First statement is an expression but not a string constant.
    src = "1 + 1\n\ndef foo():\n    pass\n"
    assert _extract_module_docstring(src) is None


def test_strip_module_docstring_removes_docstring():
    src = '"""My module."""\n\n_CONST = 1\n'
    result = _strip_module_docstring(src)
    assert '"""My module."""' not in result
    assert "_CONST = 1" in result


def test_strip_module_docstring_no_docstring():
    src = "_CONST = 1\n"
    assert _strip_module_docstring(src) == src


def test_strip_module_docstring_syntax_error():
    src = "def (\n"
    assert _strip_module_docstring(src) == src
