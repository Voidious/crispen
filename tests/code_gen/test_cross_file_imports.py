from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    _abs_package_for_dir,
    _add_re_exports,
    _find_cross_file_imports,
    _relative_import_prefix,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .helpers import _classified, _make_entity, _plan


def test_find_cross_file_imports_basic():
    # fn_a references _MODEL which is defined in block_1.py
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "llm_extract.py"
    )
    assert result == ["from .block_1 import _MODEL"]


def test_find_cross_file_imports_same_file_excluded():
    # _MODEL goes to the same file as fn_a → no cross-file import needed
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "llm_extract.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "llm_extract.py"
    )
    assert result == []


def test_find_cross_file_imports_no_match():
    # Referenced name not in name_to_target_file → no cross-file import
    entity_source_map = {"fn_a": "def fn_a():\n    return os.getcwd()\n"}
    result = _find_cross_file_imports(["fn_a"], entity_source_map, {}, "utils.py")
    assert result == []


def test_find_cross_file_imports_entity_not_in_map():
    # Entity name not in entity_source_map → treated as empty source, no imports
    result = _find_cross_file_imports(["ghost"], {}, {"x": "other.py"}, "utils.py")
    assert result == []


def test_relative_import_prefix_same_directory():
    # Both files at the root level → single dot.
    assert _relative_import_prefix("a.py", "b.py") == ".b"


def test_relative_import_prefix_sibling_subdir():
    # from_file is in sub/, to_file is in helpers/ → go up one, then down.
    assert _relative_import_prefix("sub/a.py", "helpers/b.py") == "..helpers.b"


def test_relative_import_prefix_same_subdir():
    # Both in the same subdirectory → single dot.
    assert _relative_import_prefix("sub/a.py", "sub/b.py") == ".b"


def test_relative_import_prefix_to_nested():
    # to_file is in a subdirectory of root while from_file is at root.
    assert _relative_import_prefix("a.py", "helpers/b.py") == ".helpers.b"


def test_find_cross_file_imports_cross_directory():
    # fn_a is in tests/test.py; helper is in helpers/entities.py.
    # Cross-directory import needs ".." to go up from tests/ to root.
    entity_source_map = {"fn_a": "def fn_a():\n    return _helper()\n"}
    name_to_target_file = {"_helper": "helpers/entities.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "tests/test.py"
    )
    assert result == ["from ..helpers.entities import _helper"]


def test_generate_cross_file_import():
    # fn_a goes to fn_module.py; _block_1 (defining _CONST) goes to constants.py.
    # fn_a references _CONST → fn_module.py must have `from .constants import _CONST`.
    source = "_CONST = 42\n\ndef fn_a():\n    return _CONST\n"
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_fn = _make_entity("fn_a", 3, 4)
    c = _classified(entities=[e_block, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["fn_a"], target_file="fn_module.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    fn_src = result.new_files["fn_module.py"]
    assert "from .constants import _CONST" in fn_src
    # constants.py should NOT have a cross-import (it defines _CONST, not uses it)
    const_src = result.new_files["constants.py"]
    assert "from .fn_module" not in const_src


def test_find_cross_file_imports_abs_pkg_package_prefix():
    # abs_pkg="tests" → "from tests.block_1 import _MODEL"
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "test_fn.py", abs_pkg="tests"
    )
    assert result == ["from tests.block_1 import _MODEL"]


def test_find_cross_file_imports_abs_pkg_root_level():
    # abs_pkg="" → "from block_1 import _MODEL" (no package prefix)
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    result = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "test_fn.py", abs_pkg=""
    )
    assert result == ["from block_1 import _MODEL"]


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


def test_generate_test_file_reexports_use_absolute_imports(tmp_path):
    # When the original is a test file, re-exports in the updated original
    # must use absolute imports so pytest can load the file.
    (tmp_path / "pyproject.toml").touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_engine.py"
    test_file.write_text("")

    source = "import os\n\ndef foo():\n    os.getcwd()\n"
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="test_helpers.py")])

    result = generate_file_splits(c, plan, source, str(test_file))

    assert "from tests.test_helpers import foo" in result.original_source
    assert "from .test_helpers import foo" not in result.original_source


def test_generate_test_file_cross_imports_use_absolute_imports(tmp_path):
    # Cross-file imports in generated test split files must also be absolute.
    (tmp_path / "pyproject.toml").touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_engine.py"
    test_file.write_text("")

    source = "_CONST = 42\n\ndef test_fn():\n    return _CONST\n"
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_fn = _make_entity("test_fn", 3, 4)
    c = _classified(entities=[e_block, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="test_constants.py"),
            GroupPlacement(group=["test_fn"], target_file="test_fns.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, str(test_file))

    fn_src = result.new_files["test_fns.py"]
    assert "from tests.test_constants import _CONST" in fn_src
    assert "from .test_constants import _CONST" not in fn_src


def test_add_re_exports_relative_from_uses_relative_prefix():
    # When relative_from is set, imports are computed via _relative_import_prefix
    # rather than _target_module_name, so "service/__init__.py" → ".utils"
    # (not ".service.utils").
    source = "# stayed\n"
    entity = _make_entity("Foo", 1, 1)
    placements = [GroupPlacement(group=["Foo"], target_file="service/utils.py")]
    entity_map = {"Foo": entity}
    entity_source_map = {"Foo": "class Foo: pass"}

    result = _add_re_exports(
        source,
        placements,
        entity_map,
        entity_source_map,
        relative_from="service/__init__.py",
    )

    assert "from .utils import Foo" in result
    # Must NOT use the fully-qualified form that would be wrong from __init__.py.
    assert "from .service.utils" not in result
