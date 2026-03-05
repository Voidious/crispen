from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    _abs_package_for_dir,
    _find_cross_file_imports,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from tests.entity_helpers import _classified, _make_entity
from tests.plan_helpers import _plan


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
