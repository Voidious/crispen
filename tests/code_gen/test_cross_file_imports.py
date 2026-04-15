from __future__ import annotations
from crispen.file_limiter.code_gen import (
    _find_cross_file_imports,
    _find_cross_file_type_checking_imports,
    _module_import_stmt,
    _relative_import_prefix,
    _target_module_name,
)


def test_target_module_name_simple():
    assert _target_module_name("utils.py") == "utils"


def test_target_module_name_nested():
    assert _target_module_name("helpers/io.py") == "helpers.io"


def test_target_module_name_init():
    # __init__.py represents the package, not a "__init__" submodule.
    assert _target_module_name("pkg/__init__.py") == "pkg"


def test_find_cross_file_imports_basic():
    # fn_a references _MODEL which is defined in block_1.py
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "llm_extract.py"
    )
    assert from_imports == ["from .block_1 import _MODEL"]
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_same_file_excluded():
    # _MODEL goes to the same file as fn_a → no cross-file import needed
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "llm_extract.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "llm_extract.py"
    )
    assert from_imports == []
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_no_match():
    # Referenced name not in name_to_target_file → no cross-file import
    entity_source_map = {"fn_a": "def fn_a():\n    return os.getcwd()\n"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, {}, "utils.py"
    )
    assert from_imports == []
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_entity_not_in_map():
    # Entity name not in entity_source_map → treated as empty source, no imports
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["ghost"], {}, {"x": "other.py"}, "utils.py"
    )
    assert from_imports == []
    assert module_imports == []
    assert rewrites == {}


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


def test_relative_import_prefix_to_init_same_dir():
    # to_file is __init__.py in the same directory → "." (the package itself).
    assert _relative_import_prefix("a.py", "__init__.py") == "."


def test_relative_import_prefix_to_init_same_subdir():
    # Both in sub/, to_file is sub/__init__.py → "." (the package itself).
    assert _relative_import_prefix("sub/a.py", "sub/__init__.py") == "."


def test_find_cross_file_imports_cross_directory():
    # fn_a is in tests/test.py; helper is in helpers/entities.py.
    # Cross-directory import needs ".." to go up from tests/ to root.
    entity_source_map = {"fn_a": "def fn_a():\n    return _helper()\n"}
    name_to_target_file = {"_helper": "helpers/entities.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "tests/test.py"
    )
    assert from_imports == ["from ..helpers.entities import _helper"]
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_top_level_var_uses_module_import():
    # SAFE_MODE is a TOP_LEVEL variable in conversion.py; runtime.py references it.
    # Should produce a module-level import (from . import conversion) in
    # module_imports, not a direct name import, so that later mutations to the
    # variable propagate correctly.
    entity_source_map = {
        "create_lua_runtime": (
            "def create_lua_runtime(safe_mode=None):\n"
            "    if safe_mode is None:\n"
            "        safe_mode = SAFE_MODE\n"
        )
    }
    name_to_target_file = {"SAFE_MODE": "conversion.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["create_lua_runtime"],
        entity_source_map,
        name_to_target_file,
        "runtime.py",
        top_level_var_names={"SAFE_MODE"},
    )
    assert from_imports == []
    assert module_imports == ["from . import conversion"]
    assert rewrites == {"SAFE_MODE": "conversion.SAFE_MODE"}


def test_find_cross_file_imports_top_level_var_abs_pkg():
    # Same as above but with abs_pkg set (test-file context).
    # Uses "import pkg.module as local" syntax to avoid test-name misclassification.
    entity_source_map = {"fn_a": "def fn_a():\n    return SAFE_MODE\n"}
    name_to_target_file = {"SAFE_MODE": "conversion.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "test_fn.py",
        abs_pkg="mylib",
        top_level_var_names={"SAFE_MODE"},
    )
    assert from_imports == []
    assert module_imports == ["import mylib.conversion as conversion"]
    assert rewrites == {"SAFE_MODE": "conversion.SAFE_MODE"}


def test_find_cross_file_imports_top_level_var_abs_pkg_empty():
    # abs_pkg="" (root-level test) — no package prefix, plain "import conversion".
    entity_source_map = {"fn_a": "def fn_a():\n    return SAFE_MODE\n"}
    name_to_target_file = {"SAFE_MODE": "conversion.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "test_fn.py",
        abs_pkg="",
        top_level_var_names={"SAFE_MODE"},
    )
    assert from_imports == []
    assert module_imports == ["import conversion"]
    assert rewrites == {"SAFE_MODE": "conversion.SAFE_MODE"}


def test_find_cross_file_imports_top_level_var_cross_directory():
    # TOP_LEVEL var in sub/constants.py, referenced from runtime.py at root.
    entity_source_map = {"fn_a": "def fn_a():\n    return TIMEOUT\n"}
    name_to_target_file = {"TIMEOUT": "sub/constants.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "runtime.py",
        top_level_var_names={"TIMEOUT"},
    )
    assert from_imports == []
    assert module_imports == ["from .sub import constants"]
    assert rewrites == {"TIMEOUT": "constants.TIMEOUT"}


def test_find_cross_file_imports_mixed_top_level_and_function():
    # SAFE_MODE is a TOP_LEVEL var; _helper is a function — mixed case.
    entity_source_map = {
        "fn_a": ("def fn_a():\n" "    if SAFE_MODE:\n" "        return _helper()\n")
    }
    name_to_target_file = {"SAFE_MODE": "conversion.py", "_helper": "helpers.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "runtime.py",
        top_level_var_names={"SAFE_MODE"},
    )
    assert from_imports == ["from .helpers import _helper"]
    assert module_imports == ["from . import conversion"]
    assert rewrites == {"SAFE_MODE": "conversion.SAFE_MODE"}


def test_module_import_stmt_sibling_relative():
    stmt, local = _module_import_stmt("runtime.py", "conversion.py", abs_pkg=None)
    assert stmt == "from . import conversion"
    assert local == "conversion"


def test_module_import_stmt_cross_directory_relative():
    stmt, local = _module_import_stmt("runtime.py", "sub/constants.py", abs_pkg=None)
    assert stmt == "from .sub import constants"
    assert local == "constants"


def test_module_import_stmt_parent_directory_relative():
    # svc/test_fns.py importing from test_svc.py (parent dir)
    stmt, local = _module_import_stmt("svc/test_fns.py", "test_svc.py", abs_pkg=None)
    assert stmt == "from .. import test_svc"
    assert local == "test_svc"


def test_module_import_stmt_abs_pkg_with_prefix():
    # Uses "import pkg.module as local" to avoid test-name collision.
    stmt, local = _module_import_stmt("test_fn.py", "conversion.py", abs_pkg="mylib")
    assert stmt == "import mylib.conversion as conversion"
    assert local == "conversion"


def test_module_import_stmt_abs_pkg_empty():
    # No package prefix → plain "import conversion".
    stmt, local = _module_import_stmt("test_fn.py", "conversion.py", abs_pkg="")
    assert stmt == "import conversion"
    assert local == "conversion"


def test_module_import_stmt_abs_pkg_nested_module():
    # source_file has a nested path within the package
    stmt, local = _module_import_stmt("test_fn.py", "sub/constants.py", abs_pkg="mylib")
    assert stmt == "import mylib.sub.constants as constants"
    assert local == "constants"


def test_find_cross_file_imports_abs_pkg_package_prefix():
    # abs_pkg="tests" → "from tests.block_1 import _MODEL"
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "test_fn.py", abs_pkg="tests"
    )
    assert from_imports == ["from tests.block_1 import _MODEL"]
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_abs_pkg_root_level():
    # abs_pkg="" → "from block_1 import _MODEL" (no package prefix)
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "test_fn.py", abs_pkg=""
    )
    assert from_imports == ["from block_1 import _MODEL"]
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_type_checking_imports_basic():
    # _LLMAccumulator appears only in a quoted annotation in fn_a.
    # It lives in block_1.py — a TYPE_CHECKING import should be generated.
    entity_source_map = {
        "fn_a": 'def fn_a(x: Optional["_LLMAccumulator"]) -> None:\n    pass\n'
    }
    name_to_target_file = {"_LLMAccumulator": "block_1.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "placements.py"
    )
    assert result == ["from .block_1 import _LLMAccumulator"]


def test_find_cross_file_type_checking_imports_same_file_excluded():
    # _LLMAccumulator goes to the same target file — no import needed.
    entity_source_map = {
        "fn_a": 'def fn_a(x: Optional["_LLMAccumulator"]) -> None:\n    pass\n'
    }
    name_to_target_file = {"_LLMAccumulator": "placements.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "placements.py"
    )
    assert result == []


def test_find_cross_file_type_checking_imports_runtime_excluded():
    # _LLMAccumulator is used at runtime (not just annotation) — excluded.
    entity_source_map = {"fn_a": "def fn_a():\n    return _LLMAccumulator()\n"}
    name_to_target_file = {"_LLMAccumulator": "block_1.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "placements.py"
    )
    assert result == []


def test_find_cross_file_type_checking_imports_no_annotations():
    # No quoted annotations at all → empty result.
    entity_source_map = {"fn_a": "def fn_a():\n    pass\n"}
    name_to_target_file = {"_LLMAccumulator": "block_1.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "placements.py"
    )
    assert result == []


def test_find_cross_file_type_checking_imports_not_in_map():
    # Referenced quoted name not in name_to_target_file → no import.
    entity_source_map = {"fn_a": 'def fn_a(x: "UnknownType") -> None:\n    pass\n'}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, {}, "placements.py"
    )
    assert result == []


def test_find_cross_file_type_checking_imports_top_level_var_excluded():
    # A name in top_level_var_names is skipped (handled separately).
    entity_source_map = {
        "fn_a": 'def fn_a(x: Optional["SAFE_MODE"]) -> None:\n    pass\n'
    }
    name_to_target_file = {"SAFE_MODE": "constants.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "placements.py",
        top_level_var_names={"SAFE_MODE"},
    )
    assert result == []


def test_find_cross_file_type_checking_imports_abs_pkg():
    # With abs_pkg set, use absolute import style.
    entity_source_map = {
        "fn_a": 'def fn_a(x: Optional["_LLMAccumulator"]) -> None:\n    pass\n'
    }
    name_to_target_file = {"_LLMAccumulator": "block_1.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "test_fn.py",
        abs_pkg="tests",
    )
    assert result == ["from tests.block_1 import _LLMAccumulator"]


def test_find_cross_file_type_checking_imports_entity_not_in_map():
    # Entity not in entity_source_map → treated as empty, no imports.
    result = _find_cross_file_type_checking_imports(
        ["ghost"], {}, {"_X": "other.py"}, "placements.py"
    )
    assert result == []
