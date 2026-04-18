from unittest.mock import patch
from crispen.engine import (
    _build_alias_map,
    _build_patch_map,
    _compute_qname,
    _file_to_module,
    _find_repo_root,
)
from crispen.file_limiter.runner import FileLimiterResult


def test_find_repo_root_finds_git(tmp_path):
    (tmp_path / ".git").mkdir()
    subdir = tmp_path / "src"
    subdir.mkdir()
    f = subdir / "code.py"
    f.write_text("x = 1\n")
    root = _find_repo_root({str(f): [(1, 1)]})
    assert root == str(tmp_path)


def test_find_repo_root_not_found(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n")
    root = _find_repo_root({str(f): [(1, 1)]})
    assert root is None


def test_file_to_module_regular_file(tmp_path):
    f = tmp_path / "mypkg" / "service.py"
    f.parent.mkdir()
    f.write_text("x = 1\n")
    assert _file_to_module(str(tmp_path), str(f)) == "mypkg.service"


def test_file_to_module_init(tmp_path):
    f = tmp_path / "mypkg" / "__init__.py"
    f.parent.mkdir()
    f.write_text("")
    assert _file_to_module(str(tmp_path), str(f)) == "mypkg"


def test_compute_qname(tmp_path):
    f = tmp_path / "pkg" / "mod.py"
    f.parent.mkdir()
    f.write_text("")
    assert _compute_qname(str(tmp_path), str(f), "my_func") == "pkg.mod.my_func"


def test_build_alias_map_identity_only(tmp_path):
    # No __init__.py in tmp_path → only identity mapping returned.
    alias_map = _build_alias_map(str(tmp_path), {"a.b.func"})
    assert alias_map == {"a.b.func": "a.b.func"}


def test_build_alias_map_with_reexport(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mypkg.service import get_user\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    assert "mypkg.get_user" in alias_map
    assert alias_map["mypkg.get_user"] == "mypkg.service.get_user"


def test_build_alias_map_star_import_skipped(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mypkg.service import *\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    # Star import does not create an alias
    assert "mypkg.get_user" not in alias_map


def test_build_alias_map_ambiguous_name_skipped(tmp_path):
    # Two canonical qnames share the same function name → alias is ambiguous.
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mypkg.service import get_user\n")
    alias_map = _build_alias_map(
        str(tmp_path),
        {"mypkg.service.get_user", "mypkg.other.get_user"},
    )
    # Ambiguous: skip adding the alias
    assert "mypkg.get_user" not in alias_map


def test_build_alias_map_invalid_init_skipped(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def f(:\n    pass\n")  # invalid Python
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    # Gracefully skips the unreadable __init__.py
    assert alias_map == {"mypkg.service.get_user": "mypkg.service.get_user"}


def test_build_alias_map_skips_compound_statement(tmp_path):
    # A function definition is a compound statement, not SimpleStatementLine (line 76).
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def helper():\n    pass\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    assert alias_map == {"mypkg.service.get_user": "mypkg.service.get_user"}


def test_build_alias_map_skips_non_import_in_simple_stmt(tmp_path):
    # An assignment in SimpleStatementLine is not ImportFrom (line 79).
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("__version__ = '1.0'\n")
    alias_map = _build_alias_map(str(tmp_path), {"mypkg.service.get_user"})
    assert alias_map == {"mypkg.service.get_user": "mypkg.service.get_user"}


def test_build_patch_map_import_alias_single_importer(tmp_path):
    """Import alias from original used in exactly one new file → added."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "module.py"
    pre_split = "from external import Helper\ndef MyFunc(): pass\n"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "sub.py": "def MyFunc(): pass\n",
            "utils.py": "from external import Helper\nHelper()\n",
        },
        entity_to_target={"MyFunc": "sub.py"},
    )
    result = _build_patch_map(str(f), fl_result, pkg, pre_split)
    assert result["mypkg.module.Helper"] == "mypkg.utils.Helper"
    assert result["mypkg.module.MyFunc"] == "mypkg.sub.MyFunc"


def test_build_patch_map_import_alias_skips_entity_names(tmp_path):
    """Import alias that is also an entity name is not double-processed."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "module.py"
    # "Helper" appears both as entity_to_target key and in pre_split imports
    pre_split = "from external import Helper\n"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={"utils.py": "from external import Helper\n"},
        entity_to_target={"Helper": "utils.py"},
    )
    result = _build_patch_map(str(f), fl_result, pkg, pre_split)
    # Entity loop handles Helper (definer=utils.py, no external callers → utils.py)
    assert result == {"mypkg.module.Helper": "mypkg.utils.Helper"}


def test_build_patch_map_import_alias_module_none(tmp_path):
    """Import alias target module can't be resolved → alias is skipped."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "module.py"
    pre_split = "from external import Helper\ndef MyFunc(): pass\n"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "sub.py": "def MyFunc(): pass\n",
            "utils.py": "from external import Helper\nHelper()\n",
        },
        entity_to_target={"MyFunc": "sub.py"},
    )
    # Third call (for alias importer utils.py) returns None
    with patch(
        "crispen.engine.file_limiter._module_path_for_file",
        side_effect=["mypkg.module", "mypkg.sub", None],
    ):
        result = _build_patch_map(str(f), fl_result, pkg, pre_split)
    # MyFunc was added (second call succeeded); Helper was skipped (third → None)
    assert result == {"mypkg.module.MyFunc": "mypkg.sub.MyFunc"}
    assert "mypkg.module.Helper" not in result


def test_build_patch_map_import_only_caller_falls_back_to_definer(tmp_path):
    """Entity imported but not used by any new file → falls back to definer."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "module.py"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "sub.py": "def MyFunc(): pass\n",
            "__init__.py": "from .sub import MyFunc\n",
        },
        entity_to_target={"MyFunc": "sub.py"},
    )
    result = _build_patch_map(str(f), fl_result, pkg)
    # __init__.py only re-exports (no Load usage) → 0 real callers → fall back to sub.py
    assert result == {"mypkg.module.MyFunc": "mypkg.sub.MyFunc"}


def test_build_patch_map_reexport_ignored_real_caller_wins(tmp_path):
    """Re-export stub ignored; the one file that actually calls the entity is used."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "module.py"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "sub.py": "def MyFunc(): pass\n",
            "caller.py": "from .sub import MyFunc\nMyFunc()\n",
            "__init__.py": "from .sub import MyFunc\n",
        },
        entity_to_target={"MyFunc": "sub.py"},
    )
    result = _build_patch_map(str(f), fl_result, pkg)
    # __init__.py has no Load usage; caller.py does → single real caller
    assert result == {"mypkg.module.MyFunc": "mypkg.caller.MyFunc"}


def test_build_patch_map_init_real_usage_counted_as_caller(tmp_path):
    """__init__.py that actually calls an entity is counted as a real caller.

    The module path strips .__init__ so the patch target is the public
    package namespace (mypkg.MyFunc, not mypkg.__init__.MyFunc).
    """
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "module.py"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "sub.py": "def MyFunc(): pass\n",
            "__init__.py": "from .sub import MyFunc\n_x = MyFunc()\n",
        },
        entity_to_target={"MyFunc": "sub.py"},
    )
    result = _build_patch_map(str(f), fl_result, pkg)
    # __init__.py has a Load reference → real caller; .__init__ stripped from path
    assert result == {"mypkg.module.MyFunc": "mypkg.MyFunc"}


def test_build_patch_map_init_real_usage_plus_other_caller_forks(tmp_path):
    """__init__.py calling entity + another caller → forking → skipped."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "module.py"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "sub.py": "def MyFunc(): pass\n",
            "caller.py": "from .sub import MyFunc\nMyFunc()\n",
            "__init__.py": "from .sub import MyFunc\nMyFunc()\n",
        },
        entity_to_target={"MyFunc": "sub.py"},
    )
    result = _build_patch_map(str(f), fl_result, pkg)
    # 2 real callers (caller.py + __init__.py) → forking → skipped
    assert "mypkg.module.MyFunc" not in result


def test_build_patch_map_import_alias_reexport_stub_skipped(tmp_path):
    """Import alias whose only importer is a re-export stub is skipped."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "module.py"
    pre_split = "from external import Helper\ndef F(): pass\n"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "sub.py": "def F(): pass\n",
            "__init__.py": "from external import Helper\n",
        },
        entity_to_target={"F": "sub.py"},
    )
    result = _build_patch_map(str(f), fl_result, pkg, pre_split)
    # __init__.py imports Helper but has no Load usage → 0 real importers → skipped
    assert "mypkg.module.Helper" not in result


def test_build_patch_map_assignment_no_callers(tmp_path):
    """Module-level variable in original file, only used in its defining new file."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    pre_split = "_TIMEOUT = 30\ndef run(): pass\n"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "core.py": "_TIMEOUT = 30\ndef run(): pass\n",
        },
        abort=False,
        entity_to_target={"run": "core.py"},
    )
    result = _build_patch_map(str(pkg / "big.py"), fl_result, pkg, pre_split)
    # 0 callers → _TIMEOUT stays in its definer core.py
    assert result["mypkg.big._TIMEOUT"] == "mypkg.core._TIMEOUT"


def test_build_patch_map_assignment_single_caller(tmp_path):
    """Variable defined in one new file, imported and used by exactly one other."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    pre_split = "_TIMEOUT = 30\n"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "core.py": "_TIMEOUT = 30\n",
            "runner.py": "from .core import _TIMEOUT\nif _TIMEOUT > 0: pass\n",
        },
        abort=False,
        entity_to_target={},
    )
    result = _build_patch_map(str(pkg / "big.py"), fl_result, pkg, pre_split)
    # runner.py imports and uses _TIMEOUT → single caller
    assert result["mypkg.big._TIMEOUT"] == "mypkg.runner._TIMEOUT"


def test_build_patch_map_assignment_forking_skipped(tmp_path):
    """Variable imported and used by two new files → forking → skipped."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    pre_split = "_TIMEOUT = 30\n"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "core.py": "_TIMEOUT = 30\n",
            "a.py": "from .core import _TIMEOUT\nif _TIMEOUT: pass\n",
            "b.py": "from .core import _TIMEOUT\nif _TIMEOUT: pass\n",
        },
        abort=False,
        entity_to_target={},
    )
    result = _build_patch_map(str(pkg / "big.py"), fl_result, pkg, pre_split)
    assert "mypkg.big._TIMEOUT" not in result


def test_build_patch_map_assignment_already_in_patch_map_skipped(tmp_path):
    """Variable in patch_map from import-alias section → assignment section skips it."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    # pre_split both imports and assigns _TIMEOUT → import-alias section maps it first.
    pre_split = "from ext import _TIMEOUT\n_TIMEOUT = 30\n"
    fl_result = FileLimiterResult(
        original_source="",
        # core.py imports, assigns, and uses _TIMEOUT: alias + assignment.
        new_files={
            "core.py": (
                "from ext import _TIMEOUT\n_TIMEOUT = 30\nif _TIMEOUT > 0: pass\n"
            )
        },
        abort=False,
        entity_to_target={},
    )
    result = _build_patch_map(str(pkg / "big.py"), fl_result, pkg, pre_split)
    # Import-alias section mapped it; assignment section hits old_path in patch_map.
    assert result["mypkg.big._TIMEOUT"] == "mypkg.core._TIMEOUT"
    # Verify mapped exactly once (assignment section did NOT add a duplicate).
    assert list(result.values()).count("mypkg.core._TIMEOUT") == 1


def test_build_patch_map_assignment_new_module_none_skipped(tmp_path):
    """When _module_path_for_file returns None for the target, entry is skipped."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    pre_split = "_TIMEOUT = 30\n"
    fl_result = FileLimiterResult(
        original_source="",
        # Target path cannot be resolved to a module (no pyproject.toml ancestor).
        new_files={"/unresolvable/abs/path.py": "_TIMEOUT = 30\n"},
        abort=False,
        entity_to_target={},
    )
    result = _build_patch_map(str(pkg / "big.py"), fl_result, pkg, pre_split)
    assert "mypkg.big._TIMEOUT" not in result
