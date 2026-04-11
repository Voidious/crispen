from unittest.mock import patch
from crispen.engine import _build_patch_map
from crispen.file_limiter.runner import FileLimiterResult


def test_build_patch_map_empty_entity_to_target(tmp_path):
    """No entity_to_target → empty map."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    f = tmp_path / "module.py"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={},
        entity_to_target={},
    )
    result = _build_patch_map(str(f), fl_result, tmp_path)
    assert result == {}


def test_build_patch_map_no_old_module(tmp_path):
    """When _module_path_for_file returns None for filepath → empty map."""
    # No pyproject.toml anywhere → cannot find project root
    f = tmp_path / "module.py"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={"utils.py": "class MyClass: pass\n"},
        entity_to_target={"MyClass": "utils.py"},
    )
    result = _build_patch_map(str(f), fl_result, tmp_path)
    assert result == {}


def test_build_patch_map_no_callers_uses_definer(tmp_path):
    """Entity with no callers maps to its definition file."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "module.py"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={"utils.py": "class MyClass: pass\n"},
        entity_to_target={"MyClass": "utils.py"},
    )
    result = _build_patch_map(str(f), fl_result, pkg)
    assert result == {"mypkg.module.MyClass": "mypkg.utils.MyClass"}


def test_build_patch_map_single_caller_uses_caller(tmp_path):
    """Entity imported and used by exactly one new file → caller's module used."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "module.py"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "sub.py": "def MyFunc(): pass\n",
            "caller.py": "from .sub import MyFunc\nMyFunc()\n",
        },
        entity_to_target={"MyFunc": "sub.py"},
    )
    result = _build_patch_map(str(f), fl_result, pkg)
    assert result == {"mypkg.module.MyFunc": "mypkg.caller.MyFunc"}


def test_build_patch_map_forking_entity_skipped(tmp_path):
    """Entity used by multiple new files (forking) → skipped."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "module.py"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "sub.py": "def MyFunc(): pass\n",
            "caller_a.py": "from .sub import MyFunc\nMyFunc()\n",
            "caller_b.py": "from .sub import MyFunc\nMyFunc()\n",
        },
        entity_to_target={"MyFunc": "sub.py"},
    )
    result = _build_patch_map(str(f), fl_result, pkg)
    assert result == {}


def test_build_patch_map_empty_new_file_skipped(tmp_path):
    """New file with empty source is skipped when building import index."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "module.py"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "sub.py": "class MyClass: pass\n",
            "empty.py": "",
        },
        entity_to_target={"MyClass": "sub.py"},
    )
    result = _build_patch_map(str(f), fl_result, pkg)
    assert result == {"mypkg.module.MyClass": "mypkg.sub.MyClass"}


def test_build_patch_map_new_module_none(tmp_path):
    """When target file's module path can't be resolved → entity is skipped."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    f = tmp_path / "module.py"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={"utils.py": "class MyClass: pass\n"},
        entity_to_target={"MyClass": "utils.py"},
    )
    with patch(
        "crispen.engine.file_limiter._module_path_for_file",
        side_effect=["mypkg.module", None],
    ):
        result = _build_patch_map(str(f), fl_result, tmp_path)
    assert result == {}


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


def test_build_patch_map_import_alias_forking_skipped(tmp_path):
    """Import alias used in zero or multiple new files is skipped."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "module.py"
    pre_split = (
        "from external import Forked\nfrom external import Nowhere\ndef F(): pass\n"
    )
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "sub.py": "from external import Forked\nForked()\ndef F(): pass\n",
            "utils.py": "from external import Forked\nForked()\n",
        },
        entity_to_target={"F": "sub.py"},
    )
    result = _build_patch_map(str(f), fl_result, pkg, pre_split)
    # Forked used in 2 files → forking → skipped; Nowhere used in 0 files → skipped
    assert "mypkg.module.Forked" not in result
    assert "mypkg.module.Nowhere" not in result


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


def test_build_patch_map_assignment_defined_in_multiple_files_skipped(tmp_path):
    """Variable appearing in two new files' assignments → ambiguous → skipped."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    pre_split = "_TIMEOUT = 30\n"
    fl_result = FileLimiterResult(
        original_source="",
        new_files={
            "core.py": "_TIMEOUT = 30\n",
            "utils.py": "_TIMEOUT = 60\n",
        },
        abort=False,
        entity_to_target={},
    )
    result = _build_patch_map(str(pkg / "big.py"), fl_result, pkg, pre_split)
    assert "mypkg.big._TIMEOUT" not in result


def test_build_patch_map_assignment_not_in_original_skipped(tmp_path):
    """Variable introduced by code generation (not in pre_split_source) → skipped."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    pre_split = "def run(): pass\n"  # _TIMEOUT not in original
    fl_result = FileLimiterResult(
        original_source="",
        new_files={"core.py": "_TIMEOUT = 30\ndef run(): pass\n"},
        abort=False,
        entity_to_target={"run": "core.py"},
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
