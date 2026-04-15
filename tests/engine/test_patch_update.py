from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import _add_fl_context, _build_patch_map, run_engine
from crispen.file_limiter.runner import FileLimiterResult
from crispen.stats import RunStats
from .file_limiter import _FL_PATCH
from .patch_update import _CG_PATCH, _REWRITE_PATCH, _make_fl_result_with_entities


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


def test_patch_update_ignore_mode(tmp_path):
    """Default 'ignore' mode → @patch strings are never updated."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    other = tmp_path / "test_other.py"
    other.write_text(
        '@patch("mypkg.big.MyClass")\ndef test_it(): pass\n', encoding="utf-8"
    )

    fl_result = _make_fl_result_with_entities()
    with patch(_FL_PATCH, return_value=fl_result):
        list(
            run_engine(
                {str(f): [(1, 10)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    file_limiter_patch_update="ignore",
                ),
                _repo_root=str(tmp_path),
            )
        )
    # test_other.py should be unchanged
    assert (
        other.read_text(encoding="utf-8")
        == '@patch("mypkg.big.MyClass")\ndef test_it(): pass\n'
    )


def test_patch_update_no_combined_map(tmp_path):
    """'update' mode but FL returned empty entity_to_target → no updates."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    other = tmp_path / "test_other.py"
    other.write_text(
        '@patch("mypkg.big.MyClass")\ndef test_it(): pass\n', encoding="utf-8"
    )

    no_entity_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"utils.py": "class MyClass: pass\n"},
        messages=["big.py: FileLimiter: moved MyClass → utils.py"],
        abort=False,
        entity_to_target={},  # empty!
    )
    with patch(_FL_PATCH, return_value=no_entity_result):
        list(
            run_engine(
                {str(f): [(1, 10)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    file_limiter_patch_update="basic",
                ),
                _repo_root=str(tmp_path),
            )
        )
    assert (
        other.read_text(encoding="utf-8")
        == '@patch("mypkg.big.MyClass")\ndef test_it(): pass\n'
    )


def test_patch_update_updates_per_file_source(tmp_path):
    """'update' mode, FL moved entities → per_file source with old path is updated."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f = pkg / "big.py"
    big_source = "".join(f"var_{i} = {i}\n" for i in range(10))
    f.write_text(big_source, encoding="utf-8")
    # Another diff file with an old @patch string
    other_diff = pkg / "test_big.py"
    other_diff.write_text(
        '@patch("mypkg.big.MyClass")\ndef test_it(): pass\n', encoding="utf-8"
    )

    fl_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"utils.py": "class MyClass: pass\n"},
        messages=["big.py: FileLimiter: moved MyClass → utils.py"],
        abort=False,
        entity_to_target={"MyClass": "utils.py"},
    )
    with patch(_FL_PATCH, return_value=fl_result):
        msgs = list(
            run_engine(
                {str(f): [(1, 10)], str(other_diff): [(1, 2)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    file_limiter_patch_update="basic",
                ),
                _repo_root=str(tmp_path),
            )
        )
    # The per_file source for other_diff should have the updated string
    updated_text = other_diff.read_text(encoding="utf-8")
    assert "mypkg.utils.MyClass" in updated_text
    assert any("patch_update" in m for m in msgs)


def test_patch_update_updates_other_file(tmp_path):
    """'update' mode, a separate file outside per_file gets updated on disk."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f = pkg / "big.py"
    big_source = "".join(f"var_{i} = {i}\n" for i in range(10))
    f.write_text(big_source, encoding="utf-8")
    # A file NOT in the diff that has the old @patch string
    other = tmp_path / "test_other.py"
    other.write_text(
        '@patch("mypkg.big.MyClass")\ndef test_it(): pass\n', encoding="utf-8"
    )

    fl_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"utils.py": "class MyClass: pass\n"},
        messages=["big.py: FileLimiter: moved MyClass → utils.py"],
        abort=False,
        entity_to_target={"MyClass": "utils.py"},
    )
    with patch(_FL_PATCH, return_value=fl_result):
        msgs = list(
            run_engine(
                {str(f): [(1, 10)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    file_limiter_patch_update="basic",
                ),
                _repo_root=str(tmp_path),
            )
        )
    updated_text = other.read_text(encoding="utf-8")
    assert "mypkg.utils.MyClass" in updated_text
    assert any("patch_update" in m for m in msgs)


def test_patch_update_skips_excluded_dir(tmp_path):
    """Files under .venv/ are excluded from Phase 4 scanning."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f = pkg / "big.py"
    big_source = "".join(f"var_{i} = {i}\n" for i in range(10))
    f.write_text(big_source, encoding="utf-8")
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    venv_file = venv_dir / "test.py"
    venv_content = '@patch("mypkg.big.MyClass")\ndef test_it(): pass\n'
    venv_file.write_text(venv_content, encoding="utf-8")

    fl_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"utils.py": "class MyClass: pass\n"},
        messages=["big.py: FileLimiter: moved MyClass → utils.py"],
        abort=False,
        entity_to_target={"MyClass": "utils.py"},
    )
    with patch(_FL_PATCH, return_value=fl_result):
        list(
            run_engine(
                {str(f): [(1, 10)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    file_limiter_patch_update="basic",
                ),
                _repo_root=str(tmp_path),
            )
        )
    # .venv/test.py must not be modified
    assert venv_file.read_text(encoding="utf-8") == venv_content


def test_patch_update_no_repo_root(tmp_path):
    """When repo_root can't be found, Phase 4 skips entirely."""
    # No .git or pyproject.toml → _find_repo_root returns None
    f = tmp_path / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")
    other = tmp_path / "test_other.py"
    other.write_text(
        '@patch("mypkg.big.MyClass")\ndef test_it(): pass\n', encoding="utf-8"
    )

    fl_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"utils.py": "class MyClass: pass\n"},
        messages=[],
        abort=False,
        entity_to_target={"MyClass": "utils.py"},
    )
    with patch(_FL_PATCH, return_value=fl_result):
        list(
            run_engine(
                {str(f): [(1, 10)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    file_limiter_patch_update="basic",
                ),
                # No _repo_root passed, no .git in tmp_path → repo_root=None
            )
        )
    # test_other.py should be unchanged
    assert (
        other.read_text(encoding="utf-8")
        == '@patch("mypkg.big.MyClass")\ndef test_it(): pass\n'
    )


def test_patch_update_oserror_skipped(tmp_path):
    """Phase 4 continues gracefully when read_text raises OSError."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f = pkg / "big.py"
    big_source = "".join(f"var_{i} = {i}\n" for i in range(10))
    f.write_text(big_source, encoding="utf-8")
    # A file that will raise OSError when read
    other = tmp_path / "test_other.py"
    other.write_text(
        '@patch("mypkg.big.MyClass")\ndef test_it(): pass\n', encoding="utf-8"
    )

    fl_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"utils.py": "class MyClass: pass\n"},
        messages=[],
        abort=False,
        entity_to_target={"MyClass": "utils.py"},
    )

    other_abs = str(other.resolve())

    original_pathlib_read = None

    def _patched_read_text(self, encoding="utf-8"):
        if str(self.resolve()) == other_abs:
            raise OSError("permission denied")
        return original_pathlib_read(self, encoding=encoding)

    import pathlib

    original_pathlib_read = pathlib.Path.read_text

    with patch.object(pathlib.Path, "read_text", _patched_read_text):
        with patch(_FL_PATCH, return_value=fl_result):
            # Should not raise even though read_text raises OSError
            msgs = list(
                run_engine(
                    {str(f): [(1, 10)]},
                    config=CrispenConfig(
                        max_file_lines=5,
                        file_limiter_patch_update="basic",
                    ),
                    _repo_root=str(tmp_path),
                )
            )

    # No patch_update message for other since it raised OSError
    assert not any("test_other" in m and "patch_update" in m for m in msgs)


def test_patch_update_accumulates_from_recursive_fl(tmp_path):
    """Entities from recursive FL results also contribute to the patch map."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f = pkg / "big.py"
    big_source = "".join(f"var_{i} = {i}\n" for i in range(10))
    f.write_text(big_source, encoding="utf-8")

    # Other file outside per_file with old @patch strings for both files
    other = tmp_path / "test_other.py"
    other.write_text(
        '@patch("mypkg.big.MyClass")\n'
        '@patch("mypkg.utils.HelperClass")\n'
        "def test_it(): pass\n",
        encoding="utf-8",
    )

    # First FL result: big.py → utils.py (MyClass moved there)
    first_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={
            "utils.py": "class MyClass: pass\n" * 10
        },  # over limit for recursion
        messages=[],
        abort=False,
        entity_to_target={"MyClass": "utils.py"},
    )

    # utils.py written by first result; set up that file
    utils_path = pkg / "utils.py"

    # Second FL result: utils.py → helpers.py (HelperClass moved there)
    second_result = FileLimiterResult(
        original_source="# utils reduced\n",
        new_files={"helpers.py": "class HelperClass: pass\n"},
        messages=[],
        abort=False,
        entity_to_target={"HelperClass": "helpers.py"},
    )

    call_count = 0

    def _fl_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Write utils.py so the recursive call can find it
            utils_path.write_text("class MyClass: pass\n" * 10, encoding="utf-8")
            return first_result
        return second_result

    with patch(_FL_PATCH, side_effect=_fl_side_effect):
        msgs = list(
            run_engine(
                {str(f): [(1, 10)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    file_limiter_recursive=True,
                    file_limiter_patch_update="basic",
                ),
                _repo_root=str(tmp_path),
            )
        )

    # Verify combined_patch_map was non-empty by checking at least one
    # patch_update message was generated (from the other file or per_file).
    updated_text = other.read_text(encoding="utf-8")
    # At minimum, MyClass should be updated (from first pass)
    assert "mypkg.utils.MyClass" in updated_text or any(
        "patch_update" in m for m in msgs
    )


def test_patch_update_chain_flattening(tmp_path):
    """Transitive chains in combined_patch_map are flattened before apply.

    When a first split produces A→B and a recursive split produces B→C,
    apply_patch_strings must map A directly to C, not to the intermediate B.
    """
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f = pkg / "big.py"
    # big.py imports get_api_key and uses it; also defines func_0 (moved entity)
    big_source = "from llm import get_api_key\n" + "".join(
        f"def func_{i}(): get_api_key()\n" for i in range(10)
    )
    f.write_text(big_source, encoding="utf-8")

    # Other file has @patch pointing at the imported alias in big.py
    other = tmp_path / "test_other.py"
    other.write_text(
        '@patch("mypkg.big.get_api_key")\ndef test_it(): pass\n',
        encoding="utf-8",
    )

    # First FL result: big.py → utils.py.
    # utils.py is over max_file_lines so it will be queued for recursive split.
    # It imports and uses get_api_key so the alias ends up in utils's map entry.
    utils_source = "from llm import get_api_key\n" + "".join(
        f"def helper_{i}(): get_api_key()\n" for i in range(10)
    )
    first_result = FileLimiterResult(
        original_source="# big reduced\n",
        new_files={"utils.py": utils_source},
        messages=[],
        abort=False,
        entity_to_target={
            "func_0": "utils.py"
        },  # non-empty to trigger _build_patch_map
    )

    # Second FL result (recursive split of utils.py) → helpers.py
    helpers_source = "from llm import get_api_key\n" "def helper_0(): get_api_key()\n"
    second_result = FileLimiterResult(
        original_source="# utils reduced\n",
        new_files={"helpers.py": helpers_source},
        messages=[],
        abort=False,
        entity_to_target={"helper_0": "helpers.py"},  # non-empty
    )

    call_count = 0

    def _fl_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        return first_result if call_count == 1 else second_result

    with patch(_FL_PATCH, side_effect=_fl_side_effect):
        list(
            run_engine(
                {str(f): [(1, len(big_source.splitlines()))]},
                config=CrispenConfig(
                    max_file_lines=5,
                    file_limiter_recursive=True,
                    file_limiter_patch_update="basic",
                ),
                _repo_root=str(tmp_path),
            )
        )

    # Round 1 map: mypkg.big.get_api_key → mypkg.utils.get_api_key
    # Round 2 map: mypkg.utils.get_api_key → mypkg.helpers.get_api_key
    # After flattening: mypkg.big.get_api_key → mypkg.helpers.get_api_key
    # Without flattening the test file would still hold the intermediate path.
    updated = other.read_text(encoding="utf-8")
    assert (
        "mypkg.helpers.get_api_key" in updated
    ), f"Expected chain-flattened path but got: {updated!r}"


def test_add_fl_context_no_module_path():
    """When module path cannot be determined, _add_fl_context does nothing."""
    fl_list = []
    fl_result = FileLimiterResult(
        original_source="",
        new_files={},
        abort=False,
        entity_to_target={"X": "a.py"},
    )
    # A path with no ancestor containing pyproject.toml / .git → returns None.
    _add_fl_context(fl_list, "/no/project/root/here/file.py", "", fl_result, {})
    assert fl_list == []


def test_add_fl_context_no_forking(tmp_path):
    """When all entities are already in combined_patch_map, nothing is appended."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    fl_result = FileLimiterResult(
        original_source="",
        new_files={},
        abort=False,
        entity_to_target={"X": "a.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    # Entity already covered by combined_patch_map → forking_old_paths is empty.
    # No _block_N entities → nothing appended.
    _add_fl_context(fl_list, filepath, "", fl_result, {"mypkg.big.X": "mypkg.a.X"})
    assert fl_list == []


def test_add_fl_context_block_entity_uses_specific_names(tmp_path):
    """When a _block_N entity was moved and all named entities are mapped,
    the block-internal names (vars, imports) from the target file are used as
    specific scan keys — NOT the broad module path — so already-updated strings
    like ``old_module.sub.run_engine`` are not re-sent to the LLM."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    fl_result = FileLimiterResult(
        original_source="modified\n",
        new_files={
            "core.py": "_REFACTORS = []\nimport libcst as cst\n\ndef X(): pass\n"
        },
        abort=False,
        entity_to_target={"_block_1": "core.py", "X": "core.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    # Both entities are in the patch map → forking_old_paths would be empty.
    # _block_1 is a TOP_LEVEL block → scan core.py for block-internal names.
    # X is in entity_to_target so it's excluded; _REFACTORS and cst are not.
    combined = {
        "mypkg.big._block_1": "mypkg.core._block_1",
        "mypkg.big.X": "mypkg.core.X",
    }
    _add_fl_context(fl_list, filepath, "original\n", fl_result, combined)
    assert len(fl_list) == 1
    # Only _REFACTORS and cst are block-internal; X is excluded (named entity).
    assert fl_list[0].forking_old_paths == {"mypkg.big._REFACTORS", "mypkg.big.cst"}
    assert fl_list[0].old_module == "mypkg.big"


def test_add_fl_context_block_entity_no_new_names(tmp_path):
    """When a _block_N entity was moved but the target file contains no names
    beyond those already in entity_to_target, nothing is appended."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    fl_result = FileLimiterResult(
        original_source="modified\n",
        # core.py only defines X, which is already in entity_to_target.
        new_files={"core.py": "def X(): pass\n"},
        abort=False,
        entity_to_target={"_block_1": "core.py", "X": "core.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    combined = {
        "mypkg.big._block_1": "mypkg.core._block_1",
        "mypkg.big.X": "mypkg.core.X",
    }
    _add_fl_context(fl_list, filepath, "original\n", fl_result, combined)
    assert fl_list == []


def test_add_fl_context_forking_and_block_combined(tmp_path):
    """Forking entities AND block-internal names are both added to forking_old_paths."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    fl_result = FileLimiterResult(
        original_source="modified\n",
        new_files={"core.py": "_TIMEOUT = 30\ndef Y(): pass\n"},
        abort=False,
        # Y is forking (not in combined_patch_map); _block_1 moved with _TIMEOUT inside.
        entity_to_target={"_block_1": "core.py", "Y": "core.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    # Only _block_1 is in combined_patch_map; Y is not (forking).
    combined = {"mypkg.big._block_1": "mypkg.core._block_1"}
    _add_fl_context(fl_list, filepath, "original\n", fl_result, combined)
    assert len(fl_list) == 1
    # Y is a forking entity; _TIMEOUT is block-internal; Y in new file is excluded
    # (it's in all_entity_names).
    assert fl_list[0].forking_old_paths == {"mypkg.big.Y", "mypkg.big._TIMEOUT"}


def test_add_fl_context_normal(tmp_path):
    """Forking entity not in combined_patch_map → appended to fl_all_contexts."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    fl_result = FileLimiterResult(
        original_source="modified\n",
        new_files={"utils.py": "class X: pass\n"},
        abort=False,
        entity_to_target={"X": "utils.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    _add_fl_context(fl_list, filepath, "original\n", fl_result, {})
    assert len(fl_list) == 1
    assert fl_list[0].forking_old_paths == {"mypkg.big.X"}
    assert fl_list[0].old_module == "mypkg.big"
    assert fl_list[0].original_source == "original\n"
    assert fl_list[0].modified_source == "modified\n"


def test_add_fl_context_forked_import_alias_added(tmp_path):
    """Import aliases forked across multiple new files are added to forking_old_paths.

    When the original file imports ``call_with_tool`` and multiple new sub-files
    also import it, basic mode skips it (forking).  _add_fl_context must still
    add ``old_module.call_with_tool`` to forking_old_paths so the LLM rewrite
    step can detect and update @patch decorators that reference it.
    """
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    # Original file imports call_with_tool; both new sub-files also import it
    # (forking) so basic mode left it out of combined_patch_map.
    pre_split = "from external import call_with_tool\ndef F(): pass\n"
    fl_result = FileLimiterResult(
        original_source="modified\n",
        new_files={
            "a.py": (
                "from external import call_with_tool\ncall_with_tool()\ndef F(): pass\n"
            ),
            "b.py": "from external import call_with_tool\ncall_with_tool()\n",
        },
        abort=False,
        entity_to_target={"F": "a.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    # F is already in combined_patch_map (non-forking entity); call_with_tool
    # is NOT in combined_patch_map (forked, skipped by basic mode).
    combined = {"mypkg.big.F": "mypkg.a.F"}
    _add_fl_context(fl_list, filepath, pre_split, fl_result, combined)
    assert len(fl_list) == 1
    # call_with_tool must be in forking_old_paths despite F being already mapped.
    assert "mypkg.big.call_with_tool" in fl_list[0].forking_old_paths


def test_add_fl_context_forked_import_alias_entity_name_skipped(tmp_path):
    """Import alias that is also an entity name is skipped by the alias loop's continue.

    Helper is in entity_to_target (not in combined_patch_map) so the entity
    section already adds it to forking_old_paths.  The alias loop hits the
    ``continue`` branch and does not process it again.  ``other`` (import alias
    only, not an entity) is picked up by the alias loop instead.
    """
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    pre_split = "from ext import Helper, other\ndef F(): pass\n"
    fl_result = FileLimiterResult(
        original_source="modified\n",
        new_files={"a.py": "from ext import other\nother()\ndef F(): pass\n"},
        abort=False,
        # Helper is both an imported alias and a named entity (forking entity).
        entity_to_target={"Helper": "a.py", "F": "a.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    # Neither entity is in combined_patch_map → both are forking.
    _add_fl_context(fl_list, filepath, pre_split, fl_result, {})
    assert len(fl_list) == 1
    # Helper was added by the entity section; other was added by the alias loop.
    assert "mypkg.big.Helper" in fl_list[0].forking_old_paths
    assert "mypkg.big.other" in fl_list[0].forking_old_paths


def test_add_fl_context_forked_import_alias_already_mapped_skipped(tmp_path):
    """Import alias already in combined_patch_map is not re-added."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    # pre_split imports Helper (already mapped by basic) and call_with_tool (forked).
    pre_split = "from ext import Helper, call_with_tool\ndef F(): pass\n"
    fl_result = FileLimiterResult(
        original_source="modified\n",
        new_files={
            "a.py": "from ext import call_with_tool\ncall_with_tool()\ndef F(): pass\n",
            "b.py": "from ext import call_with_tool\ncall_with_tool()\n",
        },
        abort=False,
        entity_to_target={"F": "a.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    # Helper is already in combined_patch_map (basic mapped it); call_with_tool is not.
    combined = {
        "mypkg.big.F": "mypkg.a.F",
        "mypkg.big.Helper": "mypkg.a.Helper",
    }
    _add_fl_context(fl_list, filepath, pre_split, fl_result, combined)
    assert len(fl_list) == 1
    # Helper is already mapped → not added again; call_with_tool is forked → added.
    assert "mypkg.big.Helper" not in fl_list[0].forking_old_paths
    assert "mypkg.big.call_with_tool" in fl_list[0].forking_old_paths


def test_add_fl_context_subdir_split_uses_init_as_modified_source(tmp_path):
    """Non-test subdir split: modified_source comes from new_files[subdir/__init__.py].

    runner.py restores fl_result.original_source to the pre-split source for
    non-test, non-has_main subdir splits and places the post-split __init__.py
    content in new_files.  _add_fl_context must use that __init__.py content as
    modified_source so _build_rename_guard_sets and the BFS terminal builder see
    the correct set of names still present in the module after the split.
    """
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    pre_split = "from llm import call_with_tool\ndef F(): pass\ndef G(): pass\n"
    # Post-split __init__.py re-exports F but call_with_tool is NOT re-exported.
    init_src = "from .sub import F\ndef advise(): pass\n"
    fl_result = FileLimiterResult(
        # runner.py restored original_source to pre-split for non-test subdir.
        original_source=pre_split,
        new_files={
            "advisor/__init__.py": init_src,
            "advisor/sub.py": "from llm import call_with_tool\ndef F(): pass\n",
        },
        abort=False,
        subdir_name="advisor",
        entity_to_target={"F": "advisor/sub.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    _add_fl_context(fl_list, filepath, pre_split, fl_result, {})
    assert len(fl_list) == 1
    # modified_source must be the __init__.py content, not original_source.
    assert fl_list[0].modified_source == init_src
    assert fl_list[0].original_source == pre_split


def test_add_fl_context_subdir_split_no_init_falls_back_to_original_source(tmp_path):
    """Test/has_main subdir split: no __init__.py → falls back to original_source.

    For test files and has_main files with subdir_name set, runner.py does NOT
    add a subdir/__init__.py to new_files.  The modified_source should therefore
    fall back to fl_result.original_source (the post-split original file).
    """
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    pre_split = "from llm import call_with_tool\ndef F(): pass\n"
    # No __init__.py in new_files; original_source is the post-split state.
    post_split_original = "from llm import call_with_tool\ndef F(): pass\n# stubs\n"
    fl_result = FileLimiterResult(
        original_source=post_split_original,
        new_files={"advisor/sub.py": "def G(): pass\n"},
        abort=False,
        subdir_name="advisor",
        entity_to_target={"G": "advisor/sub.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    _add_fl_context(fl_list, filepath, pre_split, fl_result, {})
    assert len(fl_list) == 1
    # Falls back to fl_result.original_source since no __init__.py in new_files.
    assert fl_list[0].modified_source == post_split_original


def test_patch_update_rewrite_mode_calls_apply_patch_rewrite(tmp_path):
    """'rewrite' mode with forking entities calls apply_patch_rewrite in Phase 4."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f = pkg / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    # Entity appears as a caller in two new files → forking → skipped by basic.
    fl_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={
            "utils.py": "class MyClass: pass\n",
            "caller_a.py": "from .big import MyClass\nMyClass()\n",
            "caller_b.py": "from .big import MyClass\nMyClass()\n",
        },
        messages=[],
        abort=False,
        entity_to_target={"MyClass": "utils.py"},
    )
    with (
        patch(_FL_PATCH, return_value=fl_result),
        patch(_REWRITE_PATCH, return_value=iter([])) as mock_rewrite,
    ):
        list(
            run_engine(
                {str(f): [(1, 10)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    file_limiter_patch_update="rewrite",
                ),
                _repo_root=str(tmp_path),
            )
        )
    mock_rewrite.assert_called_once()
    contexts = mock_rewrite.call_args[0][0]
    assert len(contexts) == 1
    assert "mypkg.big.MyClass" in contexts[0].forking_old_paths


def test_patch_update_rewrite_mode_records_llm_stats(tmp_path):
    """Rewrite accumulator with non-zero elapsed/tokens triggers record_llm_call."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f = pkg / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    fl_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={
            "utils.py": "class MyClass: pass\n",
            "caller_a.py": "from .big import MyClass\nMyClass()\n",
            "caller_b.py": "from .big import MyClass\nMyClass()\n",
        },
        messages=[],
        abort=False,
        entity_to_target={"MyClass": "utils.py"},
    )

    def _rewrite_with_acc(
        fl_contexts, per_file, repo_root, config, verbose=False, _acc=None, **_kwargs
    ):
        if _acc is not None:
            _acc.calls = 2
            _acc.elapsed = 1.5
            _acc.input_tokens = 100
            _acc.output_tokens = 20
            _acc.files_updated = 1
        return iter([])

    stats = RunStats()
    with (
        patch(_FL_PATCH, return_value=fl_result),
        patch(_REWRITE_PATCH, side_effect=_rewrite_with_acc),
    ):
        list(
            run_engine(
                {str(f): [(1, 10)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    file_limiter_patch_update="rewrite",
                ),
                _repo_root=str(tmp_path),
                stats=stats,
            )
        )
    assert stats.patch_rewrite_llm_calls == 2
    assert stats.patch_update_edits == 1
    assert stats.llm_elapsed == 1.5
    assert stats.llm_input_tokens == 100
    assert "patch_rewriter" in stats.llm_elapsed_by_refactor


def test_patch_update_rewrite_mode_no_fl_contexts_skips_apply(tmp_path):
    """'rewrite' mode but no forking entities → apply_patch_rewrite not called."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f = pkg / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    # Entity has only ONE caller → non-forking → goes into combined_patch_map.
    fl_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"utils.py": "class MyClass: pass\n"},
        messages=[],
        abort=False,
        entity_to_target={"MyClass": "utils.py"},
    )
    with (
        patch(_FL_PATCH, return_value=fl_result),
        patch(_REWRITE_PATCH, return_value=iter([])) as mock_rewrite,
    ):
        list(
            run_engine(
                {str(f): [(1, 10)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    file_limiter_patch_update="rewrite",
                ),
                _repo_root=str(tmp_path),
            )
        )
    mock_rewrite.assert_not_called()


def test_patch_update_rewrite_mode_recursive_fl_context_added(tmp_path):
    """'rewrite' mode: forking entity from recursive FL pass is collected."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f = pkg / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    # Main FL result: produces medium.py with 6 lines (> max_file_lines=5),
    # which triggers the recursive pass.  No entity_to_target here so the
    # main-loop rewrite branch is not entered.
    medium_src = "".join(f"med_{i} = {i}\n" for i in range(6))
    main_fl_result = FileLimiterResult(
        original_source="# big_reduced\n",
        new_files={"medium.py": medium_src},
        messages=[],
        abort=False,
        entity_to_target={},
    )

    # Recursive FL result: MyClass appears in two callers → forking → skipped
    # by _build_patch_map → not in combined_patch_map → triggers _add_fl_context.
    recursive_fl_result = FileLimiterResult(
        original_source="# medium_reduced\n",
        new_files={
            "small.py": "class MyClass: pass\n",
            "caller_a.py": "from .medium import MyClass\nMyClass()\n",
            "caller_b.py": "from .medium import MyClass\nMyClass()\n",
        },
        messages=[],
        abort=False,
        entity_to_target={"MyClass": "small.py"},
    )

    with (
        patch(_FL_PATCH, side_effect=[main_fl_result, recursive_fl_result]),
        patch(_REWRITE_PATCH, return_value=iter([])) as mock_rewrite,
    ):
        list(
            run_engine(
                {str(f): [(1, 10)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    file_limiter_recursive=True,
                    file_limiter_patch_update="rewrite",
                ),
                _repo_root=str(tmp_path),
            )
        )

    mock_rewrite.assert_called_once()
    contexts = mock_rewrite.call_args[0][0]
    assert any("mypkg.medium.MyClass" in ctx.forking_old_paths for ctx in contexts)


def test_patch_update_callgraph_yields_message(tmp_path):
    """apply_patch_callgraph message increments patch_update_edits and is yielded."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f = pkg / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    # Entity appears in two callers → forking → _fl_all_contexts is populated
    fl_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={
            "utils.py": "class MyClass: pass\n",
            "caller_a.py": "from .big import MyClass\nMyClass()\n",
            "caller_b.py": "from .big import MyClass\nMyClass()\n",
        },
        messages=[],
        abort=False,
        entity_to_target={"MyClass": "utils.py"},
    )

    cg_msg = "test_other.py: patch_callgraph: resolved MyClass"

    stats = RunStats()
    with (
        patch(_FL_PATCH, return_value=fl_result),
        patch(_CG_PATCH, return_value=iter([cg_msg])) as mock_cg,
    ):
        msgs = list(
            run_engine(
                {str(f): [(1, 10)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    file_limiter_patch_update="basic",
                ),
                _repo_root=str(tmp_path),
                stats=stats,
            )
        )

    mock_cg.assert_called_once()
    assert cg_msg in msgs
    assert stats.patch_update_edits >= 1


def test_patch_update_ignore_mode_recursive_fl_entity_to_target(tmp_path):
    """'ignore' mode: recursive FL result with entity_to_target skips _add_fl_context."""  # noqa: E501
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f = pkg / "big.py"
    f.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    medium_src = "".join(f"med_{i} = {i}\n" for i in range(6))
    main_fl_result = FileLimiterResult(
        original_source="# big_reduced\n",
        new_files={"medium.py": medium_src},
        messages=[],
        abort=False,
        entity_to_target={},  # empty — no _add_fl_context for main result
    )

    # Recursive FL result has non-empty entity_to_target; with "ignore" mode the
    # branch at engine.py line 1278 is False → _add_fl_context is not called.
    recursive_fl_result = FileLimiterResult(
        original_source="# medium_reduced\n",
        new_files={"small.py": "class MyClass: pass\n"},
        messages=[],
        abort=False,
        entity_to_target={"MyClass": "small.py"},
    )

    medium_path = pkg / "medium.py"
    call_count = 0

    def _fl_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            medium_path.write_text(medium_src, encoding="utf-8")
            return main_fl_result
        return recursive_fl_result

    with patch(_FL_PATCH, side_effect=_fl_side_effect):
        msgs = list(
            run_engine(
                {str(f): [(1, 10)]},
                config=CrispenConfig(
                    max_file_lines=5,
                    file_limiter_recursive=True,
                    file_limiter_patch_update="ignore",
                ),
                _repo_root=str(tmp_path),
            )
        )

    assert call_count == 2  # main pass + one recursive pass
    assert not any("callgraph" in m for m in msgs)
