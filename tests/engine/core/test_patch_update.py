from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import _build_patch_map, run_engine
from crispen.file_limiter.runner import FileLimiterResult
from ..helpers import _FL_PATCH


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
