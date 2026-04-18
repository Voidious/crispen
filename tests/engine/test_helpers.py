from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import (
    _add_fl_context,
    _build_patch_map,
    _collect_code_referenced_names,
    _patch_inline_imports_after_test_deletion,
    _should_run,
    run_engine,
)
from crispen.file_limiter.runner import FileLimiterResult
from .helpers import _FL_PATCH


def test_should_run_disabled_list_allows_unlisted():
    cfg = CrispenConfig(disabled_refactors=["function_splitter"])
    assert _should_run("if_not_else", cfg) is True
    assert _should_run("tuple_dataclass", cfg) is True


def test_file_limiter_subdir_split_empty_source_file_already_deleted(tmp_path):
    """Subdir split deletes the original file; empty original_source skips re-unlink."""
    f = tmp_path / "big.py"
    original = "".join(f"var_{i} = {i}\n" for i in range(10))
    f.write_text(original, encoding="utf-8")
    # subdir_name causes Phase 3 to delete the original file.  original_source=""
    # means the per_file loop sees an empty source for a file that no longer
    # exists — exercising the elif-is-False branch (803→805).
    # new_files content kept short (≤ max_file_lines) to avoid recursive queue.
    subdir_result = FileLimiterResult(
        original_source="",
        new_files={"big/__init__.py": "# package\n"},
        messages=[f"{f}: FileLimiter: subdir split → big/"],
        abort=False,
        subdir_name="big",
    )
    with patch(_FL_PATCH, return_value=subdir_result):
        msgs = list(
            run_engine(
                {str(f): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5),
            )
        )
    assert any("FileLimiter" in m for m in msgs)
    # Original file was deleted by the subdir split; must not exist.
    assert not f.exists()
    # Package init was created.
    assert (tmp_path / "big" / "__init__.py").exists()


def test_file_limiter_recursive_empty_init_py_preserved(tmp_path):
    """__init__.py created during recursive split is kept even when drained empty."""
    orig = tmp_path / "big.py"
    orig.write_text("".join(f"var_{i} = {i}\n" for i in range(10)), encoding="utf-8")

    oversized = "".join(f"x_{i} = {i}\n" for i in range(10))
    first_result = FileLimiterResult(
        original_source="# reduced\n",
        new_files={"pkg/__init__.py": oversized},
        messages=[],
        abort=False,
    )
    # Recursive pass drains pkg/__init__.py; original_source is empty.
    second_result = FileLimiterResult(
        original_source="",
        new_files={"pkg/utils.py": "# utils\n"},
        messages=[],
        abort=False,
    )
    with patch(_FL_PATCH, side_effect=[first_result, second_result]):
        list(
            run_engine(
                {str(orig): [(1, 1)]},
                config=CrispenConfig(max_file_lines=5, file_limiter_recursive=True),
            )
        )
    # pkg/__init__.py must survive as an empty file, not be deleted.
    assert (tmp_path / "pkg" / "__init__.py").exists()
    assert (tmp_path / "pkg" / "__init__.py").read_text(encoding="utf-8") == ""


def test_patch_inline_imports_after_test_deletion_empty_fl_entry_skipped(tmp_path):
    # fl_new_file_final entry with empty/None content is skipped without error.
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    sub = tmp_path / "pkg" / "sub"
    sub.mkdir(parents=True)
    (sub / "test_new.py").write_text("class TestFoo:\n    pass\n", encoding="utf-8")
    deleted_path = str(tmp_path / "pkg" / "test_old.py")
    deleted_dir = tmp_path / "pkg"
    fl_new_file_final = {str(tmp_path / "empty.py"): ""}
    _patch_inline_imports_after_test_deletion(
        deleted_path,
        deleted_dir,
        {"sub/test_new.py": "class TestFoo:\n    pass\n"},
        {},
        fl_new_file_final,
    )
    # Empty entry was skipped; dict unchanged.
    assert fl_new_file_final[str(tmp_path / "empty.py")] == ""


def test_collect_code_referenced_names_syntax_error():
    """Returns empty set on unparseable source."""
    assert _collect_code_referenced_names("def (broken:") == set()


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
