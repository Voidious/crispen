from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import _build_patch_map, run_engine
from crispen.file_limiter.runner import FileLimiterResult
from crispen.stats import RunStats
from .helpers import _FL_PATCH, _make_fl_result_with_entities
from .patch_update import _CG_PATCH, _REWRITE_PATCH


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
