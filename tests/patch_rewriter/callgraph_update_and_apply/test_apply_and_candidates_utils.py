from __future__ import annotations
from crispen.patch_rewriter import (
    _FLContext,
    _candidates_check,
    _patch_strings_in_text,
    _rewrite_candidates_check,
    apply_patch_callgraph,
)
from ..helpers import _make_cuf_contexts, _make_fl_ctx


def test_candidates_check_no_candidates():
    # No candidates for any path → None.
    assert _candidates_check({"pkg.orig.A": "pkg.sub.A"}, ["pkg.orig.A"], {}) is None


def test_candidates_check_rename_valid():
    # Rename is in candidates → None.
    cands = {"pkg.orig.A": ["pkg.placement.A", "pkg.helpers.A"]}
    assert (
        _candidates_check({"pkg.orig.A": "pkg.placement.A"}, ["pkg.orig.A"], cands)
        is None
    )


def test_candidates_check_rename_invalid():
    # Rename proposes a path not in candidates → error message.
    cands = {"pkg.orig.A": ["pkg.placement.A"]}
    result = _candidates_check({"pkg.orig.A": "pkg.wrong.A"}, ["pkg.orig.A"], cands)
    assert result is not None
    assert "pkg.wrong.A" in result
    assert "pkg.placement.A" in result


def test_candidates_check_no_change_with_candidates():
    # No rename proposed for a path that has candidates → error message.
    cands = {"pkg.orig.A": ["pkg.placement.A"]}
    result = _candidates_check({}, ["pkg.orig.A"], cands)
    assert result is not None
    assert "pkg.orig.A" in result
    assert "pkg.placement.A" in result


def test_candidates_check_path_not_in_candidates():
    # Another path has no candidates → passes; only paths with candidates are checked.
    cands = {"pkg.orig.A": ["pkg.placement.A"]}
    # pkg.orig.B has no candidates; even though no rename proposed → None
    assert _candidates_check({}, ["pkg.orig.B"], cands) is None


def test_candidates_check_no_change_when_old_in_candidates():
    # No rename proposed but old path is itself one of the candidates (e.g. the entity
    # is still accessible at the original module via __init__.py re-export) → None.
    cands = {"pkg.orig.A": ["pkg.orig.A", "pkg.resolver.A"]}
    assert _candidates_check({}, ["pkg.orig.A"], cands) is None


def test_apply_patch_callgraph_candidates_out_per_file(tmp_path):
    # Multiple candidates → saved in candidates_out for per_file entry.
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
        "    resolve()\n"
    )
    test_file = tmp_path / "test_orig.py"
    test_file.write_text(test_src, encoding="utf-8")
    per_file = {str(test_file): {"source": test_src, "msgs": []}}
    candidates_out: dict = {}
    list(
        apply_patch_callgraph(
            _make_cuf_contexts(), per_file, str(tmp_path), candidates_out=candidates_out
        )
    )
    abs_fp = str(test_file.resolve())
    assert abs_fp in candidates_out
    assert "test_f" in candidates_out[abs_fp]


def test_apply_patch_callgraph_candidates_out_disk_file(tmp_path):
    # Multiple candidates → saved in candidates_out for disk file.
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
        "    resolve()\n"
    )
    test_file = tmp_path / "test_orig.py"
    test_file.write_text(test_src, encoding="utf-8")
    candidates_out: dict = {}
    list(
        apply_patch_callgraph(
            _make_cuf_contexts(), {}, str(tmp_path), candidates_out=candidates_out
        )
    )
    abs_fp = str(test_file.resolve())
    assert abs_fp in candidates_out
    assert "test_f" in candidates_out[abs_fp]


def test_patch_strings_in_text_decorator():
    text = '@patch("pkg.mod.A")\ndef test_f(m): pass\n'
    assert _patch_strings_in_text(text) == {"pkg.mod.A"}


def test_patch_strings_in_text_attribute_decorator():
    text = '@mock.patch("pkg.mod.B")\ndef test_f(m): pass\n'
    assert _patch_strings_in_text(text) == {"pkg.mod.B"}


def test_patch_strings_in_text_context_manager():
    text = 'def test_f():\n    with patch("pkg.mod.C") as m: pass\n'
    assert _patch_strings_in_text(text) == {"pkg.mod.C"}


def test_patch_strings_in_text_multiple():
    text = (
        '@patch("pkg.mod.A")\n' '@mock.patch("pkg.mod.B")\n' "def test_f(a, b): pass\n"
    )
    assert _patch_strings_in_text(text) == {"pkg.mod.A", "pkg.mod.B"}


def test_patch_strings_in_text_empty():
    assert _patch_strings_in_text("def test_f(): pass\n") == set()


def test_rewrite_candidates_check_no_candidates():
    # No candidates for any path → None.
    text = '@patch("pkg.mod.A")\ndef test_f(m): pass\n'
    assert _rewrite_candidates_check(["pkg.mod.A"], text, {}) is None


def test_rewrite_candidates_check_valid_rename():
    # Old path absent, one candidate present → None.
    text = '@patch("pkg.placement.A")\ndef test_f(m): pass\n'
    cands = {"pkg.mod.A": ["pkg.placement.A", "pkg.other.A"]}
    assert _rewrite_candidates_check(["pkg.mod.A"], text, cands) is None


def test_rewrite_candidates_check_old_still_present():
    # Old path still present even though candidates exist → error.
    text = '@patch("pkg.mod.A")\ndef test_f(m): pass\n'
    cands = {"pkg.mod.A": ["pkg.placement.A"]}
    result = _rewrite_candidates_check(["pkg.mod.A"], text, cands)
    assert result is not None
    assert "pkg.mod.A" in result
    assert "pkg.placement.A" in result


def test_rewrite_candidates_check_renamed_to_unknown():
    # Old path absent, no known candidate appears — could be a wrong rename or a
    # dead-code removal. Let the LLM verify step decide; no error returned here.
    text = '@patch("pkg.wrong.A")\ndef test_f(m): pass\n'
    cands = {"pkg.mod.A": ["pkg.placement.A", "pkg.other.A"]}
    assert _rewrite_candidates_check(["pkg.mod.A"], text, cands) is None


def test_rewrite_candidates_check_deleted_patch():
    # Old path absent and decorator was removed entirely → dead-code removal is
    # allowed; let the LLM verify step confirm correctness.
    text = "def test_f(): pass\n"
    cands = {"pkg.mod.A": ["pkg.placement.A", "pkg.other.A"]}
    assert _rewrite_candidates_check(["pkg.mod.A"], text, cands) is None


def test_rewrite_candidates_check_path_without_candidates_ignored():
    # A path with no candidates in the dict → skip it.
    text = '@patch("pkg.mod.B")\ndef test_f(m): pass\n'
    cands = {"pkg.mod.A": ["pkg.placement.A"]}  # A has candidates, B does not
    assert _rewrite_candidates_check(["pkg.mod.B"], text, cands) is None


def test_apply_patch_callgraph_empty_contexts():
    result = list(apply_patch_callgraph([], {}, "/repo"))
    assert result == []


def test_apply_patch_callgraph_no_forking_paths():
    ctx = _make_fl_ctx(forking_old_paths=set())
    result = list(apply_patch_callgraph([ctx], {}, "/repo"))
    assert result == []


def test_apply_patch_callgraph_per_file_update(tmp_path):
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    conflict_src = "from external import use_fn\ndef resolve(): use_fn()\n"
    ctx = _FLContext(
        filepath=str(tmp_path / "pkg" / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src, "conflict.py": conflict_src},
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    file_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn):\n"
        "    helper()\n"
    )
    test_file = tmp_path / "test_orig.py"
    test_file.write_text(file_src, encoding="utf-8")
    per_file = {str(test_file): {"source": file_src, "msgs": []}}
    list(apply_patch_callgraph([ctx], per_file, str(tmp_path)))
    assert '@patch("pkg.placement.use_fn")' in per_file[str(test_file)]["source"]


def test_apply_patch_callgraph_repo_scan(tmp_path):
    test_file = tmp_path / "test_something.py"
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    test_file.write_text(
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn):\n"
        "    helper()\n",
        encoding="utf-8",
    )
    ctx = _FLContext(
        filepath=str(tmp_path / "pkg" / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    msgs = list(apply_patch_callgraph([ctx], {}, str(tmp_path)))
    updated = test_file.read_text(encoding="utf-8")
    assert '@patch("pkg.placement.use_fn")' in updated
    assert any("call-graph" in m for m in msgs)


def test_apply_patch_callgraph_repo_scan_no_change(tmp_path):
    test_file = tmp_path / "test_something.py"
    test_file.write_text("def test_f(): pass\n", encoding="utf-8")
    ctx = _FLContext(
        filepath=str(tmp_path / "pkg" / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": "from external import use_fn\ndef f(): use_fn()\n"},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    msgs = list(apply_patch_callgraph([ctx], {}, str(tmp_path)))
    assert msgs == []


def test_apply_patch_callgraph_repo_root_none():
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={},
        new_module_paths={},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    result = list(apply_patch_callgraph([ctx], {}, None))
    assert result == []


def test_apply_patch_callgraph_per_file_no_change(tmp_path):
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath=str(tmp_path / "pkg" / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    file_src = "# pkg.orig.use_fn mentioned here but no test functions.\nx = 1\n"
    key = str(tmp_path / "module.py")
    per_file = {key: {"source": file_src, "msgs": []}}
    list(apply_patch_callgraph([ctx], per_file, None))
    assert per_file[key]["source"] == file_src


def test_apply_patch_callgraph_per_file_no_match(tmp_path):
    """per_file entry whose source contains no forking path string → continue."""
    ctx = _FLContext(
        filepath=str(tmp_path / "pkg" / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n"
        },
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    file_src = "x = 1\n"  # no mention of forking path
    key = str(tmp_path / "module.py")
    per_file = {key: {"source": file_src, "msgs": []}}
    list(apply_patch_callgraph([ctx], per_file, None))
    assert per_file[key]["source"] == file_src


def test_apply_patch_callgraph_repo_scan_oserror(tmp_path):
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath=str(tmp_path / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    bad_file = tmp_path / "test_bad.py"
    bad_file.write_text(
        '@patch("pkg.orig.use_fn")\ndef test_f(): helper()\n', encoding="utf-8"
    )
    bad_file.chmod(0o000)
    try:
        msgs = list(apply_patch_callgraph([ctx], {}, str(tmp_path)))
        assert msgs == []
    finally:
        bad_file.chmod(0o644)


def test_apply_patch_callgraph_repo_scan_file_no_change(tmp_path):
    test_file = tmp_path / "helper.py"
    test_file.write_text(
        "# references pkg.orig.use_fn in a comment\nx = 1\n",
        encoding="utf-8",
    )
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath=str(tmp_path / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    msgs = list(apply_patch_callgraph([ctx], {}, str(tmp_path)))
    assert msgs == []
    assert "x = 1" in test_file.read_text(encoding="utf-8")


def test_apply_patch_callgraph_excluded_dirs(tmp_path):
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    excluded_file = venv_dir / "test_something.py"
    excluded_file.write_text(
        '@patch("pkg.orig.use_fn")\ndef test_f(): helper()\n', encoding="utf-8"
    )
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath=str(tmp_path / "orig.py"),
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    msgs = list(apply_patch_callgraph([ctx], {}, str(tmp_path)))
    assert '@patch("pkg.orig.use_fn")' in excluded_file.read_text(encoding="utf-8")
    assert msgs == []
