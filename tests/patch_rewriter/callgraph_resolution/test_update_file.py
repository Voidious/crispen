from __future__ import annotations
from crispen.patch_rewriter import _FLContext, _callgraph_update_file, _cg_build_index
from .utils import _make_cuf_contexts, _make_cuf_index


def test_callgraph_update_file_no_functions():
    src = "x = 1\n"
    result, changed, _unresolved = _callgraph_update_file(
        src, {"pkg.orig.use_fn"}, _make_cuf_contexts()
    )
    assert not changed
    assert result == src


def test_callgraph_update_file_index_none(tmp_path):
    # index=None → BFS skipped → no resolution even if test calls helper.
    test_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=None,
    )
    assert not changed


def test_callgraph_update_file_string_literal_resolved(tmp_path):
    test_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.placement.use_fn")' in result


def test_callgraph_update_file_no_resolution(tmp_path):
    # Test calls 'unrelated' — not imported → BFS queue empty → no resolution.
    # Static fallback has 2 candidates (placement + conflict) → unresolved saved.
    test_src = (
        '@patch("pkg.orig.use_fn")\n' "def test_f(mock_use_fn):\n" "    unrelated()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert not changed
    cands = unresolved.get("test_f", {}).get("pkg.orig.use_fn", [])
    assert sorted(cands) == ["pkg.conflict.use_fn", "pkg.placement.use_fn"]


def test_callgraph_update_file_zero_cands_single_static_auto_resolve(tmp_path):
    # BFS finds 0 candidates but static terminal has exactly 1 → auto-resolve.
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        '@patch("pkg.orig.use_fn")\n' "def test_f(mock_use_fn):\n" "    unrelated()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.placement.use_fn")' in result
    assert "test_f" not in unresolved


def test_callgraph_update_file_zero_cands_single_static_clears_unresolved(tmp_path):
    # ctx_ambig: BFS finds 2 candidates (saves to unresolved).
    # ctx_uniq_static: BFS finds 0, static has 1 → auto-resolves AND clears the
    # previously saved unresolved entry (exercises the delete-entry branch).
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
        "    resolve()\n"
    )
    ctx_ambig = _make_cuf_contexts()[0]  # placement + conflict → 2 BFS candidates
    ctx_uniq_static = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"singleton.py": "from external import use_fn\ndef fn(): use_fn()\n"},
        new_module_paths={"singleton.py": "pkg.singleton"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    scan = str(tmp_path / "test_foo.py")
    # Index only knows pkg.test_mod; pkg.placement/conflict have no source so
    # ctx_uniq_static's BFS reaches 0 candidates while static_cands = 1.
    index = _make_cuf_index(scan, test_src)
    result, changed, unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx_ambig, ctx_uniq_static],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.singleton.use_fn")' in result
    assert "test_f" not in unresolved  # static single-cand cleared the entry


def test_callgraph_update_file_const_ref_unanimous(tmp_path):
    test_src = (
        "from pkg.placement import helper\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_a(mock_use_fn):\n"
        "    helper()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_b(mock_use_fn):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '_PATCH_USE = "pkg.placement.use_fn"' in result
    assert "@patch(_PATCH_USE)" in result


def test_callgraph_update_file_const_ref_conflicting(tmp_path):
    # test_a: helper() → placement; test_b: resolve() → conflict → conflicting.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
            "conflict.py": "from external import use_fn\ndef resolve(): use_fn()\n",
        },
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_a(mock_use_fn):\n"
        "    helper()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_b(mock_use_fn):\n"
        "    resolve()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src, {"pkg.orig.use_fn"}, [ctx], scan_file=scan, index=index
    )
    assert changed
    assert '_PATCH_USE = "pkg.orig.use_fn"' in result  # const def unchanged
    assert '@patch("pkg.placement.use_fn")' in result  # test_a inlined
    assert '@patch("pkg.conflict.use_fn")' in result  # test_b inlined


def test_callgraph_update_file_non_forking_path_skipped(tmp_path):
    test_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.stable.some_func")\n'
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn, mock_some):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.placement.use_fn")' in result
    assert '@patch("pkg.stable.some_func")' in result  # unchanged


def test_callgraph_update_file_multi_context_second_matches(tmp_path):
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx_a = _FLContext(
        filepath="/proj/pkg/other.py",
        old_module="pkg.other",
        original_source="from external import other_fn\n",
        modified_source="",
        new_files={},
        new_module_paths={},
        entity_to_target={},
        forking_old_paths={"pkg.other.other_fn"},
    )
    ctx_b = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn", "pkg.other.other_fn"},
        [ctx_a, ctx_b],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.placement.use_fn")' in result


def test_callgraph_update_file_const_ref_no_resolution_passthrough(tmp_path):
    test_src = (
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_f(mock_use_fn):\n"
        "    unrelated()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert not changed
    assert '_PATCH_USE = "pkg.orig.use_fn"' in result


def test_callgraph_update_file_const_ref_passthrough_single_proposal_updates_const(
    tmp_path,
):
    # test_a: BFS fails (calls unrelated()) → passthrough (if not resolved → continue).
    # test_b: BFS → placement → single proposal for _PATCH_USE.
    # Old: passthrough + single proposal → conflicting → inline test_b.
    # New: single proposal (passthrough no longer blocks) → const def updated, no
    #   inline.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
        },
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_a(m):\n"
        "    unrelated()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_b(m):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src, {"pkg.orig.use_fn"}, [ctx], scan_file=scan, index=index
    )
    assert changed
    # Const definition updated (single proposal, passthrough no longer blocks).
    assert '_PATCH_USE = "pkg.placement.use_fn"' in result
    # Decorators stay as const refs — no per-function inlining.
    assert "@patch(_PATCH_USE)" in result
    assert '@patch("pkg.placement.use_fn")' not in result


def test_callgraph_update_file_const_ref_partial_resolution(tmp_path):
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn, other_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n"
        },
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn", "pkg.orig.other_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        '_PATCH_OTHER = "pkg.orig.other_fn"\n'
        "@patch(_PATCH_OTHER)\n"
        "@patch(_PATCH_USE)\n"
        "def test_f(mock_use, mock_other):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn", "pkg.orig.other_fn"},
        [ctx],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '_PATCH_USE = "pkg.placement.use_fn"' in result
    assert '_PATCH_OTHER = "pkg.orig.other_fn"' in result


def test_callgraph_update_file_inline_no_inline_subs_continue(tmp_path):
    # test_a: string literal (no const_refs → inline_subs empty → continue)
    # test_b: const ref → placement; test_c: const ref → conflict → conflicting
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
            "conflict.py": "from external import use_fn\ndef resolve(): use_fn()\n",
        },
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_a(m):\n"
        "    helper()\n"
        "\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_b(m):\n"
        "    helper()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_c(m):\n"
        "    resolve()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src, {"pkg.orig.use_fn"}, [ctx], scan_file=scan, index=index
    )
    assert changed
    assert '@patch("pkg.placement.use_fn")' in result  # test_a updated
    assert '@patch("pkg.conflict.use_fn")' in result  # test_c inlined


def test_callgraph_update_file_inline_ref_from_different_file(tmp_path):
    # Const ref from constants.py (≠ scan_file) → inline_subs empty → no change.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
            "conflict.py": "from external import use_fn\ndef resolve(): use_fn()\n",
        },
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('_PATCH_USE = "pkg.orig.use_fn"\n', encoding="utf-8")
    test_src = (
        "from constants import _PATCH_USE\n"
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        "@patch(_PATCH_USE)\n"
        "def test_b(m):\n"
        "    helper()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_c(m):\n"
        "    resolve()\n"
    )
    scan_file = tmp_path / "test_cases.py"
    scan_file.write_text(test_src, encoding="utf-8")
    scan = str(scan_file)
    # Build index from disk so file_to_module is populated for test_cases.py
    index = _cg_build_index(str(tmp_path), {}, [ctx])
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx],
        scan_file=scan,
        repo_root=str(tmp_path),
        index=index,
    )
    assert not changed


def test_callgraph_update_file_inline_new_val_same_as_old(tmp_path):
    # placement.py → "pkg.orig" (same as old_module); test_b→helper→same val; skipped.
    # test_c → resolve → "pkg.conflict" → different val → inlined → changed.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
            "conflict.py": "from external import use_fn\ndef resolve(): use_fn()\n",
        },
        new_module_paths={
            "placement.py": "pkg.orig",  # same as old_module → new_val == old_val
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.orig import helper\n"
        "from pkg.conflict import resolve\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_b(m):\n"
        "    helper()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_c(m):\n"
        "    resolve()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src, {"pkg.orig.use_fn"}, [ctx], scan_file=scan, index=index
    )
    assert changed  # test_c inlined to pkg.conflict.use_fn


def test_callgraph_update_file_inline_existing_splice_updated(tmp_path):
    # test_a: use_fn → func_splice; other_fn const ref conflicting → inline.
    # Inline finds existing splice and updates it.  test_b: const → new splice.
    placement_src = (
        "from external import use_fn, other_fn\n" "def helper(): use_fn(); other_fn()\n"
    )
    conflict2_src = "from external import other_fn\ndef resolve2(): other_fn()\n"
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn, other_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src, "conflict2.py": conflict2_src},
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict2.py": "pkg.conflict2",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn", "pkg.orig.other_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict2 import resolve2\n"
        '_PATCH_OTHER = "pkg.orig.other_fn"\n'
        '@patch("pkg.orig.use_fn")\n'
        "@patch(_PATCH_OTHER)\n"
        "def test_a(m_other, m_use):\n"
        "    helper()\n"
        "\n"
        "@patch(_PATCH_OTHER)\n"
        "def test_b(m_other):\n"
        "    resolve2()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn", "pkg.orig.other_fn"},
        [ctx],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert '@patch("pkg.placement.use_fn")' in result
    assert '@patch("pkg.placement.other_fn")' in result
    assert '@patch("pkg.conflict2.other_fn")' in result


def test_callgraph_update_file_verbose(tmp_path, capsys):
    test_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(mock_use_fn):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "patch_callgraph" in captured.err
