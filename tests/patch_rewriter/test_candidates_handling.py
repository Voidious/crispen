from __future__ import annotations
from crispen.patch_rewriter import (
    _CG_MAX_DEPTH,
    _CG_MAX_MODULES,
    _CgIndex,
    _FLContext,
    _candidates_check,
    _cg_collect_defined_names,
    _expand_module_terminals,
    _resolve_forking_path_candidates,
    _rewrite_candidates_check,
)
from .test_callgraph_resolution import _make_bfs_ctx, _make_bfs_index


def test_resolve_forking_path_candidates_single():
    # Single candidate: path returned, candidates=[path], truncated=False.
    ctx = _make_bfs_ctx()
    index = _make_bfs_index("from pkg.placement import helper\n")
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert path == "pkg.placement.use_fn"
    assert cands == ["pkg.placement.use_fn"]
    assert not truncated


def test_resolve_forking_path_candidates_multiple():
    # Multiple candidates → path=None, cands=[...], truncated=False.
    ctx = _make_bfs_ctx()
    test_src = "from pkg.placement import helper\nfrom pkg.conflict import resolve\n"
    index = _make_bfs_index(test_src)
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): helper(); resolve()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    assert path is None
    assert sorted(cands) == ["pkg.conflict.use_fn", "pkg.placement.use_fn"]
    assert not truncated


def test_resolve_forking_path_candidates_no_calling_module():
    ctx = _make_bfs_ctx()
    index = _make_bfs_index("from pkg.placement import helper\n")
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn", "def test_f(): helper()\n", ctx, index, ""
    )
    assert path is None
    assert cands == []
    assert not truncated


def test_resolve_forking_path_candidates_truncated_depth():
    # Chain of exactly _CG_MAX_DEPTH + 1 hops; the last hop is cut off → truncated=True.
    # Chain: test_mod -[f0]-> mid0 -> mid1 -> ... -> mid{n-1} -[helper]-> placement
    # n = _CG_MAX_DEPTH + 1 intermediate modules; helper is at depth n-1 = 13,
    # but the depth limit cuts off at depth 12 before enqueuing helper.
    n = _CG_MAX_DEPTH + 1  # 13 hops from test_mod to placement
    ctx = _make_bfs_ctx()
    all_src: dict = {}
    all_src["pkg.test_mod"] = "from pkg.mid0 import f0\n"
    for i in range(n):
        caller = f"f{i}"
        if i < n - 1:
            callee = f"f{i + 1}"
            callee_mod = f"pkg.mid{i + 1}"
        else:
            callee = "helper"
            callee_mod = "pkg.placement"
        all_src[f"pkg.mid{i}"] = (
            f"from {callee_mod} import {callee}\n" f"def {caller}(): {callee}()\n"
        )
    all_src["pkg.placement"] = "from external import use_fn\ndef helper(): use_fn()\n"
    index = _CgIndex(
        module_to_source=all_src,
        module_to_package={m: "pkg" for m in all_src},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in all_src.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn", "def test_f(): f0()\n", ctx, index, "pkg.test_mod"
    )
    assert path is None
    assert truncated


def test_resolve_forking_path_candidates_truncated_modules():
    # Re-export chain of _CG_MAX_MODULES + 1 intermediate modules; the last one
    # is cut off before pkg.placement (a terminal) is ever reached.
    n = _CG_MAX_MODULES + 1
    ctx = _make_bfs_ctx()
    src_map: dict = {}
    for i in range(n):
        next_mod = f"pkg.m{i + 1}" if i < n - 1 else "pkg.placement"
        src_map[f"pkg.m{i}"] = f"from {next_mod} import helper\n"
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    src_map["pkg.placement"] = placement_src
    test_src = "from pkg.m0 import helper\n"
    all_src = {"pkg.test_mod": test_src, **src_map}
    index = _CgIndex(
        module_to_source=all_src,
        module_to_package={m: "pkg" for m in all_src},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in all_src.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert path is None
    assert truncated


def test_resolve_forking_path_candidates_original_module_only():
    # modified_source still has a function using use_fn; no new sub-file uses it.
    # → only terminal is (pkg.orig, func_a) → unique resolution to original path.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source=("from external import use_fn\n" "def func_a(): use_fn()\n"),
        new_files={
            "placement.py": "from external import other\ndef helper(): other()\n"
        },
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    orig_src = ctx.modified_source
    test_src = "from pkg.orig import func_a\n"
    modules = {"pkg.test_mod": test_src, "pkg.orig": orig_src}
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn", "def test_f(): func_a()\n", ctx, index, "pkg.test_mod"
    )
    assert path == "pkg.orig.use_fn"
    assert cands == ["pkg.orig.use_fn"]
    assert not truncated


def test_resolve_forking_path_candidates_original_and_new_both_candidates():
    # modified_source keeps func_a (uses use_fn); placement.py moves func_b
    # (also uses use_fn).  Test calls both → 2 candidates → ambiguous.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source=("from external import use_fn\n" "def func_a(): use_fn()\n"),
        new_files={
            "placement.py": (
                "from external import use_fn\n" "def func_b(): use_fn()\n"
            ),
        },
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    orig_src = ctx.modified_source
    placement_src = ctx.new_files["placement.py"]
    test_src = "from pkg.orig import func_a\nfrom pkg.placement import func_b\n"
    modules = {
        "pkg.test_mod": test_src,
        "pkg.orig": orig_src,
        "pkg.placement": placement_src,
    }
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): func_a(); func_b()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    assert path is None  # ambiguous
    assert sorted(cands) == ["pkg.orig.use_fn", "pkg.placement.use_fn"]
    assert not truncated


def test_expand_module_terminals_no_direct():
    # No direct terminal in this module → nothing added.
    terminal: dict = {}
    _expand_module_terminals(
        "def a(): b()\ndef b(): pass\n", "pkg.mod", "use_fn", terminal
    )
    assert terminal == {}


def test_expand_module_terminals_direct_only():
    # A direct terminal is seeded before calling; only transitive callers are added.
    terminal: dict = {("pkg.mod", "b"): "pkg.mod.use_fn"}
    _expand_module_terminals(
        "def a(): b()\ndef b(): use_fn()\n", "pkg.mod", "use_fn", terminal
    )
    # a calls b (direct terminal) → a becomes transitive terminal.
    assert terminal[("pkg.mod", "a")] == "pkg.mod.use_fn"
    # Original direct entry unchanged.
    assert terminal[("pkg.mod", "b")] == "pkg.mod.use_fn"


def test_expand_module_terminals_multi_level():
    # c → b → a (direct); all three end up in terminal.
    terminal: dict = {("pkg.mod", "a"): "pkg.mod.use_fn"}
    src = "def a(): use_fn()\ndef b(): a()\ndef c(): b()\n"
    _expand_module_terminals(src, "pkg.mod", "use_fn", terminal)
    assert ("pkg.mod", "b") in terminal
    assert ("pkg.mod", "c") in terminal


def test_resolve_forking_path_candidates_intra_module_chain():
    # BFS follows locally-defined calls within a non-terminal intermediate module.
    # Chain: test_mod → pkg.service.public_func
    #                        (local) ↓
    #                   pkg.service._local_helper
    #                        (import) ↓
    #                   pkg.placement.use_target  ← terminal (calls use_fn)
    #
    # pkg.service is neither orig nor a new sub-file, so _expand_module_terminals
    # never seeds it.  The elif branch in the BFS must queue _local_helper from
    # public_func's body so we eventually reach the terminal in pkg.placement.
    placement_src = "from external import use_fn\ndef use_target(): use_fn()\n"
    service_src = (
        "from pkg.placement import use_target\n"
        "def _local_helper(): use_target()\n"
        "def public_func(): _local_helper()\n"
    )
    ctx2 = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="from .placement import use_target\n",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={"use_target": "placement.py"},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "from pkg.service import public_func\n"
    modules = {
        "pkg.test_mod": test_src,
        "pkg.service": service_src,
        "pkg.placement": placement_src,
    }
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): public_func()\n",
        ctx2,
        index,
        "pkg.test_mod",
    )
    assert path == "pkg.placement.use_fn"
    assert cands == ["pkg.placement.use_fn"]
    assert not truncated


def test_resolve_forking_path_candidates_intra_module_local_already_visited():
    # The elif branch fires but the visited guard suppresses re-queuing.
    # A recursive function calls itself: when processing its body calls, itself
    # is already in visited → (module, called_name) in visited → branch skipped.
    placement_src = "from external import use_fn\ndef use_target(): use_fn()\n"
    service_src = (
        "from pkg.placement import use_target\n"
        # recursive_func calls use_target (imported) AND itself (local, recursive)
        "def recursive_func(n): use_target() if n <= 0 else recursive_func(n-1)\n"
    )
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="from .placement import use_target\n",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={"use_target": "placement.py"},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "from pkg.service import recursive_func\n"
    modules = {
        "pkg.test_mod": test_src,
        "pkg.service": service_src,
        "pkg.placement": placement_src,
    }
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): recursive_func(5)\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    # recursive_func's body calls reach use_target (terminal in pkg.placement).
    assert path == "pkg.placement.use_fn"
    assert not truncated


def test_resolve_forking_path_candidates_new_module_intra_chain():
    # BFS follows intra-module calls within a new submodule to reach a terminal.
    # Chain: test_mod → pkg.placement.wrapper (new-module, not terminal)
    #                        (local) ↓
    #                   pkg.placement._inner  ← terminal (calls use_fn directly)
    #
    # Before the fix, the BFS hit pkg.placement in new_module_set and stopped at
    # wrapper without following _inner — no candidate was found.  After the fix,
    # it follows the local call to _inner and discovers pkg.placement.use_fn.
    placement_src = (
        "from external import use_fn\n"
        "def _inner(): use_fn()\n"
        "def wrapper(): _inner()\n"
    )
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={"wrapper": "placement.py", "_inner": "placement.py"},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "from pkg.placement import wrapper\n"
    modules = {
        "pkg.test_mod": test_src,
        "pkg.placement": placement_src,
    }
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): wrapper()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    assert path == "pkg.placement.use_fn"
    assert cands == ["pkg.placement.use_fn"]
    assert not truncated


def test_resolve_forking_path_candidates_new_module_intra_chain_cycle():
    # Intra-module traversal inside a new submodule respects the visited guard:
    # a mutually recursive pair (a calls b, b calls a) does not loop.
    placement_src = (
        "from external import use_fn\n"
        "def _inner(): use_fn()\n"
        "def a(): b()\n"
        "def b(): a(); _inner()\n"  # b is terminal (uses use_fn via _inner)
    )
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={"a": "placement.py"},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "from pkg.placement import a\n"
    modules = {
        "pkg.test_mod": test_src,
        "pkg.placement": placement_src,
    }
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): a()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    assert path == "pkg.placement.use_fn"
    assert not truncated


def test_resolve_forking_path_candidates_new_module_cross_module_terminal():
    # BFS follows cross-module calls FROM a terminal function inside a new submodule.
    # Scenario:
    #   pkg.main:  _run_step() calls use_fn() directly (terminal)
    #              orchestrate() calls _run_step() [local] + do_step() [pkg.steps]
    #              _expand_module_terminals makes orchestrate terminal for pkg.main.use_fn  # noqa: E501
    #   pkg.steps: do_step() calls use_fn() (terminal for pkg.steps.use_fn)
    #
    # When BFS hits (pkg.main, orchestrate) — which IS in terminal — it should
    # record pkg.main.use_fn AND then follow the cross-module call to
    # (pkg.steps, do_step), discovering pkg.steps.use_fn as a second candidate.
    # Line 1472 in the BFS is covered only by this cross-module append.
    main_src = (
        "from external import use_fn\n"
        "from pkg.steps import do_step\n"
        "def _run_step(): use_fn()\n"
        "def orchestrate(): _run_step(); do_step()\n"
    )
    steps_src = "from external import use_fn\n" "def do_step(): use_fn()\n"
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"main.py": main_src, "steps.py": steps_src},
        new_module_paths={"main.py": "pkg.main", "steps.py": "pkg.steps"},
        entity_to_target={
            "_run_step": "main.py",
            "orchestrate": "main.py",
            "do_step": "steps.py",
        },
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "from pkg.main import orchestrate\n"
    modules = {
        "pkg.test_mod": test_src,
        "pkg.main": main_src,
        "pkg.steps": steps_src,
    }
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): orchestrate()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    # Both submodules use use_fn — two candidates, no single resolved path.
    assert path is None
    assert sorted(cands) == ["pkg.main.use_fn", "pkg.steps.use_fn"]
    assert not truncated


def test_resolve_forking_path_candidates_import_alias_direct():
    # Test uses ``import pkg.placement as pl; pl.helper()`` to call the terminal.
    # The BFS must follow ``pl.helper`` by resolving alias ``pl`` → ``pkg.placement``.
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="from .placement import helper\n",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={"helper": "placement.py"},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "import pkg.placement as pl\n"
    # Import map: "pl" → ("pkg.placement", "pkg.placement"); call "pl.helper()"
    # → _cg_collect_called_names emits "pl.helper"; BFS resolves alias pl →
    # module pkg.placement, queues (pkg.placement, "helper") → terminal hit.
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src},
        module_to_package={"pkg.test_mod": "pkg"},
        module_to_defs={"pkg.test_mod": set()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): pl.helper()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    assert path == "pkg.placement.use_fn"
    assert cands == ["pkg.placement.use_fn"]
    assert not truncated


def test_resolve_forking_path_candidates_body_call_via_alias():
    # An intermediate function uses ``mod.helper()`` (module alias) to reach
    # the terminal.  The BFS body-call step must follow the alias.
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    service_src = "import pkg.placement as pl\n" "def public_func(): pl.helper()\n"
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="from .placement import helper\n",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={"helper": "placement.py"},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "from pkg.service import public_func\n"
    modules = {
        "pkg.test_mod": test_src,
        "pkg.service": service_src,
        "pkg.placement": placement_src,
    }
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs={m: _cg_collect_defined_names(s) for m, s in modules.items()},
        file_to_module={},
    )
    path, cands, truncated, _static = _resolve_forking_path_candidates(
        "use_fn",
        "def test_f(): public_func()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    assert path == "pkg.placement.use_fn"
    assert cands == ["pkg.placement.use_fn"]
    assert not truncated


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
