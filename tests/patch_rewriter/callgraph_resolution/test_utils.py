from __future__ import annotations
from crispen.patch_rewriter import (
    _CG_MAX_DEPTH,
    _CG_MAX_MODULES,
    _CgIndex,
    _FLContext,
    _cg_collect_defined_names,
    _expand_module_terminals,
    _resolve_forking_path_candidates,
)
from .utils import _make_bfs_ctx, _make_bfs_index


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


def test_expand_module_terminals_syntax_error():
    # Unparseable source → silently returns without modifying terminal.
    terminal: dict = {("pkg.mod", "a"): "pkg.mod.use_fn"}
    _expand_module_terminals("def (broken\n", "pkg.mod", "use_fn", terminal)
    # Only the original entry remains.
    assert list(terminal.keys()) == [("pkg.mod", "a")]


def test_expand_module_terminals_unrelated_module():
    # Direct terminal is in a different module → nothing added for pkg.other.
    terminal: dict = {("pkg.mod", "a"): "pkg.mod.use_fn"}
    _expand_module_terminals("def b(): a()\n", "pkg.other", "use_fn", terminal)
    # b is in pkg.other which has no direct terminals → not added.
    assert ("pkg.other", "b") not in terminal


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
