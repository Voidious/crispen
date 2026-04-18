from __future__ import annotations
from crispen.patch_rewriter import (
    _CG_MAX_DEPTH,
    _CG_MAX_MODULES,
    _CgIndex,
    _FLContext,
    _cg_collect_defined_names,
    _resolve_forking_path_via_callgraph,
)


def _make_bfs_ctx() -> _FLContext:
    """Context with placement.py (helper) and conflict.py (resolve) using use_fn."""
    return _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="from .placement import helper\n",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
            "conflict.py": "from external import use_fn\ndef resolve(): use_fn()\n",
        },
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={"helper": "placement.py"},
        forking_old_paths={"pkg.orig.use_fn"},
    )


def _make_bfs_index(test_src: str, calling_module: str = "pkg.test_mod") -> _CgIndex:
    """Minimal index: only the calling module's source (for import resolution)."""
    parts = calling_module.split(".")
    pkg = ".".join(parts[:-1]) if len(parts) > 1 else ""
    return _CgIndex(
        module_to_source={calling_module: test_src},
        module_to_package={calling_module: pkg},
        module_to_defs={calling_module: set()},
        file_to_module={},
    )


def test_resolve_callgraph_no_calling_module():
    ctx = _make_bfs_ctx()
    index = _make_bfs_index("from pkg.placement import helper\n")
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper()\n", ctx, index, ""
    )
    assert result is None


def test_resolve_callgraph_pre_check_fails():
    # original_source has no external import of 'use_fn' → pre-check fails
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="def helper(): use_fn()\n",  # not imported externally
        modified_source="",
        new_files={"placement.py": "def helper(): use_fn()\n"},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    index = _make_bfs_index("from pkg.placement import helper\n")
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_no_terminal():
    # New files don't reference 'use_fn' → terminal empty → None
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": "def helper(): pass\n"},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    index = _make_bfs_index("from pkg.placement import helper\n")
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_direct_call():
    # Test directly calls 'helper'; helper in placement uses use_fn.
    ctx = _make_bfs_ctx()
    index = _make_bfs_index("from pkg.placement import helper\n")
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert result == "pkg.placement.use_fn"


def test_resolve_callgraph_multi_hop():
    # Test → intermediary → helper → terminal (placement.use_fn)
    ctx = _make_bfs_ctx()
    middle_src = "from pkg.placement import helper\ndef intermediary(): helper()\n"
    test_src = "from pkg.middle import intermediary\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.middle": middle_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.middle": "pkg"},
        module_to_defs={"pkg.test_mod": set(), "pkg.middle": {"intermediary"}},
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): intermediary()\n", ctx, index, "pkg.test_mod"
    )
    assert result == "pkg.placement.use_fn"


def test_resolve_callgraph_reexport():
    # Test imports helper from pkg.orig; pkg.orig re-exports helper from placement.
    # Re-export is followed without incrementing depth.
    ctx = _make_bfs_ctx()
    orig_src = "from .placement import helper\n"  # re-exports (fn not defined)
    test_src = "from pkg.orig import helper\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.orig": orig_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.orig": "pkg"},
        module_to_defs={"pkg.test_mod": set(), "pkg.orig": set()},  # helper not defined
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert result == "pkg.placement.use_fn"


def test_resolve_callgraph_multiple_candidates():
    # Both placement.helper and conflict.resolve are reachable → ambiguous → None.
    ctx = _make_bfs_ctx()
    test_src = "from pkg.placement import helper\n" "from pkg.conflict import resolve\n"
    index = _make_bfs_index(test_src)
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper(); resolve()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_not_reachable():
    # Test doesn't import anything relevant → BFS queue empty → None.
    ctx = _make_bfs_ctx()
    index = _make_bfs_index("")  # no imports
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): unrelated()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_new_submodule_non_terminal():
    # Test imports 'other' from placement; 'other' doesn't use use_fn.
    # 'other' is in a new sub-module but NOT in terminal → skipped.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": (
                "from external import use_fn\n"
                "def helper(): use_fn()\n"
                "def other(): pass\n"
            )
        },
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = "from pkg.placement import other\n"
    index = _make_bfs_index(test_src)
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): other()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_init_reexport():
    # pkg/orig.py split into pkg/orig/__init__.py (re-exports helper) and
    # pkg/orig/placement.py (defines helper, uses use_fn).
    # Test imports helper from pkg.orig (the new __init__).
    # __init__ is excluded from new_module_set so BFS traverses through it
    # and follows the re-export to placement.py, finding the terminal.
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "orig/__init__.py": "from .placement import helper\n",
            "orig/placement.py": (
                "from external import use_fn\ndef helper(): use_fn()\n"
            ),
        },
        new_module_paths={
            "orig/__init__.py": "pkg.orig",
            "orig/placement.py": "pkg.orig.placement",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    init_src = "from .placement import helper\n"
    test_src = "from pkg.orig import helper\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.orig": init_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.orig": "pkg.orig"},
        module_to_defs={"pkg.test_mod": set(), "pkg.orig": set()},
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert result == "pkg.orig.placement.use_fn"


def test_resolve_callgraph_visited_dedup():
    # 'intermediary' and 'inter2' both map to same (module, func); processed once.
    ctx = _make_bfs_ctx()
    middle_src = "from pkg.placement import helper\ndef intermediary(): helper()\n"
    test_src = (
        "from pkg.middle import intermediary\n"
        "from pkg.middle import intermediary as inter2\n"
    )
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.middle": middle_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.middle": "pkg"},
        module_to_defs={"pkg.test_mod": set(), "pkg.middle": {"intermediary"}},
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn",
        "def test_f(): intermediary(); inter2()\n",
        ctx,
        index,
        "pkg.test_mod",
    )
    assert result == "pkg.placement.use_fn"


def test_resolve_callgraph_empty_new_file():
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "empty.py": "",
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
        },
        new_module_paths={"empty.py": "pkg.empty", "placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    index = _make_bfs_index("from pkg.placement import helper\n")
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): helper()\n", ctx, index, "pkg.test_mod"
    )
    assert result == "pkg.placement.use_fn"


def test_resolve_callgraph_missing_module_path():
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": "from external import use_fn\ndef f(): use_fn()\n"},
        new_module_paths={},  # missing → terminal empty → None
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    index = _make_bfs_index("from pkg.placement import f\n")
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): f()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_missing_source():
    # Called function's module is not in the index → src=None → continue
    ctx = _make_bfs_ctx()
    test_src = "from pkg.missing import something\n"
    index = _make_bfs_index(test_src)
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): something()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_func_defined_no_calls():
    # Function IS defined but has no imported calls → BFS dead-end → None
    ctx = _make_bfs_ctx()
    middle_src = "def standalone(): pass\n"
    test_src = "from pkg.middle import standalone\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.middle": middle_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.middle": "pkg"},
        module_to_defs={"pkg.test_mod": set(), "pkg.middle": {"standalone"}},
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): standalone()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_body_call_not_importable():
    # Function's body calls something not in its import map → BFS dead-end → None
    ctx = _make_bfs_ctx()
    middle_src = "def fn(): bar()\n"  # bar not imported
    test_src = "from pkg.middle import fn\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.middle": middle_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.middle": "pkg"},
        module_to_defs={"pkg.test_mod": set(), "pkg.middle": {"fn"}},
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): fn()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_func_not_defined_not_reexported():
    # func_name not defined and not re-exported → BFS dead-end → None
    ctx = _make_bfs_ctx()
    middle_src = "def other(): pass\n"  # 'fn' not defined, not re-exported
    test_src = "from pkg.middle import fn\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.middle": middle_src},
        module_to_package={"pkg.test_mod": "pkg", "pkg.middle": "pkg"},
        module_to_defs={"pkg.test_mod": set(), "pkg.middle": {"other"}},
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): fn()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_body_call_already_visited():
    # fn_a → fn_b → fn_a (mutual recursion); fn_a already visited when fn_b adds it
    ctx = _make_bfs_ctx()
    m_a = "from pkg.m_b import fn_b\ndef fn_a(): fn_b()\n"
    m_b = "from pkg.m_a import fn_a\ndef fn_b(): fn_a()\n"
    test_src = "from pkg.m_a import fn_a\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.m_a": m_a, "pkg.m_b": m_b},
        module_to_package={
            "pkg.test_mod": "pkg",
            "pkg.m_a": "pkg",
            "pkg.m_b": "pkg",
        },
        module_to_defs={
            "pkg.test_mod": set(),
            "pkg.m_a": {"fn_a"},
            "pkg.m_b": {"fn_b"},
        },
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): fn_a()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_reexport_already_visited():
    # m_x re-exports fn_x from m_y; m_y re-exports fn_x from m_x (cycle).
    # When m_y checks re-export, (m_x, fn_x) is already visited → skip.
    ctx = _make_bfs_ctx()
    m_x = "from pkg.m_y import fn_x\n"  # re-exports fn_x from m_y
    m_y = "from pkg.m_x import fn_x\n"  # re-exports fn_x from m_x (cycle)
    test_src = "from pkg.m_x import fn_x\n"
    index = _CgIndex(
        module_to_source={"pkg.test_mod": test_src, "pkg.m_x": m_x, "pkg.m_y": m_y},
        module_to_package={
            "pkg.test_mod": "pkg",
            "pkg.m_x": "pkg",
            "pkg.m_y": "pkg",
        },
        module_to_defs={"pkg.test_mod": set(), "pkg.m_x": set(), "pkg.m_y": set()},
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): fn_x()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_depth_limit():
    # Chain of _CG_MAX_DEPTH + 1 hops; last function calls terminal but is cut off.
    n = _CG_MAX_DEPTH + 1  # 13 intermediate functions
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"end.py": "from external import use_fn\ndef end_fn(): use_fn()\n"},
        new_module_paths={"end.py": "pkg.end"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    modules = {}
    for i in range(n):
        if i < n - 1:
            src = f"from pkg.m{i + 1} import f{i + 1}\ndef f{i}(): f{i + 1}()\n"
        else:
            src = f"from pkg.end import end_fn\ndef f{i}(): end_fn()\n"
        modules[f"pkg.m{i}"] = src
    modules["pkg.test_mod"] = "from pkg.m0 import f0\n"
    defs = {m: _cg_collect_defined_names(s) for m, s in modules.items()}
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs=defs,
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): f0()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_module_limit():
    # Re-export chain of _CG_MAX_MODULES + 1 unique modules; 51st is cut off.
    n = _CG_MAX_MODULES  # 50 re-export hops before the cut-off
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"end.py": "from external import use_fn\ndef final(): use_fn()\n"},
        new_module_paths={"end.py": "pkg.end"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    modules = {}
    for i in range(n + 1):
        if i < n:
            modules[f"pkg.m{i}"] = f"from pkg.m{i + 1} import fn\n"
        else:
            modules[f"pkg.m{i}"] = "from pkg.end import final\ndef fn(): final()\n"
    modules["pkg.test_mod"] = "from pkg.m0 import fn\n"
    defs = {m: _cg_collect_defined_names(s) for m, s in modules.items()}
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs=defs,
        file_to_module={},
    )
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): fn()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None


def test_resolve_callgraph_same_module_twice():
    """Two called names both resolve to the same intermediate module.

    The second BFS entry hits the 'module already in modules_seen' fast path
    (branch 1250->1255 in patch_rewriter.py).
    """
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"end.py": "from external import use_fn\ndef final(): use_fn()\n"},
        new_module_paths={"end.py": "pkg.end"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    # Both fn_a and fn_b live in pkg.middle; neither calls anything reachable.
    middle_src = "def fn_a(): pass\ndef fn_b(): pass\n"
    modules = {
        "pkg.test_mod": "from pkg.middle import fn_a, fn_b\n",
        "pkg.middle": middle_src,
    }
    defs = {m: _cg_collect_defined_names(s) for m, s in modules.items()}
    index = _CgIndex(
        module_to_source=modules,
        module_to_package={m: "pkg" for m in modules},
        module_to_defs=defs,
        file_to_module={},
    )
    # BFS enqueues (pkg.middle, fn_a, 0) and (pkg.middle, fn_b, 0).
    # First pop adds pkg.middle to modules_seen; second pop hits the fast path.
    result = _resolve_forking_path_via_callgraph(
        "use_fn", "def test_f(): fn_a(); fn_b()\n", ctx, index, "pkg.test_mod"
    )
    assert result is None
