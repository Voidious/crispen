from __future__ import annotations
from crispen.patch_rewriter import _CgIndex, _FLContext


def _make_cuf_contexts() -> list:
    """FL context with placement (helper) and conflict (resolve) using use_fn."""
    return [
        _FLContext(
            filepath="/proj/pkg/orig.py",
            old_module="pkg.orig",
            original_source="from external import use_fn\ndef helper(): use_fn()\n",
            modified_source="from .placement import helper\n",
            new_files={
                "placement.py": (
                    "from external import use_fn\ndef helper(): use_fn()\n"
                ),
                "conflict.py": (
                    "from external import use_fn\ndef resolve(): use_fn()\n"
                ),
            },
            new_module_paths={
                "placement.py": "pkg.placement",
                "conflict.py": "pkg.conflict",
            },
            entity_to_target={"helper": "placement.py"},
            forking_old_paths={"pkg.orig.use_fn"},
        )
    ]


def _make_cuf_index(scan_abs: str, test_src: str) -> _CgIndex:
    """Minimal index for _callgraph_update_file: maps scan_abs → 'pkg.test_mod'."""
    return _CgIndex(
        module_to_source={"pkg.test_mod": test_src},
        module_to_package={"pkg.test_mod": "pkg"},
        module_to_defs={"pkg.test_mod": set()},
        file_to_module={scan_abs: "pkg.test_mod"},
    )
