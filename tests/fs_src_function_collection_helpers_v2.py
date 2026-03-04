from __future__ import annotations
import libcst as cst
from libcst.metadata import MetadataWrapper
from crispen.refactors.function_splitter import _FunctionCollector


def _collect_functions_from_src(src: str, expected_count: int = 0):
    tree = cst.parse_module(src)
    wrapper = MetadataWrapper(tree)
    collector = _FunctionCollector()
    wrapper.visit(collector)
    assert len(collector.functions) == expected_count
    return collector
