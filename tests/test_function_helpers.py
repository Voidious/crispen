import libcst as cst
from libcst.metadata import MetadataWrapper
from crispen.refactors.duplicate_extractor import _FunctionCollector


def _collect_functions(source: str):
    tree = cst.parse_module(source)
    lines = source.splitlines(keepends=True)
    collector = _FunctionCollector(lines)
    MetadataWrapper(tree).visit(collector)
    return collector.functions
