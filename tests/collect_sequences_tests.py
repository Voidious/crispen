import libcst as cst
from libcst.metadata import MetadataWrapper
from crispen.refactors.duplicate_extractor import _SequenceCollector


def _collect_sequences(source: str, max_seq_len: int = 8):
    tree = cst.parse_module(source)
    lines = source.splitlines(keepends=True)
    collector = _SequenceCollector(lines, max_seq_len=max_seq_len)
    MetadataWrapper(tree).visit(collector)
    return collector.sequences
