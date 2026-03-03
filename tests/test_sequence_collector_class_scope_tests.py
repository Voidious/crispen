import textwrap
import libcst as cst
from libcst.metadata import MetadataWrapper
from crispen.refactors.duplicate_extractor import _SequenceCollector


def test_sequence_collector_class_scope():
    """_SequenceCollector sets class_scope for sequences inside class methods."""
    import libcst as cst
    from libcst.metadata import MetadataWrapper

    from crispen.refactors.duplicate_extractor import _SequenceCollector

    source = textwrap.dedent(
        """\
        x = 1
        y = 2
        z = 3

        class MyClass:
            def method(self):
                a = 1
                b = 2
                c = 3
        """
    )
    lines = source.splitlines(keepends=True)
    tree = cst.parse_module(source)
    collector = _SequenceCollector(lines, max_seq_len=8)
    MetadataWrapper(tree).visit(collector)

    module_seqs = [s for s in collector.sequences if s.class_scope is None]
    class_seqs = [s for s in collector.sequences if s.class_scope == "MyClass"]
    assert module_seqs, "expected module-level sequences with class_scope=None"
    assert class_seqs, "expected class-method sequences with class_scope='MyClass'"
