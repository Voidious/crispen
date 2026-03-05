from __future__ import annotations
from .models import _FunctionInfo, _SeqInfo


def _generate_no_arg_call(seq: _SeqInfo, func: _FunctionInfo) -> str:
    """Algorithmically generate a no-argument call to *func*, preserving indentation."""
    first_line = seq.source.splitlines()[0]
    indent = first_line[: len(first_line) - len(first_line.lstrip())]
    return indent + func.name + "()\n"
