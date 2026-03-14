"""Shared import-sorting utility."""

import re
import sys
from typing import List


def _sort_imports_pep8(imports: List[str]) -> List[str]:
    """Re-order *imports* following PEP 8: future → stdlib → third-party → local.

    Within each group the original relative order is preserved (stable sort).
    """
    stdlib = sys.stdlib_module_names  # frozenset; available since Python 3.10

    def _group(imp: str) -> int:
        if imp.startswith("from __future__"):
            return 0  # __future__
        if re.match(r"^from\s+\.", imp):
            return 3  # relative / local
        m = re.match(r"^(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", imp)
        if m and m.group(1) in stdlib:
            return 1  # stdlib
        return 2  # third-party

    return sorted(imports, key=_group)
