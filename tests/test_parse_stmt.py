from __future__ import annotations
import libcst as cst


def _parse_stmt(src: str) -> cst.BaseStatement:
    return cst.parse_module(src).body[0]
