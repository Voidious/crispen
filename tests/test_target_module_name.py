from __future__ import annotations
from crispen.file_limiter.code_gen import _target_module_name


def test_target_module_name_simple():
    assert _target_module_name("utils.py") == "utils"


def test_target_module_name_nested():
    assert _target_module_name("helpers/io.py") == "helpers.io"
