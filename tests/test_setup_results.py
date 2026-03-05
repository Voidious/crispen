from __future__ import annotations
from typing import Any
from dataclasses import dataclass


@dataclass
class GenerateFileSplitsSetupResult:
    e_foo: Any
    e_bar: Any
    c: Any
    plan: Any
    result: Any


@dataclass
class SetupAddReExportsTestResult:
    source: Any
    entity: Any
    placement: Any
    result: Any
