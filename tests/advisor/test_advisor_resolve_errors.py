from __future__ import annotations
import pytest
from crispen.errors import CrispenAPIError
from crispen.file_limiter.advisor import resolve_naming_conflicts
from .test_advisor_plan_core import _CONFIG, _classified, _make_entity
from .test_advisor_resolve_core import _CONFLICTING_PLACEMENTS


def test_resolve_api_key_error_propagates(monkeypatch):
    """Missing API key raises CrispenAPIError before any LLM call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = _classified(entities=[_make_entity("foo", 1, 5)])
    with pytest.raises(CrispenAPIError):
        resolve_naming_conflicts(
            _CONFLICTING_PLACEMENTS, c, "src/big.py", frozenset(), frozenset(), _CONFIG
        )
