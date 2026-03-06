from __future__ import annotations
import pytest
from crispen.errors import CrispenAPIError
from crispen.file_limiter.advisor import advise_file_limiter
from .test_advisor_plan_core import _CONFIG, _classified, _make_entity


def test_plan_api_key_error_propagates(monkeypatch):
    """Missing API key raises CrispenAPIError before any LLM call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = _classified(
        entities=[_make_entity("foo", 1, 5)],
        set_2_groups=[["foo"]],
    )
    with pytest.raises(CrispenAPIError):
        advise_file_limiter(c, "src/big.py", _CONFIG)
