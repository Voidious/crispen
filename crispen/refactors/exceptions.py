from __future__ import annotations


class _ApiTimeout(Exception):
    """Raised when an LLM API call exceeds the hard per-call timeout."""
