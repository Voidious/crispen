"""Crispen-specific exceptions."""


class CrispenAPIError(Exception):
    """Raised when an LLM API call fails; causes the commit to be blocked.

    Callers should print the message and exit non-zero so that pre-commit
    treats the hook as failed.  To bypass: git commit --no-verify
    """
