from __future__ import annotations


def _propose_ok(*filenames: str) -> dict:
    """Return a valid propose_output_files LLM response for the given filenames."""
    return {
        "files": [{"filename": f, "description": "auto-generated"} for f in filenames]
    }
