from __future__ import annotations


def _parse_validation_result(result):
    if result is not None:
        return (
            result["is_valid_duplicate"],
            result.get("reason", ""),
            result.get("extraction_notes", ""),
        )
    return False, "no tool response", ""
