from __future__ import annotations

"""Refactor: extract duplicate code blocks into helper functions using an LLM."""


_MODEL = "claude-sonnet-4-6"
_MIN_WEIGHT = 3
_MAX_SEQ_LEN = 8

_VETO_TOOL: dict = {
    "name": "evaluate_duplicate",
    "description": (
        "Evaluate whether code blocks are semantic duplicates worth extracting"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_valid_duplicate": {
                "type": "boolean",
                "description": (
                    "True if extracting a shared helper would improve clarity"
                ),
            },
            "reason": {"type": "string"},
            "extraction_notes": {
                "type": "string",
                "description": (
                    "If accepting, note any potential pitfalls the extraction "
                    "step should watch out for — e.g., tricky variable scoping, "
                    "mutable arguments, subtle differences in variable names, or "
                    "return-value handling. Leave empty if none."
                ),
            },
        },
        "required": ["is_valid_duplicate", "reason"],
    },
}

_VERIFY_TOOL: dict = {
    "name": "verify_extraction",
    "description": "Verify that an extracted helper function is semantically correct",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_correct": {
                "type": "boolean",
                "description": (
                    "True if the extraction is semantically equivalent to the originals"
                ),
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Specific issues found. Empty if correct. Each issue should "
                    "describe what is wrong and how the extraction should be fixed."
                ),
            },
        },
        "required": ["is_correct", "issues"],
    },
}

_EXTRACT_TOOL: dict = {
    "name": "extract_helper",
    "description": "Extract duplicate code blocks into a helper function",
    "input_schema": {
        "type": "object",
        "properties": {
            "function_name": {"type": "string"},
            "placement": {
                "type": "string",
                "description": (
                    "Where to place the helper: 'module_level' or "
                    "'staticmethod:ClassName'"
                ),
            },
            "helper_source": {
                "type": "string",
                "description": "Complete source of the helper function",
            },
            "call_site_replacements": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Replacement source for each duplicate block, "
                    "in the same order as the input blocks. "
                    "Each replacement must preserve the original block's "
                    "leading indentation and end with a trailing newline. "
                    "Cover only the exact lines of the specified block — "
                    "do not include any code from before or after the block."
                ),
            },
        },
        "required": [
            "function_name",
            "placement",
            "helper_source",
            "call_site_replacements",
        ],
    },
}

_CALL_GEN_TOOL: dict = {
    "name": "generate_call",
    "description": "Generate a call to an existing function that replaces a code block",
    "input_schema": {
        "type": "object",
        "properties": {
            "replacement": {
                "type": "string",
                "description": (
                    "Complete replacement source "
                    "(including indentation and trailing newline)"
                ),
            }
        },
        "required": ["replacement"],
    },
}
