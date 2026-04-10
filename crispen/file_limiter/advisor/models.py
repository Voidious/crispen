from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class _LLMAccumulator:
    """Mutable accumulator for LLM call counts, timing, and token usage."""

    calls: int = 0
    elapsed: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class GroupPlacement:
    """Placement decision for one SCC group."""

    group: List[str]  # entity names in the SCC
    target_file: str  # relative filename (e.g. "utils.py")


@dataclass
class FileLimiterPlan:
    """Complete placement plan from the LLM advisor."""

    # Set 3 groups the LLM chose to migrate (rest stay in original file).
    set3_migrate: List[List[str]]
    # Placement for set_2 groups + migrating set_3 groups.
    placements: List[GroupPlacement]
    # True if planning failed and the file should not be split.
    abort: bool
    abort_reason: str = ""  # human-readable explanation when abort=True
    llm_calls: int = 0  # number of LLM API calls made during planning
    llm_elapsed: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0


_SET3_TOOL: dict = {
    "name": "advise_set3_actions",
    "description": (
        "For each modified-entity group, decide whether to migrate it to a new "
        "file or leave it in the original file."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group_id": {
                            "type": "integer",
                            "description": "0-based index of the group",
                        },
                        "action": {
                            "type": "string",
                            "enum": ["migrate", "stay"],
                            "description": (
                                "'migrate' to move to a new file, "
                                "'stay' to keep in original"
                            ),
                        },
                    },
                    "required": ["group_id", "action"],
                },
            }
        },
        "required": ["decisions"],
    },
}

_PLACEMENT_TOOL: dict = {
    "name": "assign_file_placements",
    "description": (
        "Assign each entity group to a target Python filename. "
        "Each group will be written to a new file in the same directory "
        "as the original."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "placements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group_id": {
                            "type": "integer",
                            "description": "0-based index of the group",
                        },
                        "target_file": {
                            "type": "string",
                            "description": (
                                "Relative filename, e.g. 'utils.py' or "
                                "'helpers/io.py'"
                            ),
                        },
                    },
                    "required": ["group_id", "target_file"],
                },
            }
        },
        "required": ["placements"],
    },
}

_RENAME_CONFLICTS_TOOL: dict = {
    "name": "rename_conflicting_placements",
    "description": (
        "Assign new, non-conflicting target filenames to entity groups "
        "that currently have naming conflicts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "placements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group_id": {"type": "integer"},
                        "target_file": {"type": "string"},
                    },
                    "required": ["group_id", "target_file"],
                },
            }
        },
        "required": ["placements"],
    },
}

_PROPOSE_FILES_TOOL: dict = {
    "name": "propose_output_files",
    "description": (
        "Propose the set of Python files to create when splitting a large module. "
        "Return exactly the files you plan to use, with descriptive names and "
        "a brief summary of what each file will contain."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Python filename, e.g. 'utils.py'",
                        },
                        "description": {
                            "type": "string",
                            "description": "What this file will contain",
                        },
                    },
                    "required": ["filename", "description"],
                },
            }
        },
        "required": ["files"],
    },
}


# Maximum number of groups per placement LLM call.  Large files may have
# dozens of groups; sending them all in one call frequently causes timeouts
# or incomplete responses.  This limit keeps each call small and reliable.
_PLACEMENT_CHUNK_SIZE = 100
