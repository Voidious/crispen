"""FileLimiter runner: orchestrates phases 1–4 for a single file."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..config import CrispenConfig
from .advisor import GroupPlacement, advise_file_limiter
from .classifier import classify_entities
from .code_gen import SplitResult, generate_file_splits
from .entity_parser import Entity, EntityKind


@dataclass
class FileLimiterResult:
    """Output of :func:`run_file_limiter` for a single file."""

    original_source: str  # updated source for the original file
    new_files: Dict[str, str]  # {relative_path: source_code}
    messages: List[str] = field(default_factory=list)
    abort: bool = False


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _verify_preservation(
    entities: List[Entity],
    split: SplitResult,
    post_source: str,
    placements: List[GroupPlacement],
) -> List[str]:
    """Return a list of failure descriptions (empty = all entities preserved).

    Checks that each entity's source text from *post_source* is present in
    either ``split.original_source`` or one of ``split.new_files.values()``.
    Empty entity sources (e.g. blank-line blocks) are skipped.
    TOP_LEVEL entities (import/docstring blocks) are always skipped because
    they are intentionally restructured during a split.
    Each failure is annotated with where the entity was expected to appear:
    ``migrated → <target>`` or ``stayed in original``.
    """
    lines = post_source.splitlines(keepends=True)
    combined = split.original_source + "".join(split.new_files.values())
    name_to_file: Dict[str, str] = {
        name: p.target_file for p in placements for name in p.group
    }
    failures: List[str] = []

    for entity in entities:
        if entity.kind == EntityKind.TOP_LEVEL:
            continue  # import/docstring blocks are intentionally restructured
        entity_src = "".join(lines[entity.start_line - 1 : entity.end_line]).rstrip()
        if entity_src and entity_src not in combined:
            preview_lines = entity_src.splitlines()[:3]
            preview = "\n    ".join(preview_lines)
            if len(entity_src.splitlines()) > 3:
                preview += "\n    ..."
            target = name_to_file.get(entity.name)
            loc = f"migrated \u2192 {target}" if target else "stayed in original"
            failures.append(
                f"  entity {entity.name!r} ({entity.kind.value},"
                f" lines {entity.start_line}\u2013{entity.end_line}) [{loc}]:\n"
                f"    {preview}"
            )

    return failures


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_file_limiter(
    filepath: str,
    original_source: str,
    post_source: str,
    diff_ranges: List[Tuple[int, int]],
    config: CrispenConfig,
) -> FileLimiterResult:
    """Run all FileLimiter phases on a single file.

    1. Classify entities (Phases 1+2).
    2. Ask the LLM for a placement plan (Phase 3).
    3. Generate new file contents (Phase 4).
    4. Verify all entity sources are preserved.

    Returns a :class:`FileLimiterResult` with ``abort=True`` when the file
    cannot be split or verification fails.  :class:`CrispenAPIError` from
    the LLM is propagated to the caller.
    """
    classified = classify_entities(original_source, post_source, diff_ranges)
    plan = advise_file_limiter(classified, filepath, config)

    if plan.abort:
        return FileLimiterResult(
            original_source=post_source,
            new_files={},
            messages=[f"SKIP {filepath} (FileLimiter): file cannot be split"],
            abort=True,
        )

    if not plan.placements:
        return FileLimiterResult(
            original_source=post_source,
            new_files={},
            abort=False,
        )

    split = generate_file_splits(classified, plan, post_source, filepath)

    failures = _verify_preservation(
        classified.entities, split, post_source, plan.placements
    )
    if failures:
        detail = "\n".join(failures)
        return FileLimiterResult(
            original_source=post_source,
            new_files={},
            messages=[f"SKIP {filepath} (FileLimiter): verification failed\n{detail}"],
            abort=True,
        )

    msgs = [
        f"{filepath}: FileLimiter: moved {', '.join(p.group)} \u2192 {p.target_file}"
        for p in plan.placements
    ]

    return FileLimiterResult(
        original_source=split.original_source,
        new_files=split.new_files,
        messages=msgs,
        abort=False,
    )
