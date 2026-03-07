from __future__ import annotations
from crispen.config import CrispenConfig
from crispen.file_limiter.classifier import ClassifiedEntities
from .runner_plans_and_splits import _make_entity


_CONFIG = CrispenConfig()
# Zero-retry config for tests that exercise a single-attempt failure path.
_CONFIG_NO_RETRY = CrispenConfig(file_limiter_retries=0)
_PATCH_CLASSIFY = "crispen.file_limiter.runner.classify_entities"
_PATCH_ADVISE = "crispen.file_limiter.runner.advise_file_limiter"
_PATCH_GEN = "crispen.file_limiter.runner.generate_file_splits"
_PATCH_RESOLVE = "crispen.file_limiter.runner.resolve_naming_conflicts"


def _make_classified(entities=None) -> ClassifiedEntities:
    return ClassifiedEntities(
        entities=entities or [],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[],
        set_3_groups=[],
        abort=False,
    )


def _classified_with_groups(entities=None) -> ClassifiedEntities:
    """Classified result with non-empty set_3_groups (triggers LLM advise)."""
    ents = entities or [_make_entity("foo", 1, 2)]
    return ClassifiedEntities(
        entities=ents,
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=[],
        set_3_groups=[[e.name for e in ents]],
        abort=False,
    )


# A two-line source whose diff_ranges covers the whole file, triggering subdir
# split for "big.py" → subdir_name="big".  Path("big") must not exist on disk.
_SUBDIR_SRC = "x = 1\ny = 2\n"
_SUBDIR_RANGES = [(1, 2)]
