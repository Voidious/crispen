from __future__ import annotations
from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import FileLimiterPlan
from crispen.file_limiter.runner import run_file_limiter
from .helpers import (
    _PATCH_ADVISE,
    _PATCH_CLASSIFY,
    _PATCH_GEN,
    _classified_with_groups,
    _empty_plan,
    _good_split,
    _make_classified,
    _make_entity,
    _plan_with,
)


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_plan_abort_retries_and_succeeds(mock_classify, mock_advise, mock_gen):
    # retries=1: first plan.abort (placement failure), second succeeds.
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.side_effect = [
        FileLimiterPlan(
            set3_migrate=[],
            placements=[],
            abort=True,
            abort_reason="LLM failed to assign file placements",
        ),
        _plan_with(["foo"], "utils.py"),
    ]
    mock_gen.return_value = _good_split()
    cfg = CrispenConfig(file_limiter_retries=1)

    result = run_file_limiter("big.py", "", "def foo():\n    pass\n", [], cfg)

    assert result.abort is False
    assert mock_advise.call_count == 2
    # Failed attempt message is preserved alongside the success message.
    assert any("cannot be split" in m for m in result.messages)
    assert any("FileLimiter: moved" in m for m in result.messages)
    # Feedback was forwarded on the second call.
    assert mock_advise.call_args_list[1].kwargs["prev_placement_failure"] != ""


@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_no_migration_retries_and_fails(mock_classify, mock_advise):
    # retries=1: both attempts → no entities selected → 2 SKIP msgs, abort=False.
    mock_classify.return_value = _classified_with_groups()
    mock_advise.return_value = _empty_plan()
    cfg = CrispenConfig(file_limiter_retries=1)

    result = run_file_limiter("big.py", "", "def foo():\n    pass\n", [], cfg)

    assert result.abort is False
    assert mock_advise.call_count == 2
    assert sum(1 for m in result.messages if "no entities selected" in m) == 2


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_no_migration_retries_and_succeeds(mock_classify, mock_advise, mock_gen):
    # retries=1: first attempt → no migration; second → placements and success.
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _classified_with_groups(entities=[entity])
    mock_advise.side_effect = [_empty_plan(), _plan_with(["foo"], "utils.py")]
    mock_gen.return_value = _good_split()
    cfg = CrispenConfig(file_limiter_retries=1)

    result = run_file_limiter("big.py", "", "def foo():\n    pass\n", [], cfg)

    assert result.abort is False
    assert mock_advise.call_count == 2
    # Failed attempt message is preserved alongside the success message.
    assert any("no entities selected" in m for m in result.messages)
    assert any("FileLimiter: moved" in m for m in result.messages)
    # Feedback about all-stay was forwarded on the second call.
    assert mock_advise.call_args_list[1].kwargs["prev_set3_failure"] != ""
