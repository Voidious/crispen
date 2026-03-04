import textwrap
from unittest.mock import MagicMock
from crispen.refactors.duplicate_extractor import DuplicateExtractor
from tests.duplicate_extractor_test_responses import (
    _make_extract_response,
    _make_veto_response,
)


def _setup_anthropic_veto_then_extract_extractor(
    mock_anthropic,
    *,
    verbose: bool,
    dup_ranges,
    dup_source: str,
    extraction_retries: int = 0,
    llm_verify_retries: int = 0,
):
    mock_client = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    mock_anthropic.APIError = Exception
    mock_client.messages.create.side_effect = [
        _make_veto_response(True, "same logic"),
        _make_extract_response(
            {
                "function_name": "_helper",
                "placement": "module_level",
                "helper_source": "def _helper(x):\n    pass\n",
                "call_site_replacements": [
                    "    _helper(data)\n",
                    "    _helper(data)\n",
                ],
            }
        ),
    ]
    return DuplicateExtractor(
        dup_ranges,
        source=dup_source,
        verbose=verbose,
        extraction_retries=extraction_retries,
        llm_verify_retries=llm_verify_retries,
    )


_POST_STEAL_SOURCE = textwrap.dedent(
    """\
    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        return z

    def bar():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        logger.info("done")
    """
)
_POST_STEAL_RANGES = [(8, 10)]  # overlaps bar's 3-statement block
