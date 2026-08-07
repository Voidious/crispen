"""CLI entry point: reads stdin diff, drives engine, reports to stdout."""

import sys
import time

from .config import load_config
from .diff_parser import parse_diff
from .engine import run_engine
from .errors import CrispenAPIError
from .stats import RunStats


def main() -> None:
    # Some progress/summary lines contain non-ASCII characters (e.g. "→").
    # On Windows the default console codepage (cp1252) can't encode these,
    # which raises UnicodeEncodeError and crashes mid-run after real LLM
    # calls have already been made. Replacing unencodable characters instead
    # of crashing costs nothing on platforms (Linux/macOS) that already
    # default to a UTF-8 stdout/stderr.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")

    diff_text = sys.stdin.read()
    if not diff_text.strip():
        print("crispen: no diff provided on stdin", file=sys.stderr)
        sys.exit(1)

    changed = parse_diff(diff_text)
    if not changed:
        return

    config = load_config()
    run_stats = RunStats()
    _t0 = time.perf_counter()
    try:
        for message in run_engine(changed, config=config, stats=run_stats):
            print(message)
    except CrispenAPIError as exc:
        print(f"crispen: {exc}", file=sys.stderr)
        sys.exit(1)
    run_stats.total_elapsed = time.perf_counter() - _t0
    if run_stats.total_llm_calls > 0 or run_stats.total_edits > 0:
        for line in run_stats.format_summary(timing=config.timing):
            print(line)
