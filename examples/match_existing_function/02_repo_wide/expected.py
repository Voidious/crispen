# After repo-wide matching: build_dashboard_widget's body is structurally
# identical to _summarize_pending_report(), already defined elsewhere in
# the repo (billing/helpers.py) -- the block is replaced by a call to it,
# with an import added. (Exact whitespace around the inserted import may
# vary slightly.)
from billing.helpers import _summarize_pending_report


def build_dashboard_widget():
    return _summarize_pending_report()
