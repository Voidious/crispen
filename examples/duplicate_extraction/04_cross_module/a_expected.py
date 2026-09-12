# After cross-module extraction: the shared 3-statement block becomes a
# helper in svc/common.py, imported by every call site. (Exact whitespace
# around the inserted import may vary slightly.)
from svc.common import normalize_customer_ref


def process_order(payload):
    return normalize_customer_ref(payload)
