# See a_expected.py -- same helper, same import, different call site.
from svc.common import normalize_customer_ref


def process_invoice(payload):
    return normalize_customer_ref(payload)
