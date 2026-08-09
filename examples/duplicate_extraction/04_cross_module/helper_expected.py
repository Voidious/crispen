# The new shared module crispen creates at the common ancestor package
# (svc/) of both call sites.
def normalize_customer_ref(payload):
    ref = payload["customer_ref"]
    normalized = ref.strip().upper()
    tag = normalized.replace("-", "")
    return tag
