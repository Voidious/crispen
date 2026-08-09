def process_invoice(payload):
    ref = payload["customer_ref"]
    normalized = ref.strip().upper()
    tag = normalized.replace("-", "")
    return tag
