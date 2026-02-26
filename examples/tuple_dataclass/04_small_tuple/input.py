# A 3-element tuple is below the default min_size=4 threshold — crispen skips it.
def _get_status(code):
    message = STATUS_MESSAGES.get(code, "unknown")
    severity = SEVERITY_MAP.get(code, 0)
    return (message, severity, code)
