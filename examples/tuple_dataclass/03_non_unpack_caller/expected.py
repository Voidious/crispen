def _get_bounds(data):
    lo = min(data)
    hi = max(data)
    mean = sum(data) / len(data)
    spread = hi - lo
    return (lo, hi, mean, spread)


def print_summary(data):
    bounds = _get_bounds(data)  # stored whole, not unpacked
    print(f"Data bounds: {bounds}")
